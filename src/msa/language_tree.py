import random
from itertools import combinations

import pandas as pd

from cldf_repo import CLDFRepository
from clustering import neighbor_joining
from config import CONFIG
from data_structures.guide_tree import TreeNode
from data_structures.models import WordForm
from data_structures.profile import Profile
from lexstat_scores import get_lexstat_score, _get_lexstat_cache_context
from msa import match_evaluator, profile_alignment_algorithm
from msa.global_lexstat_model import GlobalLexstatModel
from src.matrix_cache import PairMatrixCache

MAX_WORD_PAIRS_FOR_DISTANCE = 10000
USABLE_CONCEPT_LANGUAGE_RATIO = 0.9
LANGUAGE_DISTANCE_CALCULATION_VERSION = "language-distance-pair-v1"


def build(
    cldf: CLDFRepository,
    concepts: list[str],
    load_from_cache: bool = True,
    store_to_cache: bool = False,
) -> tuple[list[Profile], TreeNode]:
    lexstat_model = build_lexstat_matrices(cldf, load_from_cache, store_to_cache)
    lang_dist = build_language_distance_matrix(
        cldf, lexstat_model, False, load_from_cache, store_to_cache
    )

    tree = neighbor_joining.build_tree_nj(lang_dist)

    profiles = []

    for concept in concepts:
        profiles.append(get_profile(tree, cldf, lexstat_model, concept))

    return profiles, tree

def build_lexstat_matrices(
    cldf: CLDFRepository,
    load_from_cache: bool = True,
    store_to_cache: bool = False,
) -> GlobalLexstatModel:
    model = GlobalLexstatModel()

    language_list = sorted(cldf.get_all_language_names())
    cache, signature = (None, None)
    if load_from_cache or store_to_cache:
        cache, signature = _get_lexstat_cache_context(cldf)

    for language in language_list:
        for language2 in language_list:
            if language == language2 or model.has_matrix(language, language2):
                continue

            print("Building Lexstat", language, "to", language2)

            lexstat_matrix = get_lexstat_score(
                cldf,
                language,
                language2,
                load_from_cache,
                store_to_cache,
                _cache=cache,
                _signature=signature,
            )
            model.add_matrix(language, language2, lexstat_matrix)

    return model

def print_lexstat_matrices(cldf: CLDFRepository) -> GlobalLexstatModel:

    model = build_lexstat_matrices(cldf)
    for language, language2 in combinations(sorted(cldf.get_all_language_names()), 2):
        matrix = model.get_matrix(language, language2)
        print(f"\nLexStat matrix: {language} -> {language2}")
        if not matrix:
            print("(empty matrix)")
            continue

        rows = sorted({pair[0] for pair in matrix})
        columns = sorted({pair[1] for pair in matrix})
        table = pd.DataFrame(
            [[matrix.get((row, column), 0.0) for column in columns] for row in rows],
            index=rows, columns=columns)
        print(table.to_string(float_format="{:.2f}".format))

    return model

def build_language_distance_matrix(
    cldf: CLDFRepository,
    lexstat_model: GlobalLexstatModel,
    random_concept_sample: bool = False,
    load_from_cache: bool = True,
    store_to_cache: bool = False,
    random_state: int | str = 101
) -> pd.DataFrame:
    language_list = sorted(cldf.get_all_language_names())
    language_distances = pd.DataFrame(0.0, index=language_list, columns=language_list)
    if load_from_cache or store_to_cache:
        cache, base_signature = _get_language_distance_cache_context(cldf)
    else:
        cache, base_signature = None, ""

    sampled_concepts = None
    if random_concept_sample:
        sampled_concepts = _get_bootstrap_concept_sample(cldf, random_state)

    for lang1, lang2 in combinations(language_list, 2):
        pair_signature = None
        if cache is not None:
            assert base_signature is not None
            lexstat_matrix = lexstat_model.get_matrix(lang1, lang2)
            pair_signature = _language_distance_pair_signature(cache, base_signature, lexstat_matrix)
            if not random_concept_sample and load_from_cache and not store_to_cache:
                cached_distance = cache.load_distance(lang1, lang2, pair_signature)
                if cached_distance is not None:
                    language_distances.at[lang1, lang2] = cached_distance
                    language_distances.at[lang2, lang1] = cached_distance
                    continue

        avg_dist = _calculate_language_pair_distance(
            cldf, lexstat_model, lang1, lang2,
            cache, pair_signature, load_from_cache, store_to_cache, sampled_concepts
        )
        if not random_concept_sample and store_to_cache:
            assert cache is not None and pair_signature is not None
            cache.store_distance(lang1, lang2, avg_dist, pair_signature)
        language_distances.at[lang1, lang2] = avg_dist
        language_distances.at[lang2, lang1] = avg_dist

    return language_distances


def _get_language_distance_cache_context(cldf: CLDFRepository) -> tuple[PairMatrixCache, str]:
    cache = PairMatrixCache()
    signature = cache.signature_for(
        cldf,
        settings={"config": CONFIG, "max_word_pairs": MAX_WORD_PAIRS_FOR_DISTANCE},
        calculation_version=LANGUAGE_DISTANCE_CALCULATION_VERSION,
    )
    return cache, signature


def _get_bootstrap_concept_sample(cldf: CLDFRepository, random_state: int | str = 101) -> list[str]:
    usable_concepts = cldf.get_usable_concepts()

    # local instance: no mutation of the global RNG state
    rng = random.Random(random_state)

    return rng.choices(usable_concepts, k=len(usable_concepts))


def _language_distance_pair_signature(
    cache: PairMatrixCache,
    base_signature: str,
    lexstat_matrix: dict[tuple[str, str], float] | None,
) -> str:
    dependency = None if lexstat_matrix is None else [
        [sound_a, sound_b, score]
        for (sound_a, sound_b), score in sorted(lexstat_matrix.items())
    ]
    return cache.signature_with_dependency(base_signature, dependency)


def _calculate_language_pair_distance(
    cldf: CLDFRepository,
    lexstat_model: GlobalLexstatModel,
    lang1: str,
    lang2: str,
    cache: PairMatrixCache | None = None,
    pair_signature: str | None = None,
    load_from_cache: bool = False,
    store_to_cache: bool = False,
    sampled_concepts: list[str] | None = None,
) -> float:
    lang1_id = cldf.find_language_id(lang1)
    lang2_id = cldf.find_language_id(lang2)
    if lang1_id is None or lang2_id is None:
        return float("inf")

    word_tuples = cldf.get_same_meaning_pairs_as_tuples(
        lang1_id, lang2_id, sampled_concepts, MAX_WORD_PAIRS_FOR_DISTANCE
    )

    cached_distances: dict[tuple[WordForm, WordForm], float] = {}
    if cache is not None and pair_signature is not None and load_from_cache:
        cached_distances = cache.load_alignments(lang1, lang2, pair_signature) or {}

    # starts from the cache so computed values are reused for duplicate
    # sampled concepts within this call
    word_distances: dict[tuple[WordForm, WordForm], float] = dict(cached_distances)
    accu_dist = 0.0
    count = 0
    for word_tuple in word_tuples:
        word1 = word_tuple[0].form
        word2 = word_tuple[1].form
        word_pair = (word1, word2)
        distance = word_distances.get(word_pair)
        if distance is None:
            _, distance, _, _ = match_evaluator.evaluate_single(
                word1, word2, lang_a=lang1, lang_b=lang2, lexstat_model=lexstat_model
            )
            word_distances[word_pair] = distance
        accu_dist += distance
        count += 1

    if cache is not None and pair_signature is not None and store_to_cache \
            and len(word_distances) > len(cached_distances):
        cache.store_alignments(lang1, lang2, word_distances, pair_signature)

    return accu_dist / count if count > 0 else float("inf")

def get_profile(tree_node: TreeNode, cldf: CLDFRepository, lexstat_model: GlobalLexstatModel, concept: str) -> Profile:

    if tree_node.is_leaf():
        lang_name = tree_node.name
        word = cldf.find_word(lang_name, concept)
        if word is None:
            return Profile.from_single_word((), lang_name)
        form = word.form
        return Profile.from_single_word(form, lang_name)

    left_profile = get_profile(tree_node.left, cldf, lexstat_model, concept)
    right_profile = get_profile(tree_node.right, cldf, lexstat_model, concept)

    profile, _, _, _, _ = profile_alignment_algorithm.align_profiles(left_profile, right_profile, False, False, None, lexstat_model)

    return profile
