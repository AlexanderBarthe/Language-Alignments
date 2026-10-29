from itertools import combinations

import pandas as pd
from pycldf import Dataset

from config import CONFIG
import cldf_repo
from cldf_repo import CLDFRepository
from clustering import neighbor_joining
from data_structures.guide_tree import TreeNode
from data_structures.profile import Profile
from lexstat_scores import get_lexstat_score, _get_lexstat_cache_context
from msa import match_evaluator, profile_alignment_algorithm
from msa.global_lexstat_model import GlobalLexstatModel
from src.matrix_cache import PairMatrixCache

MAX_WORD_PAIRS_FOR_DISTANCE = 10000
LANGUAGE_DISTANCE_CALCULATION_VERSION = "language-distance-pair-v1"


def build(
    ds: Dataset,
    concept: str,
    load_from_cache: bool = True,
    store_to_cache: bool = False,
) -> tuple[Profile, TreeNode]:
    cldf = cldf_repo.CLDFRepository(ds)
    lexstat_model = build_lexstat_matrices(cldf, load_from_cache, store_to_cache)
    lang_dist = build_language_distance_matrix(
        cldf, lexstat_model, load_from_cache, store_to_cache
    )

    tree = neighbor_joining.build_tree_nj(lang_dist)

    profile = get_profile(tree, cldf, lexstat_model, concept)

    return profile, tree

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
    load_from_cache: bool = True,
    store_to_cache: bool = False,
) -> pd.DataFrame:
    language_list = sorted(cldf.get_all_language_names())
    language_distances = pd.DataFrame(0.0, index=language_list, columns=language_list)
    if load_from_cache or store_to_cache:
        cache, base_signature = _get_language_distance_cache_context(cldf)
    else:
        cache, base_signature = None, ""

    for lang1, lang2 in combinations(language_list, 2):
        pair_signature = None
        if cache is not None:
            assert base_signature is not None
            lexstat_matrix = lexstat_model.get_matrix(lang1, lang2)
            pair_signature = _language_distance_pair_signature(cache, base_signature, lexstat_matrix)
            if load_from_cache and not store_to_cache:
                cached_distance = cache.load_distance(lang1, lang2, pair_signature)
                if cached_distance is not None:
                    language_distances.at[lang1, lang2] = cached_distance
                    language_distances.at[lang2, lang1] = cached_distance
                    continue

        print("Determining distance from", lang1, "to", lang2)
        avg_dist = _calculate_language_pair_distance(cldf, lexstat_model, lang1, lang2)
        if store_to_cache:
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
) -> float:
    lang1_id = cldf.find_language_id(lang1)
    lang2_id = cldf.find_language_id(lang2)
    if lang1_id is None or lang2_id is None:
        return float("inf")
    word_tuples = cldf.get_same_meaning_pairs_as_tuples(
        lang1_id, lang2_id, MAX_WORD_PAIRS_FOR_DISTANCE
    )

    accu_dist = 0.0
    count = 0
    for word_tuple in word_tuples:
        word1 = word_tuple[0].form
        word2 = word_tuple[1].form
        _, dist, _, _ = match_evaluator.evaluate_single(
            word1, word2, lang_a=lang1, lang_b=lang2, lexstat_model=lexstat_model
        )
        accu_dist += dist
        count += 1

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