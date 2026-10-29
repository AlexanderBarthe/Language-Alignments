import math
from collections import defaultdict
from dataclasses import dataclass

from lingpy import Model
from pycldf import Dataset

from config import CONFIG
from data_structures.phonemes import sound_class
from msa import profile_alignment_algorithm, match_evaluator
from src import cldf_repo
from src.cldf_repo import CLDFRepository
from src.data_structures.models import WordTuple, LexstatMatrix
from src.matrix_cache import PairMatrixCache

model = Model(CONFIG['alignment']['model'])
ALPHA = CONFIG['lexstat'].get("alpha", 1.0)
LEXSTAT_CALCULATION_VERSION = "lexstat-pair-v1"


@dataclass
class CorrespondenceDistribution:
    counts: dict[tuple[str, str], int]
    probabilities: LexstatMatrix


def get_lexstat_score(
    data: Dataset | CLDFRepository,
    lang_name_a: str,
    lang_name_b: str,
    load_from_cache: bool = True,
    store_to_cache: bool = False,
    *,
    _cache: PairMatrixCache | None = None,
    _signature: str | None = None,
) -> LexstatMatrix:
    if isinstance(data, Dataset):
        cldf = cldf_repo.CLDFRepository(data)
    else:
        cldf = data

    if not load_from_cache and not store_to_cache:
        return _calculate_lexstat_score(cldf, lang_name_a, lang_name_b)

    if _cache is None or _signature is None:
        _cache, _signature = _get_lexstat_cache_context(cldf)

    if load_from_cache and not store_to_cache:
        cached_matrix = _cache.load_lexstat(lang_name_a, lang_name_b, _signature)
        if cached_matrix is not None:
            return cached_matrix

    lexstat_matrix = _calculate_lexstat_score(cldf, lang_name_a, lang_name_b)
    if store_to_cache:
        _cache.store_lexstat(lang_name_a, lang_name_b, lexstat_matrix, _signature)
    return lexstat_matrix


def _calculate_lexstat_score(cldf: CLDFRepository, lang_name_a: str, lang_name_b: str) -> LexstatMatrix:
    expected_dist = calculate_expected_distribution(cldf, lang_name_a, lang_name_b)
    attested_dist = calculate_attested_distribution(cldf, lang_name_a, lang_name_b)

    lexstat_scores = calculate_lexstat_scoring_matrix(attested_dist, expected_dist)

    return lexstat_scores


def _get_lexstat_cache_context(data) -> tuple[PairMatrixCache, str]:
    cache = PairMatrixCache()
    settings = {
        "config": CONFIG,
        "alpha": ALPHA,
        "model": CONFIG["alignment"]["model"],
        "model_converter": model.converter,
    }
    signature = cache.signature_for(
        data,
        settings=settings,
        calculation_version=LEXSTAT_CALCULATION_VERSION,
    )
    return cache, signature


def calculate_expected_distribution(cldf: CLDFRepository, lang_name_a: str, lang_name_b: str) \
        -> CorrespondenceDistribution:

    words_lang_a = cldf.get_words_for_language_as_tuples(lang_name_a)
    words_lang_b = cldf.get_words_for_language_as_tuples(lang_name_b)

    noise_samples = CLDFRepository.get_noise_sample_for_language_pair(words_lang_a, words_lang_b)

    return calculate_distribution(noise_samples)

def calculate_attested_distribution(cldf: CLDFRepository, lang_name_a: str, lang_name_b: str) \
        -> CorrespondenceDistribution:

    lang_id_a = cldf.find_language_id(lang_name_a)
    lang_id_b = cldf.find_language_id(lang_name_b)

    if lang_id_a is None or lang_id_b is None:
        return CorrespondenceDistribution({}, {})
    same_word_tuples = cldf.get_same_meaning_pairs_as_tuples(lang_id_a, lang_id_b)
    return calculate_distribution(same_word_tuples)


def calculate_lexstat_scoring_matrix(attested_dist: CorrespondenceDistribution,
                                    expected_dist: CorrespondenceDistribution) -> LexstatMatrix:
    """Keep unseen pairs neutral and score evidenced pairs in their own direction."""
    if not attested_dist.counts or not expected_dist.counts:
        return {}

    lexstat_matrix = {}

    all_pairs = attested_dist.probabilities.keys() | expected_dist.probabilities.keys()

    for pair in all_pairs:
        if attested_dist.counts.get(pair, 0) == 0 and expected_dist.counts.get(pair, 0) == 0:
            lexstat_matrix[pair] = 0.0
            continue

        f_attested = attested_dist.probabilities[pair]
        f_expected = expected_dist.probabilities[pair]

        ratio = (f_attested) / (f_expected)
        score = 2 * math.log2(ratio)

        lexstat_matrix[pair] = score

    return lexstat_matrix

def calculate_distribution(
    samples: list[tuple[WordTuple, WordTuple]]
) -> CorrespondenceDistribution:

    counts = defaultdict(int)
    total_pairs = 0

    classes = get_classes()
    class_set = set(classes)
    class_combination_amount = len(classes) ** 2

    for sample in samples:

        seq1 = sample[0].form
        seq2 = sample[1].form

        _, _, _, traceback_matrix = match_evaluator.evaluate_single(
            seq1,
            seq2,
            lang_a=sample[0].language,
            lang_b=sample[1].language
        )

        matched_pairs = profile_alignment_algorithm.get_matched_seqs(
            traceback_matrix, seq1, seq2
        )

        for char1, char2 in matched_pairs:
            discrete_pairs = convert_to_discrete_pairs(char1, char2)

            for s1, s2 in discrete_pairs:

                if s1 in {"-", "0"} or s2 in {"-", "0"}:
                    continue

                if s1 not in class_set or s2 not in class_set:
                    raise ValueError(
                        f"Unknown sound combination"
                        f"{s1!r}, {s2!r}"
                    )

                counts[(s1, s2)] += 1
                total_pairs += 1

    if total_pairs == 0:
        return CorrespondenceDistribution({}, {})

    denominator = (
        total_pairs + ALPHA * class_combination_amount
    )

    distribution = {
        (s1, s2): (
            counts.get((s1, s2), 0) + ALPHA
        ) / denominator
        for s1 in classes
        for s2 in classes
    }

    return CorrespondenceDistribution(dict(counts), distribution)



def get_class(char: str):
    return sound_class(char, model)

def get_classes() -> tuple[str, ...]:
    classes = {
        c for c in model.converter.values()
        if c not in {"0", "-"}
    }
    return tuple(sorted(classes))


def get_class_count() -> int:
    return len(get_classes())

def convert_to_discrete_pairs(
    char1, char2
) -> list[tuple[str, str]]:

    tokens1 = (char1,) if isinstance(char1, str) else char1
    tokens2 = (char2,) if isinstance(char2, str) else char2

    def to_class(token: str) -> str:
        if token == "-":
            return "-"
        return sound_class(token, model)

    return [
        (to_class(c1), to_class(c2))
        for c1 in tokens1
        for c2 in tokens2
    ]