import math
from collections import defaultdict

from lingpy import Model
from pycldf import Dataset

from config import CONFIG
from data_structures.phonemes import sound_class
from msa import profile_alignment_algorithm, match_evaluator
from src import cldf_repo
from src.cldf_repo import CLDFRepository
from src.data_structures.models import WordTuple, LexstatMatrix

model = Model(CONFIG['alignment']['model'])
epsilon = 1e-5

def get_lexstat_score(data: Dataset | CLDFRepository, lang_name_a: str, lang_name_b: str) -> LexstatMatrix:

    if isinstance(data, Dataset):
        cldf = cldf_repo.CLDFRepository(data)
    else:
        cldf = data

    expected_dist = calculate_expected_distribution(cldf, lang_name_a, lang_name_b)
    attested_dist = calculate_attested_distribution(cldf, lang_name_a, lang_name_b)

    lexstat_scores = calculate_lexstat_scoring_matrix(attested_dist, expected_dist)

    return lexstat_scores

def calculate_expected_distribution(cldf: CLDFRepository, lang_name_a: str, lang_name_b: str) \
        -> LexstatMatrix:

    words_lang_a = cldf.get_words_for_language_as_tuples(lang_name_a)
    words_lang_b = cldf.get_words_for_language_as_tuples(lang_name_b)

    noise_samples = CLDFRepository.get_noise_sample_for_language_pair(words_lang_a, words_lang_b)

    return calculate_distribution(noise_samples)

def calculate_attested_distribution(cldf: CLDFRepository, lang_name_a: str, lang_name_b: str) \
        -> LexstatMatrix:

    lang_id_a = cldf.find_language_id(lang_name_a)
    lang_id_b = cldf.find_language_id(lang_name_b)

    if lang_id_a is None or lang_id_b is None:
        return {}
    same_word_tuples = cldf.get_same_meaning_pairs_as_tuples(lang_id_a, lang_id_b)
    return calculate_distribution(same_word_tuples)


def calculate_lexstat_scoring_matrix(attested_dist: LexstatMatrix, expected_dist: LexstatMatrix) -> LexstatMatrix:
    lexstat_matrix = {}

    all_pairs = attested_dist.keys() | expected_dist.keys()

    for pair in all_pairs:
        f_attested = attested_dist.get(pair, 0.0)
        f_expected = expected_dist.get(pair, 0.0)

        ratio = (f_attested + epsilon) / (f_expected + epsilon)
        score = math.log2(ratio ** 2)

        s1, s2 = pair
        lexstat_matrix[(s1, s2)] = score
        lexstat_matrix[(s2, s1)] = score

    return lexstat_matrix

def calculate_distribution(samples: list[tuple[WordTuple, WordTuple]]) -> LexstatMatrix:

    expected_counts = defaultdict(int)
    total_expected_pairs = 0

    for sample in samples:

        seq1 = sample[0].form
        seq2 = sample[1].form
        _, _, _, traceback_matrix = match_evaluator.evaluate_single(
            seq1, seq2, lang_a=sample[0].language, lang_b=sample[1].language)

        matched_cognates = profile_alignment_algorithm.get_matched_seqs(traceback_matrix, seq1, seq2)

        for char1, char2 in matched_cognates:
            discrete_pairs = convert_to_discrete_pairs(char1, char2)

            for s1, s2 in discrete_pairs:
                if s1 == "-" or s2 == "-":
                    continue

                pair = (s1, s2)
                expected_counts[pair] += 1
                total_expected_pairs += 1

    distribution = {}

    if total_expected_pairs > 0:
        for pair, count in expected_counts.items():
            distribution[pair] = count / total_expected_pairs

    return distribution



def get_class(char: str):
    return sound_class(char, model)


def convert_to_discrete_pairs(char1, char2) -> list[tuple[str, str]]:
    tokens1 = (char1,) if isinstance(char1, str) else char1
    tokens2 = (char2,) if isinstance(char2, str) else char2
    return [(sound_class(c1, model), sound_class(c2, model)) for c1 in tokens1 for c2 in tokens2]