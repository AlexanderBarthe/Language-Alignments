from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CONFIG
from data_structures.models import ScoreMatrix, TracebackMatrix, WordTuple, DistanceMatrix, ScoringParams, LexstatMatrix
from data_structures.phonemes import tokenize
from msa import profile_alignment_algorithm
from msa.global_lexstat_model import GlobalLexstatModel

MAX_WORKERS = CONFIG['alignment']['mt_workers']


def _align_worker(task):
    i, j, word_i, word_j, custom_params, lexstat_matrix, lexstat_model = task
    score, distance, _, _ = evaluate_single(word_i.form, word_j.form, custom_params, lexstat_matrix,
                                          lang_a=word_i.language, lang_b=word_j.language, lexstat_model=lexstat_model)
    return i, j, score, distance


def _get_model(lexstat_matrix: LexstatMatrix | None, lexstat_model: GlobalLexstatModel | None,
               lang_a: str, lang_b: str) -> GlobalLexstatModel | None:
    if lexstat_matrix is not None:
        if lexstat_model is not None:
            raise ValueError("provide either a LexStat matrix or a global model, not both")
        lexstat_model = GlobalLexstatModel()
        lexstat_model.add_matrix(lang_a, lang_b, lexstat_matrix)
    return lexstat_model


def evaluate_single_semiglobally(seq1: str | Sequence[str], seq2: str | Sequence[str],
                                custom_params: ScoringParams | None = None, lexstat_matrix: LexstatMatrix | None = None,
                                *, lang_a: str = "word_a", lang_b: str = "word_b",
                                lexstat_model: GlobalLexstatModel | None = None) \
        -> tuple[float, int, int, ScoreMatrix, TracebackMatrix]:
    model = _get_model(lexstat_matrix, lexstat_model, lang_a, lang_b)
    _, _, fs_score, fs_matrix, fs_traceback = profile_alignment_algorithm.align(
        seq1, lang_a, seq2, lang_b, True, False, custom_params, model)
    _, _, fe_score, fe_matrix, fe_traceback = profile_alignment_algorithm.align(
        seq1, lang_a, seq2, lang_b, False, True, custom_params, model)
    if fs_score > fe_score:
        return fs_score, len(fs_matrix) - 1, len(fs_matrix[0]) - 1, fs_matrix, fs_traceback
    _, fe_i, fe_j = profile_alignment_algorithm.get_final_absolute_score(True, fe_matrix)
    return fe_score, fe_i, fe_j, fe_matrix, fe_traceback


def evaluate_single(seq1: str | Sequence[str], seq2: str | Sequence[str], custom_params: ScoringParams | None = None,
                    lexstat_matrix: LexstatMatrix | None = None, *, lang_a: str = "word_a", lang_b: str = "word_b",
                    lexstat_model: GlobalLexstatModel | None = None) -> tuple[float, float, ScoreMatrix, TracebackMatrix]:
    model = _get_model(lexstat_matrix, lexstat_model, lang_a, lang_b)
    _, _, score, matrix, traceback = profile_alignment_algorithm.align(
        seq1, lang_a, seq2, lang_b, False, False, custom_params, model)
    return score, profile_alignment_algorithm.score_to_distance(score), matrix, traceback


def find_best_match(seq1: str | Sequence[str], match_partners: Sequence[str | Sequence[str]],
                    custom_params: ScoringParams | None = None, lexstat_matrix: LexstatMatrix | None = None,
                    *, lang_a: str = "word_a", lang_b: str = "word_b", lexstat_model: GlobalLexstatModel | None = None) \
        -> tuple[str | Sequence[str] | None, float, ScoreMatrix | None, TracebackMatrix | None, int]:
    best_score = float("-inf")
    best_match = None
    best_alignment = None
    best_traceback = None
    for match_partner in match_partners:
        score, _, alignment, traceback = evaluate_single(seq1, match_partner, custom_params, lexstat_matrix,
                                                         lang_a=lang_a, lang_b=lang_b, lexstat_model=lexstat_model)
        if score > best_score:
            best_score = score
            best_match = match_partner
            best_alignment = alignment
            best_traceback = traceback
    return best_match, best_score, best_alignment, best_traceback, len(match_partners)


def match_every_to_distance(sequences: list[WordTuple], custom_params: ScoringParams | None = None,
                            lexstat_matrix: LexstatMatrix | None = None, *,
                            lexstat_model: GlobalLexstatModel | None = None) -> DistanceMatrix:
    sequences = [WordTuple(word.language, word.concept, tokenize(word.form)) for word in sequences]
    n = len(sequences)
    distance_matrix = np.zeros((n, n), dtype='float16')
    tasks = [(i, j, sequences[i], sequences[j], custom_params, lexstat_matrix, lexstat_model)
             for i in range(n) for j in range(i + 1, n)]
    if tasks:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(_align_worker, tasks, chunksize=1000)
            for i, j, _, dist in tqdm(iterator, total=len(tasks), desc="Calculating Alignments"):
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    multi_index = pd.MultiIndex.from_tuples(sequences, names=["Language", "Concept", "Form"])
    return pd.DataFrame(distance_matrix, index=multi_index, columns=multi_index)


def show_top_matches(sequences: list[WordTuple], top_n: int):
    df_matrix = match_every_to_distance(sequences)
    upper_triangle_mask = np.triu(np.ones(df_matrix.shape), k=1).astype(bool)
    row_indices, col_indices = np.where(upper_triangle_mask)
    pairs_df = pd.DataFrame({'row_idx': row_indices, 'col_idx': col_indices,
                             'distance': df_matrix.values[upper_triangle_mask]})
    top_pairs = pairs_df.nsmallest(top_n, 'distance') if top_n > 0 else pairs_df.nlargest(-top_n, 'distance')
    for _, row in top_pairs.iterrows():
        lang_i, concept_i, form_i = df_matrix.index[int(row['row_idx'])]
        lang_j, concept_j, form_j = df_matrix.columns[int(row['col_idx'])]
        profile, _, _, _, _ = profile_alignment_algorithm.align(form_i, lang_i, form_j, lang_j, False, False)
        print("##############################")
        print(f"Distance: {row['distance']:.4f}")
        print(f"Word 1: {' '.join(form_i)}, Language: {lang_i}, Concept: {concept_i}")
        print(f"Word 2: {' '.join(form_j)}, Language: {lang_j}, Concept: {concept_j}")
        print(profile)