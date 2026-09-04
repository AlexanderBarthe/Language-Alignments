from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CONFIG
from simple_alignment import alignment_algorithm, scores
from src.data_structures.models import ScoreMatrix, TracebackMatrix, WordTuple, DistanceMatrix, ScoringParams, \
    LexstatMatrix

MAX_WORKERS = CONFIG['alignment']['mt_workers']


def _align_worker(task):
    i, j, form_i, form_j, custom_params, lexstat_matrix = task

    score, distance, _, _, = evaluate_single(form_i, form_j, custom_params, lexstat_matrix)
    return i, j, score, distance

def evaluate_single_semiglobally(seq1: str, seq2: str, custom_params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None) -> tuple[float, int, int, ScoreMatrix, TracebackMatrix]:
    fs_score, dist, fs_i, fs_j, fs_matrix, fs_traceback = alignment_algorithm.align(seq1, seq2, True, False, custom_params, lexstat_matrix)

    fe_score, dist, fe_i, fe_j, fe_matrix, fe_traceback = alignment_algorithm.align(seq1, seq2, False, True, custom_params, lexstat_matrix)

    if fs_score > fe_score:
        return fs_score, fs_i, fs_j, fs_matrix, fs_traceback
    else:
        return fe_score, fe_i, fe_j, fe_matrix, fe_traceback

def evaluate_single(seq1: str, seq2: str, custom_params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None) -> tuple[float, float, ScoreMatrix, TracebackMatrix]:
    score, distance, _, _, score_matrix, traceback_matrix = alignment_algorithm.align(seq1, seq2, False, False, custom_params, lexstat_matrix)
    return score, distance, score_matrix, traceback_matrix

def find_best_match(seq1: str, match_partners: list[str], custom_params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None) -> tuple[
    str | None, float, list[list[float]] | None, list[list[str]] | None, int]:

    best_score = float("-inf")
    best_match = None
    best_alignment = None
    best_traceback = None
    comparisons = 0

    for match_partner in match_partners:
        score, distance, alignment, traceback = evaluate_single(seq1, match_partner, custom_params, lexstat_matrix)

        if score > best_score:
            best_score = score
            best_match = match_partner
            best_alignment = alignment
            best_traceback = traceback

        comparisons += 1

    return best_match, best_score, best_alignment, best_traceback, comparisons

def match_every_to_distance(sequences: list[WordTuple], custom_params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None) -> DistanceMatrix:

    n = len(sequences)
    distance_matrix = np.zeros((n, n), dtype='float16')

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, sequences[i].form, sequences[j].form, custom_params, lexstat_matrix))

    if not tasks:
        return pd.DataFrame(distance_matrix)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = executor.map(_align_worker, tasks, chunksize=1000)
        results = list(tqdm(iterator, total=len(tasks), desc="Calculating Alignments", disable=False))

    for i, j, score, dist in results:
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

    multi_index = pd.MultiIndex.from_tuples(sequences, names=["Language", "Concept", "Form"])
    return pd.DataFrame(distance_matrix, index=multi_index, columns=multi_index)

def show_top_matches(sequences: list[WordTuple], top_n: int):

    df_matrix = match_every_to_distance(sequences)

    matrix_values = df_matrix.values
    upper_triangle_mask = np.triu(np.ones(matrix_values.shape), k=1).astype(bool)

    row_indices, col_indices = np.where(upper_triangle_mask)
    distances = matrix_values[upper_triangle_mask]

    pairs_df = pd.DataFrame({
        'row_idx': row_indices,
        'col_idx': col_indices,
        'distance': distances
    })

    if top_n > 0:
        top_pairs = pairs_df.nsmallest(top_n, 'distance')
    else:
        top_pairs = pairs_df.nlargest(top_n*(-1), 'distance')

    for idx, row in top_pairs.iterrows():
        word_i_info = df_matrix.index[int(row['row_idx'])]
        word_j_info = df_matrix.columns[int(row['col_idx'])]

        lang_i, concept_i, form_i = word_i_info
        lang_j, concept_j, form_j = word_j_info

        score, distance, alignment, traceback = evaluate_single(form_i, form_j)

        print("##############################")
        print(f"Distance: {row['distance']:.4f}")
        print(f"Word 1: {form_i:<15}, Language: {lang_i:<12}, Concept: {concept_i}")
        print(f"Word 2: {form_j:<15}, Language: {lang_j:<12}, Concept: {concept_j}")
        alignment_algorithm.print_alignment(traceback, "-" + form_i, "-" + form_j)
