from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

import alignment_algorithm
from language_input import WordTuple
import pandas as pd
MAX_WORKERS = 10


def _align_worker(task):
    i, j, form_i, form_j, params = task

    if params:
        import scores
        scores.override_scoring_params(params)

    score, _, _, _, _ = evaluate_single(form_i, form_j)
    return i, j, score

def evaluate_single_semiglobally(seq1: str, seq2: str):
    fs_score, fs_i, fs_j, fs_matrix, fs_traceback = alignment_algorithm.align(seq1, seq2, True, False)

    fe_score, fe_i, fe_j, fe_matrix, fe_traceback = alignment_algorithm.align(seq1, seq2, False, True)

    if fs_score > fe_score:
        return fs_score, fs_i, fs_j, fs_matrix, fs_traceback
    else:
        return fe_score, fe_i, fe_j, fe_matrix, fe_traceback

def evaluate_single(seq1: str, seq2: str):
    return alignment_algorithm.align(seq1, seq2, False, False)

def find_best_match(seq1: str, match_partners: list[str]):

    best_score = float("-inf")
    best_match = None
    best_alignment = None
    best_traceback = None
    comparisons = 0

    for match_partner in match_partners:
        score, i, j, alignment, traceback = evaluate_single(seq1, match_partner)

        if score > best_score:
            best_score = score
            best_match = match_partner
            best_alignment = alignment
            best_traceback = traceback

        comparisons += 1

    return best_match, best_score, best_alignment, best_traceback, comparisons

# Returns distance matrix
def match_every(sequences: list[WordTuple], scoring_params: dict = None) -> pd.DataFrame:

    filename = 'distances.dat'
    n = len(sequences)
    #matrix = np.zeros((n, n), dtype='float16')
    disk_score_matrix = np.memmap(filename, dtype='float32',
                           mode='w+', shape=(n, n))

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, sequences[i].form, sequences[j].form, scoring_params))

    if not tasks:
        return pd.DataFrame(disk_score_matrix)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = executor.map(_align_worker, tasks, chunksize=1000)
        results = list(tqdm(iterator, total=len(tasks), desc="Calculating Alignments", disable=False))

    for i, j, score in results:
        disk_score_matrix[i, j] = score
        disk_score_matrix[j, i] = score

    ram_score_matrix = np.array(disk_score_matrix)
    del disk_score_matrix

    best_score = ram_score_matrix.max()

    ram_distance_matrix = np.array([score_to_distance(best_score, row) for row in ram_score_matrix])

    np.fill_diagonal(ram_distance_matrix, 0.0)

    multi_index = pd.MultiIndex.from_tuples(sequences, names=["Language", "Concept", "Form"])
    return pd.DataFrame(ram_distance_matrix, index=multi_index, columns=multi_index)


def load_existing_matrix(filename: str, sequences: list[WordTuple]):
    n = len(sequences)

    matrix_disk = np.memmap(filename, dtype='float32', mode='r', shape=(n, n))

    ram_score_matrix = np.array(matrix_disk)

    del matrix_disk

    best_score = ram_score_matrix.max()

    ram_distance_matrix = np.array([score_to_distance(best_score, row) for row in ram_score_matrix])

    max_dist = ram_distance_matrix.max()
    if max_dist > 0:
        ram_distance_matrix = ram_distance_matrix / max_dist

    np.fill_diagonal(ram_distance_matrix, 0.0)

    multi_index = pd.MultiIndex.from_tuples(sequences, names=["Language", "Concept", "Form"])
    return pd.DataFrame(ram_distance_matrix, index=multi_index, columns=multi_index)

def score_to_distance(best_score: float, score: float):
    return best_score - score


def show_top_matches(sequences: list[WordTuple], top_n: int):

    df_matrix = match_every(sequences)

    matrix_values = df_matrix.values
    upper_triangle_mask = np.triu(np.ones(matrix_values.shape), k=1).astype(bool)

    row_indices, col_indices = np.where(upper_triangle_mask)
    distances = matrix_values[upper_triangle_mask]

    pairs_df = pd.DataFrame({
        'row_idx': row_indices,
        'col_idx': col_indices,
        'distance': distances
    })

    top_pairs = None

    if top_n > 0:
        top_pairs = pairs_df.nsmallest(top_n, 'distance')
    else:
        top_pairs = pairs_df.nlargest(top_n*(-1), 'distance')

    for idx, row in top_pairs.iterrows():
        word_i_info = df_matrix.index[int(row['row_idx'])]
        word_j_info = df_matrix.columns[int(row['col_idx'])]

        lang_i, concept_i, form_i = word_i_info
        lang_j, concept_j, form_j = word_j_info

        score, i, j, alignment, traceback = evaluate_single(form_i, form_j)

        print("##############################")
        print(f"Distance: {row['distance']:.4f}")
        print(f"Word 1: {form_i:<15}, Language: {lang_i:<12}, Concept: {concept_i}")
        print(f"Word 2: {form_j:<15}, Language: {lang_j:<12}, Concept: {concept_j}")
        alignment_algorithm.print_alignment(traceback, "-" + form_i, "-" + form_j)