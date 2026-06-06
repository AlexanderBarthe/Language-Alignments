import os
from concurrent.futures import ProcessPoolExecutor

import alignment_algorithm
import numpy as np
import pandas as pd
from tqdm import tqdm
from language_input import WordTuple

MAX_WORKERS = 10

def _align_worker(task):
    i, j, form_i, form_j = task
    score, _, _, _, _ = evaluate_single(form_i, form_j)
    distance = score_to_distance(score)
    return i, j, distance

def evaluate_single(seq1: str, seq2: str):

    fs_score, fs_i, fs_j, fs_matrix, fs_traceback = alignment_algorithm.align(seq1, seq2, True, False)

    fe_score, fe_i, fe_j, fe_matrix, fe_traceback = alignment_algorithm.align(seq1, seq2, False, True)

    if fs_score > fe_score:
        return fs_score, fs_i, fs_j, fs_matrix, fs_traceback
    else:
        return fe_score, fe_i, fe_j, fe_matrix, fe_traceback

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

def match_every(sequences: list[WordTuple]):

    filename = 'distances.dat'
    n = len(sequences)
    #matrix = np.zeros((n, n), dtype='float16')
    matrix = np.memmap(filename, dtype='float32',
                           mode='w+', shape=(n, n))

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, sequences[i].form, sequences[j].form))

    if not tasks:
        return pd.DataFrame(matrix)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = executor.map(_align_worker, tasks, chunksize=1000)
        results = list(tqdm(iterator, total=len(tasks), desc="Calculate Alignments"))

    for i, j, distance in results:
        matrix[i, j] = distance
        matrix[j, i] = distance

    ram_matrix = np.array(matrix)
    del matrix

    multi_index = pd.MultiIndex.from_tuples(sequences, names=["Language", "Concept", "Form"])
    return pd.DataFrame(ram_matrix, index=multi_index, columns=multi_index)


def load_existing_matrix(filename: str, sequences: list[WordTuple]):
    n = len(sequences)

    matrix_disk = np.memmap(filename, dtype='float32', mode='r', shape=(n, n))

    ram_matrix = np.array(matrix_disk)

    del matrix_disk

    multi_index = pd.MultiIndex.from_tuples(sequences, names=["Language", "Concept", "Form"])

    return pd.DataFrame(ram_matrix, index=multi_index, columns=multi_index)

def score_to_distance(score: float):
    return np.exp(-score)