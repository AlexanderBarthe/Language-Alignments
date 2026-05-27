import alignment_algorithm
import scores


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












