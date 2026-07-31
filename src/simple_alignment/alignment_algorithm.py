
from src.data_structures.models import ScoreMatrix, TracebackMatrix, ScoringParams, LexstatMatrix
from src.simple_alignment import scores
from src.simple_alignment.scores import AlignmentScorer


def align(s1: str, s2: str, free_start_gaps: bool, free_end_gaps: bool, custom_params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None) \
        -> tuple[float, int, int , ScoreMatrix, TracebackMatrix]:

    scorer = scores.AlignmentScorer(custom_params, lexstat_matrix)

    if not(s1.startswith("-")):
        s1 = "-" + s1
    if not(s2.startswith("-")):
        s2 = "-" + s2

    rows = len(s2)
    columns = len(s1)

    score_matrix, trace_matrix = init_matrix(rows, columns, free_start_gaps, scorer)
    score_matrix, trace_matrix = fill_alignment(s1, s2, score_matrix, trace_matrix, scorer)

    abs_final_score, end_cutoff_i, end_cutoff_j = get_final_absolute_score(free_end_gaps, score_matrix)

    rel_final_score = scorer.get_relative_score(abs_final_score, s1, s2)

    return rel_final_score, end_cutoff_i, end_cutoff_j, score_matrix, trace_matrix

def init_matrix(rows: int, columns: int, free_start_gaps: bool, scorer: AlignmentScorer) -> tuple[ScoreMatrix, TracebackMatrix]:

    score_matrix = [[0 for _ in range(columns)] for _ in range(rows)]
    trace_matrix = [["/" for _ in range(columns)] for _ in range(rows)]

    for j in range(1, columns):
        if not free_start_gaps:
            score_matrix[0][j] = j * scorer.params.gap

        trace_matrix[0][j] = "I"

    for i in range(1, rows):
        if not free_start_gaps:
            score_matrix[i][0] = i * scorer.params.gap

        trace_matrix[i][0] = "D"

    return score_matrix, trace_matrix

def fill_alignment(s1: str, s2: str, score_matrix: ScoreMatrix, trace_matrix: TracebackMatrix, scorer: AlignmentScorer) \
        -> tuple[ScoreMatrix, TracebackMatrix]:

    for i in range(1, len(score_matrix)):
        for j in range(1, len(score_matrix[0])):
            score_matrix[i][j], trace_matrix[i][j] = scorer.calculate_best(score_matrix, s1, s2, i, j)

    return score_matrix, trace_matrix

def get_final_absolute_score(free_end_gaps: bool, score_matrix: ScoreMatrix) -> tuple[float, int, int]:

    if free_end_gaps:

        best_score= float("-inf")
        best_score_i = 0
        best_score_j = 0

        for i in range(1, len(score_matrix)):
            for j in range(1, len(score_matrix[0])):
                if score_matrix[i][j] > best_score:
                    best_score = score_matrix[i][j]
                    best_score_i = i
                    best_score_j = j

        return best_score, best_score_i, best_score_j

    else:
        return score_matrix[-1][-1], -1, -1


def print_matrix(matrix, seq1: str, seq2: str):
    width = 10

    print(f"{'#':>{width}}", end="")
    for char in seq1:
        print(f"{char:>{width}}", end="")
    print()

    for i in range(len(matrix)):
        print(f"{seq2[i]:>{width}}", end="")

        for j in range(len(matrix[0])):
            val = matrix[i][j]

            if isinstance(val, float):
                print(f"{val:>{width}.2f}", end="")
            else:
                print(f"{val:>{width}}", end="")
        print()

def get_matched_seqs(traceback: TracebackMatrix, seq1: str, seq2: str) -> list[tuple[str, str]]:

    if not(seq1.startswith("-")):
        seq1 = "-" + seq1
    if not(seq2.startswith("-")):
        seq2 = "-" + seq2

    i = len(seq2) - 1
    j = len(seq1) - 1

    pairs = []

    while i > 0 or j > 0:
        op = traceback[i][j]
        char1 = ""
        char2 = ""

        if op == "M":
            char1 = seq1[j]
            char2 = seq2[i]
            i -= 1
            j -= 1
        elif op == "D":
            char1 = "-"
            char2 = seq2[i]
            i -= 1
        elif op == "I":
            char1 = seq1[j]
            char2 = "-"
            j -= 1
        elif op == "C":
            char1 = seq1[j]
            char2 = seq2[i - 1], seq2[i]
            i -= 2
            j -= 1
        elif op == "E":
            char1 = seq1[j - 1], seq1[j]
            char2 = seq2[i]
            i -= 1
            j -= 2
        elif op.startswith("S"):
            swap_length = int(op.split("_")[1])

            char1 = seq1[j - swap_length * 2 + 1: j + 1]
            char2 = seq2[i - swap_length * 2 + 1: i + 1]

            i -= swap_length * 2
            j -= swap_length * 2
        else:
            break

        pairs.append(tuple([char1, char2]))

    pairs.reverse()

    return pairs

def print_alignment(traceback: TracebackMatrix, seq1: str, seq2: str):

    if not(seq1.startswith("-")):
        seq1 = "-" + seq1
    if not(seq2.startswith("-")):
        seq2 = "-" + seq2

    i = len(seq2) - 1
    j = len(seq1) - 1

    aln_seq1 = []
    aln_mid = []
    aln_seq2 = []

    while i > 0 or j > 0:
        op = traceback[i][j]

        if op == "M":
            char1 = seq1[j]
            char2 = seq2[i]
            aln_seq1.append(char1)
            aln_seq2.append(char2)
            aln_mid.append("|" if char1 == char2 else " ")
            i -= 1
            j -= 1

        elif op == "D":
            aln_seq1.append("-")
            aln_seq2.append(seq2[i])
            aln_mid.append(" ")
            i -= 1

        elif op == "I":
            aln_seq1.append(seq1[j])
            aln_seq2.append("-")
            aln_mid.append(" ")
            j -= 1

        elif op == "C":
            aln_seq1.extend(["-", seq1[j]])
            aln_seq2.extend([seq2[i], seq2[i - 1]])
            aln_mid.extend([" ", "v"])
            i -= 2
            j -= 1

        elif op == "E":
            aln_seq1.extend([seq1[j], seq1[j - 1]])
            aln_seq2.extend(["-", seq2[i]])
            aln_mid.extend([" ", "^"])
            i -= 1
            j -= 2

        elif op.startswith("S"):
            swap_length = int(op.split("_")[1])

            # Append letters in swap area from original strings in reverse
            aln_seq1.extend(seq1[j - swap_length*2 + 1 : j+1][::-1])
            aln_seq2.extend(seq2[i - swap_length*2 + 1 : i+1][::-1])

            for k in range(1, swap_length * 2 + 1):
                if k <= swap_length:
                    aln_mid.append("⟨")
                elif k >= swap_length+1:
                    aln_mid.append("⟩")
                else:
                    aln_mid.append(" ")

            i -= swap_length*2
            j -= swap_length*2

        else:
            break

    aln_seq1.reverse()
    aln_mid.reverse()
    aln_seq2.reverse()

    str_seq1 = "".join(aln_seq1)
    str_mid = "".join(aln_mid)
    str_seq2 = "".join(aln_seq2)

    print(str_seq1)
    print(str_mid)
    print(str_seq2)
