import scores

score_matrix = []
trace_matrix = []

free_start_gaps = False
free_end_gaps = False

def align(seq1, seq2, free_start, free_end):
    global free_start_gaps
    global free_end_gaps

    s1 = "-" + seq1
    s2 = "-" + seq2

    free_start_gaps = free_start
    free_end_gaps = free_end

    rows = len(s2)
    columns = len(s1)

    init_matrix(rows, columns)
    fill_alignment(s1, s2)

    abs_final_score, i, j = get_final_absolute_score()

    rel_final_score = scores.get_relative_score(abs_final_score, s1, s2)

    return rel_final_score, i, j, score_matrix, trace_matrix

def init_matrix(rows: int, columns: int):
    global score_matrix, trace_matrix

    score_matrix = [[0 for _ in range(columns)] for _ in range(rows)]
    trace_matrix = [["/" for _ in range(columns)] for _ in range(rows)]

    for j in range(1, columns):
        if not free_start_gaps:
            score_matrix[0][j] = j * scores.DEFAULT_GAP_PENALTY

        trace_matrix[0][j] = "I"

    for i in range(1, rows):
        if not free_start_gaps:
            score_matrix[i][0] = i * scores.DEFAULT_GAP_PENALTY

        trace_matrix[i][0] = "D"

def fill_alignment(s1: str, s2: str):
    global score_matrix, trace_matrix

    for i in range(1, len(score_matrix)):
        for j in range(1, len(score_matrix[0])):
            score_matrix[i][j], trace_matrix[i][j] = scores.calculate_best(score_matrix, s1, s2, i, j)

def get_final_absolute_score():

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


def print_alignment(traceback: list[list[str]], seq1: str, seq2: str):
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
