
from lingpy.data import Model

asjp_model = Model('sca')

GAP_PENALTY = -3
METATHESIS_PENALTY = -1.5
FUSION_PENALTY = -4

OP_MATCH = "M"       # Match / Mismatch
OP_DELETION = "D"    # Deletion (Gap in word 2)
OP_INSERTION = "I"   # Insertion (Gap in word 1)
OP_CONTRACTION = "C" # Contraction (2 to 1)
OP_EXPANSION = "E"   # Expansion (1 to 2)
OP_METATHESIS = "S"  # Metathesis / Swap

def calculate_best(matrix: list[list[float]], seq1: str, seq2: str, i: int, j: int):

    direct_score = score_direct(matrix[i - 1][j - 1], seq1[j], seq2[i])

    deletion_score = score_deletion(matrix[i - 1][j])

    insertion_score = score_insertion(matrix[i][j - 1])

    contraction_score = float('-inf')
    if i >= 2:
        contraction_score = score_contraction(matrix[i - 2][j - 1], seq2[i - 1], seq2[i], seq1[j])

    expansion_score = float('-inf')
    if j >= 2:
        expansion_score = score_expansion(matrix[i - 1][j - 2], seq2[i], seq1[j - 1], seq1[j])

    metathesis_score = float('-inf')
    if i >= 2 and j >= 2:
        metathesis_score = score_metathesis(matrix[i - 2][j - 2], seq1[j - 1], seq1[j], seq2[i - 1], seq2[i])

    options = [
        (direct_score, OP_MATCH),
        (metathesis_score, OP_METATHESIS),
        (contraction_score, OP_CONTRACTION),
        (expansion_score, OP_EXPANSION),
        (deletion_score, OP_DELETION),
        (insertion_score, OP_INSERTION)
    ]
    best_score, best_op = max(options, key=lambda x: x[0])

    return best_score, best_op


def score_direct(base_score: float, char1: str, char2: str):

    return base_score + get_lingpy_score(char1, char2)

def score_deletion(base_score: float):
    return base_score + GAP_PENALTY

def score_insertion(base_score: float):
    return base_score + GAP_PENALTY

def score_contraction(base_score: float, char1_a: str, char1_b: str, char2_target: str):

    # Calculate and find best anchor candidate
    score_a = get_lingpy_score(char1_a, char2_target)
    score_b = get_lingpy_score(char1_b, char2_target)

    best_anchor_score = max(score_a, score_b)

    return base_score + best_anchor_score + FUSION_PENALTY

def score_expansion(base_score: float, char1_source: str, char2_a: str, char2_b: str):

    return score_contraction(base_score, char2_a, char2_b, char1_source)

def score_metathesis(base_score: float, char1_prev: str, char1_curr: str, char2_prev: str, char2_curr: str):
    cross_match_1 = get_lingpy_score(char1_prev, char2_curr)
    cross_match_2 = get_lingpy_score(char1_curr, char2_prev)
    return base_score + cross_match_1 + cross_match_2 + METATHESIS_PENALTY

def get_lingpy_score(char1: str, char2: str):
    if char1 == char2:
        return 2.5

    class1 = asjp_model.converter.get(char1, char1)
    class2 = asjp_model.converter.get(char2, char2)
    raw_score = asjp_model.scorer[class1, class2]


    if raw_score >= 4.0:
        return 1.5

    elif raw_score >= 0.0:
        return 0.5

    elif raw_score >= -3.0:
        return -1.0

    else:
        return -4.0

def get_relative_score(raw_score: float, seq1: str, seq2: str):
    word_len = max(len(seq1)-1, len(seq2)-1)
    relative_score = raw_score / word_len
    return relative_score