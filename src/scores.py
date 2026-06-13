
from lingpy.data import Model

asjp_model = Model('sca')

MAX_METATHESIS_LENGTH = 3
EXACT_MATCH_FAVOR = 2.5

DEFAULT_GAP_PENALTY = -3
DEFAULT_METATHESIS_PENALTY = -2
DEFAULT_METATHESIS_EXTEND_PENALTY = -1.25
DEFAULT_FUSION_PENALTY = -4

gap_penalty = DEFAULT_GAP_PENALTY
metathesis_penalty = DEFAULT_METATHESIS_PENALTY
metathesis_extend_penalty = DEFAULT_METATHESIS_EXTEND_PENALTY
fusion_penalty = DEFAULT_FUSION_PENALTY

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

    '''
    metathesis_score = float('-inf')
    if i >= 2 and j >= 2:
        metathesis_score = score_metathesis(matrix[i - 2][j - 2], seq1[j - 1], seq1[j], seq2[i - 1], seq2[i])
    '''
    metathesis_score, metathesis_length = score_syllable_metathesis(seq1, seq2, matrix, i, j, MAX_METATHESIS_LENGTH)
    metathesis_op = f"{OP_METATHESIS}_{metathesis_length}"

    options = [
        (direct_score, OP_MATCH),
        (metathesis_score, metathesis_op),
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
    return base_score + gap_penalty

def score_insertion(base_score: float):
    return base_score + gap_penalty

def score_contraction(base_score: float, char1_a: str, char1_b: str, char2_target: str):

    # Calculate and find best anchor candidate
    score_a = get_lingpy_score(char1_a, char2_target)
    score_b = get_lingpy_score(char1_b, char2_target)

    best_anchor_score = max(score_a, score_b)

    return base_score + best_anchor_score + fusion_penalty

def score_expansion(base_score: float, char1_source: str, char2_a: str, char2_b: str):

    return score_contraction(base_score, char2_a, char2_b, char1_source)

def score_metathesis(base_score: float, char1_prev: str, char1_curr: str, char2_prev: str, char2_curr: str):
    cross_match_1 = get_lingpy_score(char1_prev, char2_curr)
    cross_match_2 = get_lingpy_score(char1_curr, char2_prev)
    return base_score + cross_match_1 + cross_match_2 + metathesis_penalty

def score_syllable_metathesis(seq1: str, seq2: str, alignment: list[list[float]], i: int, j: int, max_length: int):

    best_score = float('-inf')
    best_syllable_length = 0

    for current_syllable_length in range(1, max_length + 1):

        if i - current_syllable_length * 2 < 0 or j - current_syllable_length * 2 < 0:
            break

        word1_syl1 = seq1[j - 2 * current_syllable_length + 1: j - current_syllable_length + 1]
        word1_syl2 = seq1[j - current_syllable_length + 1: j + 1]

        word2_syl1 = seq2[i - 2 * current_syllable_length + 1: i - current_syllable_length + 1]
        word2_syl2 = seq2[i - current_syllable_length + 1: i + 1]

        origin_score = alignment[i - current_syllable_length * 2][j - current_syllable_length * 2]

        unchanged_match_score_syl1 = get_lingpy_string_score(word1_syl1, word2_syl1)
        unchanged_match_score_syl2 = get_lingpy_string_score(word1_syl2, word2_syl2)

        swapped_match_score_syl1 = get_lingpy_string_score(word1_syl1, word2_syl2)
        swapped_match_score_syl2 = get_lingpy_string_score(word1_syl2, word2_syl1)

        if unchanged_match_score_syl1 + unchanged_match_score_syl2 > swapped_match_score_syl1 + swapped_match_score_syl2:
            continue

        swap_score = origin_score + swapped_match_score_syl1 + swapped_match_score_syl2
        penalized_score = swap_score + metathesis_penalty+ (current_syllable_length - 1) * metathesis_extend_penalty

        if penalized_score > best_score:
            best_score = penalized_score
            best_syllable_length = current_syllable_length

    return best_score, best_syllable_length


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

def get_lingpy_string_score(str1: str, str2: str):
    accu = 0

    for i in range (0, min(len(str1), len(str2))):
        accu += get_lingpy_score(str1[i], str2[i])

    return accu


def get_relative_score(raw_score: float, seq1: str, seq2: str) -> float:

        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return 0.0

        max_possible_score = max_len * EXACT_MATCH_FAVOR
        relative_score = raw_score / max_possible_score

        return relative_score


def override_scoring_params(params: dict[str, float]):
    global gap_penalty
    gap_penalty = params['gap_penalty']

    global metathesis_penalty
    metathesis_penalty = params['metathesis_penalty']

    global metathesis_extend_penalty
    metathesis_extend_penalty = params['metathesis_extend_penalty']

    global fusion_penalty
    fusion_penalty = params['fusion_penalty']

def reset_scoring_params():
    global gap_penalty
    global metathesis_penalty
    global metathesis_extend_penalty
    global fusion_penalty

    gap_penalty = DEFAULT_GAP_PENALTY
    metathesis_penalty = DEFAULT_METATHESIS_PENALTY
    metathesis_extend_penalty = DEFAULT_METATHESIS_EXTEND_PENALTY
    fusion_penalty = DEFAULT_FUSION_PENALTY