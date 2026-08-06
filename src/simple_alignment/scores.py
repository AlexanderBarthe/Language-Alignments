from lingpy.data import Model

from src.data_structures.models import ScoreMatrix, ScoringParams, LexstatMatrix
from config import CONFIG


class AlignmentScorer:

    def __init__(self, params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None):
        self.params = params or ScoringParams.from_defaults()
        self.model = Model(CONFIG['alignment']['model'])
        self.lexstat_matrix = lexstat_matrix

    def calculate_best(self, matrix: ScoreMatrix, seq1: str, seq2: str, pos_i: int, pos_j: int) -> tuple[float, str]:

        direct_score = self.score_direct(matrix[pos_i - 1][pos_j - 1], seq1[pos_j], seq2[pos_i])

        deletion_score = self.score_deletion(matrix[pos_i - 1][pos_j])

        insertion_score = self.score_insertion(matrix[pos_i][pos_j - 1])

        contraction_score = float('-inf')
        if pos_i >= 2:
            contraction_score = self.score_contraction(matrix[pos_i - 2][pos_j - 1], seq2[pos_i - 1], seq2[pos_i], seq1[pos_j])

        expansion_score = float('-inf')
        if pos_j >= 2:
            expansion_score = self.score_expansion(matrix[pos_i - 1][pos_j - 2], seq2[pos_i], seq1[pos_j - 1], seq1[pos_j])

        max_met_len = CONFIG['alignment']['max_metathesis_length']
        metathesis_score, metathesis_length = self.score_syllable_metathesis(seq1, seq2, matrix, pos_i, pos_j, max_met_len)

        ops = CONFIG['operations']
        metathesis_op = f"{ops['metathesis']}_{metathesis_length}"

        options = [
            (direct_score, ops['match']),
            (metathesis_score, metathesis_op),
            (contraction_score, ops['contraction']),
            (expansion_score, ops['expansion']),
            (deletion_score, ops['deletion']),
            (insertion_score, ops['insertion'])
        ]
        best_score, best_op = max(options, key=lambda x: x[0])

        return best_score, best_op


    def score_direct(self, base_score: float, char1: str, char2: str) -> float:

        return base_score + self.get_lingpy_comparison_score(char1, char2)

    def score_deletion(self, base_score: float) -> float:
        return base_score + self.params.gap

    def score_insertion(self, base_score: float) -> float:
        return base_score + self.params.gap

    def score_contraction(self, base_score: float, char1_a: str, char1_b: str, char2_target: str) -> float:

        # Calculate and find best anchor candidate
        score_a = self.get_lingpy_comparison_score(char1_a, char2_target)
        score_b = self.get_lingpy_comparison_score(char1_b, char2_target)

        best_anchor_score = max(score_a, score_b)

        return base_score + best_anchor_score + self.params.fusion

    def score_expansion(self, base_score: float, char1_source: str, char2_a: str, char2_b: str) -> float:

        return self.score_contraction(base_score, char2_a, char2_b, char1_source)

    def score_metathesis(self, base_score: float, char1_prev: str, char1_curr: str, char2_prev: str, char2_curr: str) -> float:
        cross_match_1 = self.get_lingpy_comparison_score(char1_prev, char2_curr)
        cross_match_2 = self.get_lingpy_comparison_score(char1_curr, char2_prev)
        return base_score + cross_match_1 + cross_match_2 + self.params.metathesis

    def score_syllable_metathesis(self, seq1: str, seq2: str, alignment: ScoreMatrix, i: int, j: int, max_length: int) -> tuple[float, int]:

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

            unchanged_match_score_syl1 = self.get_lingpy_string_score(word1_syl1, word2_syl1)
            unchanged_match_score_syl2 = self.get_lingpy_string_score(word1_syl2, word2_syl2)

            swapped_match_score_syl1 = self.get_lingpy_string_score(word1_syl1, word2_syl2)
            swapped_match_score_syl2 = self.get_lingpy_string_score(word1_syl2, word2_syl1)

            if unchanged_match_score_syl1 + unchanged_match_score_syl2 > swapped_match_score_syl1 + swapped_match_score_syl2:
                continue

            swap_score = origin_score + swapped_match_score_syl1 + swapped_match_score_syl2
            penalized_score = swap_score + self.params.metathesis + (current_syllable_length - 1) * self.params.metathesis_extend

            if penalized_score > best_score:
                best_score = penalized_score
                best_syllable_length = current_syllable_length

        return best_score, best_syllable_length

    def get_lingpy_comparison_score(self, char1: str, char2: str) -> float:
        lp = CONFIG["lingpy"]

        class1 = self.model.converter.get(char1, char1)
        class2 = self.model.converter.get(char2, char2)

        # determine base raw score from model or exact match
        if char1 == char2:
            raw_score = lp['exact_match_score']
        else:
            raw_score = self.model.scorer[class1, class2]
            if raw_score >= lp["threshold_high"]:
                raw_score = lp["score_high"]
            elif raw_score >= lp["threshold_mid"]:
                raw_score = lp["score_mid"]
            elif raw_score >= lp["threshold_low"]:
                raw_score = lp["score_low"]
            else:
                raw_score = lp["score_mismatch"]

        # adjust raw score using calculated lexstat matrix if available
        if self.lexstat_matrix is not None:
            pair = (class1, class2)
            lexstat_weight = CONFIG["alignment"].get("lexstat_weight", 0.5)

            ls_score = self.lexstat_matrix.get(pair, 0.0)
            return (1-lexstat_weight)*raw_score + (lexstat_weight * ls_score)

        return raw_score

    def get_lingpy_string_score(self, str1: str, str2: str) -> float:
        accu = 0

        for i in range (0, min(len(str1), len(str2))):
            accu += self.get_lingpy_comparison_score(str1[i], str2[i])

        return accu


    def get_relative_score(self, raw_score: float, seq1: str, seq2: str) -> float:

            max_len = max(len(seq1), len(seq2))
            if max_len == 0:
                return 0.0

            max_possible_score = max_len * CONFIG['lingpy']['exact_match_score']
            relative_score = raw_score / max_possible_score

            return relative_score