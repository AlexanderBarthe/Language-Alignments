from lingpy import Model

from data_structures.models import ScoringParams, LexstatMatrix, ScoreMatrix
from data_structures.profile import Profile, ProfileColumn
from environment.config import CONFIG
from msa.global_lexstat_model import GlobalLexstatModel


class ProfileScorer:

    def __init__(self, params: ScoringParams = None, lexstat_model: GlobalLexstatModel = None):
        self.params = params or ScoringParams.from_defaults()
        self.model = Model(CONFIG['alignment']['model'])
        self.lexstat_model = lexstat_model

    def calculate_best(self, matrix: ScoreMatrix, profile1: Profile, profile2: Profile, pos_i: int, pos_j: int) -> \
    tuple[float, str]:

        col1 = profile1.get_column(pos_i)
        col2 = profile2.get_column(pos_j)

        direct_score = matrix[pos_i - 1][pos_j - 1] + self.calculate_direct(col1, col2)
        deletion_score = matrix[pos_i - 1][pos_j] + self.calculate_gap(col1, col2)

        insertion_score = matrix[pos_i][pos_j - 1] + self.calculate_gap(col1, col2)

        ops = CONFIG['operations']
        options = [
            (direct_score, ops['match']),
            (deletion_score, ops['deletion']),
            (insertion_score, ops['insertion'])
        ]

        best_score, best_op = max(options, key=lambda x: x[0])

        return best_score, best_op

    def calculate_direct(self, col1: ProfileColumn, col2: ProfileColumn) -> float:

        gap = CONFIG["penalties"].get("gap", -3)
        score = 0

        for lang1, char1 in col1.items():
            for lang2, char2 in col2.items():

                if char1 == "-" and char2 == "-":
                    continue
                elif char1 == "-" or char2 == "-":
                    score += gap
                else:
                    score += self.get_lingpy_comparison_score(lang1, lang2, char1, char2)

        return score

    def calculate_gap(self, col1: ProfileColumn, col2: ProfileColumn) -> float:

        gap = CONFIG["penalties"].get("gap", -3)

        return len(col1) * len(col2) * gap




    def get_lingpy_comparison_score(self, lang_name1: str, lang_name2: str, char1: str, char2: str) -> float:
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
        if self.lexstat_model is not None:
            pair = (class1, class2)
            lexstat_weight = CONFIG["alignment"].get("lexstat_weight", 0.5)

            ls_score = self.lexstat_model.get_score(lang_name1, lang_name2, char1, char2)

            return (1-lexstat_weight)*raw_score + (lexstat_weight * ls_score)

        return raw_score

    def get_lingpy_string_score(self, str1: str, str2: str) -> float:
        accu = 0

        for i in range(0, min(len(str1), len(str2))):
            accu += self.get_lingpy_comparison_score(str1[i], str2[i])

        return accu

