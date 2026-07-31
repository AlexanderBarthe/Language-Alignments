from lingpy import Model

from data_structures.models import ScoringParams, LexstatMatrix, ScoreMatrix
from data_structures.profile import Profile
from environment.config import CONFIG


class ProfileScorer:

    def __init__(self, params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None):
        self.params = params or ScoringParams.from_defaults()
        self.model = Model(CONFIG['alignment']['model'])
        self.lexstat_matrix = lexstat_matrix

    def calculate_best(self, matrix: ScoreMatrix, profile1: Profile, profile2: Profile, pos_i: int, pos_j: int) -> tuple[float, str]:

        direct_score = matrix[pos_i - 1][pos_j - 1] + self.calculate_direct_modif(profile1.get_column(pos_j), profile2.get_column(pos_i))


    def calculate_direct_modif(self, col1: list[str], col2: list[str]) -> float:



    def calculate_deletion_modif(self, col1: list[str], col2: list[str]) -> float:





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
            return raw_score + (lexstat_weight * ls_score)

        return raw_score

    def get_lingpy_string_score(self, str1: str, str2: str) -> float:
        accu = 0

        for i in range(0, min(len(str1), len(str2))):
            accu += self.get_lingpy_comparison_score(str1[i], str2[i])

        return accu

