from lingpy import Model

from config import CONFIG
from data_structures.models import ScoringParams, ScoreMatrix
from data_structures.phonemes import sound_class
from data_structures.profile import Profile, ProfileColumn
from msa.global_lexstat_model import GlobalLexstatModel


class ProfileScorer:

    def __init__(self, params: ScoringParams | None = None, lexstat_model: GlobalLexstatModel | None = None):
        self.params = params or ScoringParams.from_defaults()
        self.model = Model(CONFIG['alignment']['model'])
        self.lexstat_model = lexstat_model

    def calculate_best(self, matrix: ScoreMatrix, profile1: Profile, profile2: Profile, pos_i: int, pos_j: int) -> \
    tuple[float, str]:

        col1 = profile1.get_column(pos_i)
        col2 = profile2.get_column(pos_j)

        direct_score = matrix[pos_i - 1][pos_j - 1] + self.compare_columns(profile1, pos_i, profile2, pos_j)
        deletion_score = matrix[pos_i - 1][pos_j] + self.calculate_gap(col1, col2)
        insertion_score = matrix[pos_i][pos_j - 1] + self.calculate_gap(col2, col1)

        ops = CONFIG['operations']
        options = [
            (direct_score, ops['match']),
        ]
        pair_count = profile1.height * profile2.height
        max_length = CONFIG['alignment']['max_metathesis_length']
        for length in range(1, min(max_length, pos_i // 2, pos_j // 2) + 1):
            start_i = pos_i - 2 * length + 1
            start_j = pos_j - 2 * length + 1
            unchanged = sum(self.compare_columns(profile1, start_i + k, profile2, start_j + k)
                            for k in range(2 * length))
            swapped = sum(self.compare_columns(profile1, start_i + k, profile2, start_j + length + k)
                          + self.compare_columns(profile1, start_i + length + k, profile2, start_j + k)
                          for k in range(length))
            if swapped >= unchanged:
                penalty = self.params.metathesis + (length - 1) * self.params.metathesis_extend
                options.append((matrix[start_i - 1][start_j - 1] + swapped + pair_count * penalty,
                                f"{ops['metathesis']}_{length}"))

        if pos_i >= 2:
            anchor = max(self.compare_columns(profile1, pos_i - 1, profile2, pos_j),
                         self.compare_columns(profile1, pos_i, profile2, pos_j))
            options.append((matrix[pos_i - 2][pos_j - 1] + anchor + pair_count * self.params.fusion,
                            ops['contraction']))
        if pos_j >= 2:
            anchor = max(self.compare_columns(profile1, pos_i, profile2, pos_j - 1),
                         self.compare_columns(profile1, pos_i, profile2, pos_j))
            options.append((matrix[pos_i - 1][pos_j - 2] + anchor + pair_count * self.params.fusion,
                            ops['expansion']))
        options.extend([(deletion_score, ops['deletion']), (insertion_score, ops['insertion'])])

        best_score, best_op = max(options, key=lambda x: x[0])

        return best_score, best_op

    def compare_columns(self, profile1: Profile, pos_i: int, profile2: Profile, pos_j: int) -> float:
        return self.calculate_direct(profile1.get_column(pos_i), profile2.get_column(pos_j),
                                     profile1.row_languages, profile2.row_languages)

    def calculate_direct(self, col1: ProfileColumn, col2: ProfileColumn,
                         languages1: dict[str, str] | None = None, languages2: dict[str, str] | None = None) -> float:

        gap = self.params.gap
        score = 0
        languages1 = languages1 if languages1 is not None else {row_id: row_id for row_id in col1}
        languages2 = languages2 if languages2 is not None else {row_id: row_id for row_id in col2}

        for lang1, char1 in col1.items():
            for lang2, char2 in col2.items():

                if char1 == "-" and char2 == "-":
                    continue
                elif char1 == "-" or char2 == "-":
                    score += gap
                else:
                    score += self.get_lingpy_comparison_score(
                        languages1[lang1], languages2[lang2], char1, char2)

        return score

    def calculate_gap(self, col1: ProfileColumn, col2: ProfileColumn) -> float:

        return sum(token != "-" for token in col1.values()) * len(col2) * self.params.gap

    def get_lingpy_comparison_score(self, lang_name1: str, lang_name2: str, char1: str, char2: str) -> float:

        class1 = sound_class(char1, self.model)
        class2 = sound_class(char2, self.model)

        raw_score = float(self.model.scorer[class1, class2])

        if self.lexstat_model is None or not self.lexstat_model.has_matrix(lang_name1, lang_name2):
            return raw_score

        ls_score = float(
            self.lexstat_model.get_score(lang_name1, lang_name2, class1, class2)
        )

        lexstat_weight = float(CONFIG["alignment"].get("lexstat_weight", 0.5))
        return (1.0 - lexstat_weight) * raw_score + (lexstat_weight * ls_score)

