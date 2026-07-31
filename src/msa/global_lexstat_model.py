from data_structures.models import LexstatMatrix

class GlobalLexstatModel:

    def __init__(self):
        self._matrices = {}

    def add_matrix(self, lang1: str, lang2: str, matrix: LexstatMatrix):
        key = tuple(sorted([lang1, lang2]))
        self._matrices[key] = matrix

    def has_matrix(self, lang1: str, lang2: str) -> bool:
        key = tuple(sorted([lang1, lang2]))
        return key in self._matrices

    def get_score(self, lang_a: str, lang_b: str, char_a: str, char_b: str) -> float:
        key = tuple(sorted([lang_a, lang_b]))
        matrix = self._matrices.get(key)

        if not matrix:
            return 0.0

        # swap characters if query order differs from internal sorted key
        if lang_a <= lang_b:
            return matrix.get_score(char_a, char_b)
        else:
            return matrix.get_score(char_b, char_a)