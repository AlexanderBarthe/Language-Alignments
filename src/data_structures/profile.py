
class Profile:
    def __init__(self, matrix: list[list[str]], lang_names: list[str]):
        self.matrix = matrix
        self.lang_names = lang_names

    @classmethod
    def from_single_word(cls, word: str, lang_name: str) -> "Profile":
        return cls([list(word)], [lang_name])

    @property
    def height(self) -> int:
        return len(self.matrix)

    @property
    def width(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    def get_column(self, col_idx: int) -> list[str]:
        return [self.matrix[row][col_idx] for row in range(self.height)]

    def append_column(self, col1: list[str], col2: list[str] = None):
        if col2 is None:
            combined = col1
        else:
            combined = col1 + col2

        if not self.matrix:
            self.matrix = [[char] for char in combined]
        else:
            for row_idx, char in enumerate(combined):
                self.matrix[row_idx].append(char)