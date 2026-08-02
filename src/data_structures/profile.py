type ProfileColumn = dict[str, str]


class Profile:
    def __init__(self, rows: dict[str, list[str]]):
        # map language name directly to character list
        self.rows = rows

    @classmethod
    def from_single_word(cls, word: str, lang_name: str) -> "Profile":
        return cls({lang_name: list(word)})

    @classmethod
    def empty(cls, lang_names: list[str]) -> "Profile":
        return cls({lid: [] for lid in lang_names})

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        # get length of first character list if dict is not empty
        return len(next(iter(self.rows.values()))) if self.rows else 0

    def get_column(self, col_idx: int) -> ProfileColumn:
        return {lang: chars[col_idx] for lang, chars in self.rows.items()}

    def append_column(self, chars: list[str]):
        if len(chars) != self.height:
            raise ValueError("character count must match profile height")

        for char_list, char in zip(self.rows.values(), chars):
            char_list.append(char)

    def insert_front(self, chars: list[str]):
        if len(chars) != self.height:
            raise ValueError("character count must match profile height")

        for char_list, char in zip(self.rows.values(), chars):
            char_list.insert(0, char)

    def insert_front_lang_sensitive(self, mapped_chars: dict[str, str]):
        # o(1) lookups instead of iterating
        if any(lang not in mapped_chars for lang in self.rows):
            raise ValueError("not all langs in input map")

        for lang, char_list in self.rows.items():
            char_list.insert(0, mapped_chars[lang])

    def update_row(self, lang_name: str, new_chars: list[str]):
        # instant access, no loops or shadowing bugs
        if lang_name not in self.rows:
            raise ValueError(f"language {lang_name} not found")

        if self.width > 0 and len(new_chars) != self.width:
            raise ValueError("character count must match profile width")

        self.rows[lang_name] = new_chars

    def get_languages(self) -> list[str]:
        return list(self.rows.keys())

    def ensure_first_column_is_gap(self):
        if self.width == 0:
            return

        first_column = self.get_column(0)

        all_gaps = all(char == "-" for char in first_column.values())

        if not all_gaps:
            self.insert_front(["-"] * self.height)

    def __str__(self) -> str:
        if not self.rows or self.width == 0:
            return "empty profile"

        col_widths = []
        for j in range(self.width):
            col = self.get_column(j)
            max_col_w = max(len(char) for char in col.values())
            col_widths.append(max_col_w)

        max_lang_pad = max(len(lang) for lang in self.rows)

        lines = []
        for lang, chars in self.rows.items():
            padded_chars = [
                f"{char:<{col_widths[j]}}" for j, char in enumerate(chars)
            ]
            lines.append(f"{lang:<{max_lang_pad}} : {' '.join(padded_chars)}")

        return "\n".join(lines)