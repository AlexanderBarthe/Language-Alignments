from collections.abc import Sequence

from lingpy import ipa2tokens
from lingpy.sequence.sound_classes import token2class


def tokenize(word: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(word, str):
        tokens = word.split() if any(char.isspace() for char in word) else ipa2tokens(word) if word else []
    else:
        tokens = word
    return tuple(token for token in tokens if token and token != "+")


def sound_class(token: str, model) -> str:
    return "-" if token == "-" else token2class(token, model, cldf=True)