from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from src.environment.config import CONFIG

type ScoreMatrix = list[list[float]]
type TracebackMatrix = list[list[str]]
type DistanceMatrix = pd.DataFrame
type LexstatMatrix = dict[tuple[str, str], float]

class WordTuple(NamedTuple):
    language: str
    concept: str
    form: str

@dataclass
class ScoringParams:
    gap: float
    metathesis: float
    metathesis_extend: float
    fusion: float

    @classmethod
    def from_defaults(cls) -> "ScoringParams":
        defs = CONFIG["penalties"]["defaults"]
        return cls(
            gap=defs["gap"],
            metathesis=defs["metathesis"],
            metathesis_extend=defs["metathesis_extend"],
            fusion=defs["fusion"]
        )

    @classmethod
    def custom_params(cls, gap: float, metathesis: float, metathesis_extend: float, fusion: float) -> "ScoringParams":
        return cls(
            gap=gap,
            metathesis=metathesis,
            metathesis_extend=metathesis_extend,
            fusion=fusion
        )

