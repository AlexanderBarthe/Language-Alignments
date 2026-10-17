from collections.abc import Sequence

from config import CONFIG
from data_structures.models import ScoreMatrix, TracebackMatrix, ScoringParams
from data_structures.phonemes import tokenize
from data_structures.profile import Profile
from msa import profile_scores
from msa.global_lexstat_model import GlobalLexstatModel


def align(string_a: str | Sequence[str], lang_a: str, string_b: str | Sequence[str], lang_b: str,
          free_start_gaps: bool, free_end_gaps: bool, custom_params: ScoringParams | None = None,
          lexstat_model: GlobalLexstatModel | None = None) -> tuple[Profile, float, float, ScoreMatrix, TracebackMatrix]:
    profile_a = Profile.from_single_word(string_a, lang_a)
    profile_b = Profile.from_single_word(string_b, lang_b)
    return align_profiles(profile_a, profile_b, free_start_gaps, free_end_gaps, custom_params, lexstat_model)


def align_profiles(profile_a: Profile, profile_b: Profile, free_start_gaps: bool, free_end_gaps: bool,
                   custom_params: ScoringParams | None = None, lexstat_model: GlobalLexstatModel | None = None) \
        -> tuple[Profile, float, float, ScoreMatrix, TracebackMatrix]:
    scorer = profile_scores.ProfileScorer(custom_params, lexstat_model)

    # reserve a DP origin without mutating profiles or consuming a real gap column
    profile_a = Profile({row_id: ["-"] + row for row_id, row in profile_a.rows.items()}, profile_a.row_languages)
    profile_b = Profile({row_id: ["-"] + row for row_id, row in profile_b.rows.items()}, profile_b.row_languages)
    score_matrix, trace_matrix = init_matrix(max(1, profile_a.width), max(1, profile_b.width), free_start_gaps, scorer)
    for i in range(1, len(score_matrix)):
        if not free_start_gaps:
            score_matrix[i][0] = score_matrix[i - 1][0] + scorer.calculate_gap(
                profile_a.get_column(i), {row_id: "-" for row_id in profile_b.rows})
    for j in range(1, len(score_matrix[0])):
        if not free_start_gaps:
            score_matrix[0][j] = score_matrix[0][j - 1] + scorer.calculate_gap(
                profile_b.get_column(j), {row_id: "-" for row_id in profile_a.rows})

    score_matrix, trace_matrix = fill_alignment(profile_a, profile_b, score_matrix, trace_matrix, scorer)
    abs_final_score, end_cutoff_i, end_cutoff_j = get_final_absolute_score(free_end_gaps, score_matrix)
    rel_final_score = get_relative_score(abs_final_score, profile_a, profile_b, scorer)
    merged_profile = build_new_profile(trace_matrix, profile_a, profile_b, end_cutoff_i, end_cutoff_j, scorer)
    return merged_profile, abs_final_score, rel_final_score, score_matrix, trace_matrix


def init_matrix(rows: int, columns: int, free_start_gaps: bool, scorer: profile_scores.ProfileScorer) \
        -> tuple[ScoreMatrix, TracebackMatrix]:
    score_matrix = [[0.0 for _ in range(columns)] for _ in range(rows)]
    trace_matrix = [["/" for _ in range(columns)] for _ in range(rows)]
    ops = CONFIG['operations']
    for j in range(1, columns):
        if not free_start_gaps:
            score_matrix[0][j] = j * scorer.params.gap
        trace_matrix[0][j] = ops['insertion']
    for i in range(1, rows):
        if not free_start_gaps:
            score_matrix[i][0] = i * scorer.params.gap
        trace_matrix[i][0] = ops['deletion']
    return score_matrix, trace_matrix


def fill_alignment(profile1: Profile, profile2: Profile, score_matrix: ScoreMatrix, trace_matrix: TracebackMatrix,
                   scorer: profile_scores.ProfileScorer) -> tuple[ScoreMatrix, TracebackMatrix]:
    for i in range(1, len(score_matrix)):
        for j in range(1, len(score_matrix[0])):
            score_matrix[i][j], trace_matrix[i][j] = scorer.calculate_best(score_matrix, profile1, profile2, i, j)
    return score_matrix, trace_matrix


def get_final_absolute_score(free_end_gaps: bool, score_matrix: ScoreMatrix) -> tuple[float, int, int]:
    last_i = len(score_matrix) - 1
    last_j = len(score_matrix[0]) - 1
    if free_end_gaps:
        endpoints = [(score_matrix[i][last_j], i, last_j) for i in range(last_i + 1)]
        endpoints.extend((score_matrix[last_i][j], last_i, j) for j in range(last_j + 1))
        return max(endpoints, key=lambda endpoint: endpoint[0])
    return score_matrix[last_i][last_j], last_i, last_j


def calculate_profile_self_score(profile: Profile, scorer: profile_scores.ProfileScorer) -> float:
    return sum(scorer.compare_columns(profile, col_idx, profile, col_idx) for col_idx in range(1, profile.width))


def get_relative_score(raw_score: float, profile1: Profile, profile2: Profile,
                       scorer: profile_scores.ProfileScorer) -> float:
    score_p1 = calculate_profile_self_score(profile1, scorer)
    score_p2 = calculate_profile_self_score(profile2, scorer)
    max_possible_score = max(score_p1, score_p2)
    if max_possible_score <= 0:
        return 1.0 if profile1.width <= 1 and profile2.width <= 1 else 0.0
    return max(0.0, min(1.0, raw_score / max_possible_score))


def build_new_profile(traceback_matrix: TracebackMatrix, profile1: Profile, profile2: Profile,
                      end_i: int | None = None, end_j: int | None = None,
                      scorer: profile_scores.ProfileScorer | None = None) -> Profile:
    ops = CONFIG['operations']
    scorer = scorer or profile_scores.ProfileScorer()
    rows = list(profile1.rows)
    right_ids = {}
    reserved = set(profile1.rows) | set(profile2.rows)
    for row_id in profile2.rows:
        new_id = row_id
        suffix = 2
        if new_id in profile1.rows:
            while f"{row_id}#{suffix}" in reserved:
                suffix += 1
            new_id = f"{row_id}#{suffix}"
        right_ids[row_id] = new_id
        reserved.add(new_id)
        rows.append(new_id)
    languages = dict(profile1.row_languages)
    languages.update({right_ids[row_id]: language for row_id, language in profile2.row_languages.items()})
    result = Profile.empty(rows, languages)

    def column(i: int | None, j: int | None) -> dict[str, str]:
        left = profile1.get_column(i) if i is not None else {row_id: "-" for row_id in profile1.rows}
        right = profile2.get_column(j) if j is not None else {row_id: "-" for row_id in profile2.rows}
        return {**left, **{right_ids[row_id]: token for row_id, token in right.items()}}

    current_i = len(traceback_matrix) - 1 if end_i is None else end_i
    current_j = len(traceback_matrix[0]) - 1 if end_j is None else end_j
    columns = []
    while current_i > 0 or current_j > 0:
        current_op = traceback_matrix[current_i][current_j]
        if current_op == ops['match']:
            columns.append(column(current_i, current_j))
            current_i -= 1
            current_j -= 1
        elif current_op == ops['deletion']:
            columns.append(column(current_i, None))
            current_i -= 1
        elif current_op == ops['insertion']:
            columns.append(column(None, current_j))
            current_j -= 1
        elif current_op == ops['contraction']:
            anchor_first = scorer.compare_columns(profile1, current_i - 1, profile2, current_j) >= \
                           scorer.compare_columns(profile1, current_i, profile2, current_j)
            columns.extend([column(current_i, None if anchor_first else current_j),
                            column(current_i - 1, current_j if anchor_first else None)])
            current_i -= 2
            current_j -= 1
        elif current_op == ops['expansion']:
            anchor_first = scorer.compare_columns(profile1, current_i, profile2, current_j - 1) >= \
                           scorer.compare_columns(profile1, current_i, profile2, current_j)
            columns.extend([column(None if anchor_first else current_i, current_j),
                            column(current_i if anchor_first else None, current_j - 1)])
            current_i -= 1
            current_j -= 2
        elif current_op.startswith(ops['metathesis'] + "_"):
            length = 2 * int(current_op.split("_")[1])
            columns.extend(column(current_i - k, current_j - k) for k in range(length))
            current_i -= length
            current_j -= length
        else:
            raise ValueError(f"unknown traceback operation: {current_op}")
    for mapped_tokens in reversed(columns):
        result.append_column([mapped_tokens[row_id] for row_id in rows])
    for i in range((end_i if end_i is not None else len(traceback_matrix) - 1) + 1, len(traceback_matrix)):
        mapped_tokens = column(i, None)
        result.append_column([mapped_tokens[row_id] for row_id in rows])
    for j in range((end_j if end_j is not None else len(traceback_matrix[0]) - 1) + 1, len(traceback_matrix[0])):
        mapped_tokens = column(None, j)
        result.append_column([mapped_tokens[row_id] for row_id in rows])
    return result


def get_matched_seqs(traceback: TracebackMatrix, seq1: str | Sequence[str], seq2: str | Sequence[str]) \
        -> list[tuple[str | tuple[str, ...], str | tuple[str, ...]]]:
    seq1 = ("-",) + tokenize(seq1)
    seq2 = ("-",) + tokenize(seq2)
    i, j = len(seq1) - 1, len(seq2) - 1
    ops = CONFIG['operations']
    pairs = []
    while i > 0 or j > 0:
        op = traceback[i][j]
        if op == ops['match']:
            pairs.append((seq1[i], seq2[j]))
            i, j = i - 1, j - 1
        elif op == ops['deletion']:
            pairs.append((seq1[i], "-"))
            i -= 1
        elif op == ops['insertion']:
            pairs.append(("-", seq2[j]))
            j -= 1
        elif op == ops['contraction']:
            pairs.append((seq1[i - 1:i + 1], seq2[j]))
            i, j = i - 2, j - 1
        elif op == ops['expansion']:
            pairs.append((seq1[i], seq2[j - 1:j + 1]))
            i, j = i - 1, j - 2
        elif op.startswith(ops['metathesis'] + "_"):
            length = 2 * int(op.split("_")[1])
            pairs.append((seq1[i - length + 1:i + 1], seq2[j - length + 1:j + 1]))
            i, j = i - length, j - length
        else:
            raise ValueError(f"unknown traceback operation: {op}")
    return list(reversed(pairs))


def print_matrix(matrix, seq1: str | Sequence[str], seq2: str | Sequence[str]):
    rows = ("-",) + tokenize(seq1)
    columns = ("-",) + tokenize(seq2)
    width = 10
    print(f"{'#':>{width}}" + "".join(f"{token:>{width}}" for token in columns))
    for token, values in zip(rows, matrix):
        formatted = [f"{value:>{width}.2f}" if isinstance(value, float) else f"{value:>{width}}" for value in values]
        print(f"{token:>{width}}" + "".join(formatted))


def score_to_distance(relative_score: float) -> float:
    return max(0.0, min(1.0, 1.0 - relative_score))
