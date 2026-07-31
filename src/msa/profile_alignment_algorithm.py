from data_structures.profile import Profile
from msa import profile_scores
from src.data_structures.models import ScoreMatrix, TracebackMatrix, ScoringParams, LexstatMatrix
from src.environment.config import CONFIG


def align_profiles(profile_a: Profile, profile_b: Profile, free_start_gaps: bool, free_end_gaps: bool,
        custom_params: ScoringParams = None, lexstat_matrix: LexstatMatrix = None) -> tuple[Profile, float, ScoreMatrix, TracebackMatrix]:
    scorer = profile_scores.ProfileScorer(custom_params, lexstat_matrix)

    la = profile_a.width
    lb = profile_b.width

    rows = lb + 1
    columns = la + 1

    score_matrix, trace_matrix = init_profile_matrix(rows, columns, free_start_gaps, scorer, profile_a, profile_b)
    score_matrix, trace_matrix = fill_profile_alignment(profile_a, profile_b, score_matrix, trace_matrix, scorer)

    abs_final_score, end_cutoff_i, end_cutoff_j = get_final_absolute_score(free_end_gaps, score_matrix)

    rel_final_score = scorer.get_relative_score(abs_final_score, la, lb)

    aligned_profile = traceback_and_construct(
        profile_a, profile_b, trace_matrix, end_cutoff_i, end_cutoff_j
    )

    return aligned_profile, rel_final_score, score_matrix, trace_matrix


def init_profile_matrix(rows: int, columns: int, free_start_gaps: bool, scorer: profile_scores.ProfileScorer, profile_a: Profile,
        profile_b: Profile) -> tuple[ScoreMatrix, TracebackMatrix]:
    score_matrix = [[0.0 for _ in range(columns)] for _ in range(rows)]
    trace_matrix = [["/" for _ in range(columns)] for _ in range(rows)]

    ops = CONFIG['operations']

    # initialize first row for profile b start gaps
    for j in range(1, columns):
        if not free_start_gaps:
            col_a = profile_a.get_column(j - 1)
            gap_sum = sum(scorer.params.gap for char in col_a if char != '-')
            score_matrix[0][j] = score_matrix[0][j - 1] + gap_sum

        trace_matrix[0][j] = ops['insertion']

    # initialize first column for profile a start gaps
    for i in range(1, rows):
        if not free_start_gaps:
            col_b = profile_b.get_column(i - 1)
            gap_sum = sum(scorer.params.gap for char in col_b if char != '-')
            score_matrix[i][0] = score_matrix[i - 1][0] + gap_sum

        trace_matrix[i][0] = ops['deletion']

    return score_matrix, trace_matrix


def fill_profile_alignment(profile_a: Profile, profile_b: Profile, score_matrix: ScoreMatrix,
        trace_matrix: TracebackMatrix, scorer: profile_scores.ProfileScorer) -> tuple[ScoreMatrix, TracebackMatrix]:
    cols_a = [profile_a.get_column(j) for j in range(profile_a.width)]
    cols_b = [profile_b.get_column(i) for i in range(profile_b.width)]

    la_names = profile_a.lang_names
    lb_names = profile_b.lang_names

    for i in range(1, len(score_matrix)):
        col_b = cols_b[i - 1]
        for j in range(1, len(score_matrix[0])):
            col_a = cols_a[j - 1]

            score, op = scorer.calculate_best(
                score_matrix, col_a, col_b, la_names, lb_names, i, j
            )
            score_matrix[i][j] = score
            trace_matrix[i][j] = op

    return score_matrix, trace_matrix


def get_final_absolute_score(free_end_gaps: bool, score_matrix: ScoreMatrix) -> tuple[float, int, int]:
    if free_end_gaps:
        best_score = float("-inf")
        best_score_i = 0
        best_score_j = 0

        for i in range(1, len(score_matrix)):
            for j in range(1, len(score_matrix[0])):
                if score_matrix[i][j] > best_score:
                    best_score = score_matrix[i][j]
                    best_score_i = i
                    best_score_j = j

        return best_score, best_score_i, best_score_j
    else:
        return score_matrix[-1][-1], len(score_matrix) - 1, len(score_matrix[0]) - 1


def traceback_and_construct(profile_a: Profile, profile_b: Profile, trace_matrix: TracebackMatrix,
        end_i: int, end_j: int) -> Profile:
    i, j = end_i, end_j
    path = []

    ops = CONFIG['operations']
    match_op = ops['match']
    deletion_op = ops['deletion']
    insertion_op = ops['insertion']

    while i > 0 or j > 0:
        op = trace_matrix[i][j]

        if op == match_op and i > 0 and j > 0:
            path.append(('match', j - 1, i - 1))
            i -= 1
            j -= 1
        elif (op == deletion_op or j == 0) and i > 0:
            path.append(('gap_in_b', j - 1 if j < profile_a.width else profile_a.width - 1, None))
            i -= 1
        elif (op == insertion_op or i == 0) and j > 0:
            path.append(('gap_in_a', None, i - 1 if i < profile_b.width else profile_b.width - 1))
            j -= 1
        else:
            if i > 0 and j > 0:
                i -= 1
                j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1

    path.reverse()

    new_height = profile_a.height + profile_b.height
    new_matrix = [[] for _ in range(new_height)]
    new_lang_names = profile_a.lang_names + profile_b.lang_names

    cols_a = [profile_a.get_column(idx) for idx in range(profile_a.width)]
    cols_b = [profile_b.get_column(idx) for idx in range(profile_b.width)]

    for action, idx_a, idx_b in path:
        if action == 'match':
            col_a = cols_a[idx_a]
            col_b = cols_b[idx_b]
            for r_idx, val in enumerate(col_a):
                new_matrix[r_idx].append(val)
            for r_idx, val in enumerate(col_b):
                new_matrix[profile_a.height + r_idx].append(val)
        elif action == 'gap_in_b':
            col_a = cols_a[idx_a]
            for r_idx, val in enumerate(col_a):
                new_matrix[r_idx].append(val)
            for r_idx in range(profile_b.height):
                new_matrix[profile_a.height + r_idx].append('-')
        else:
            for r_idx in range(profile_a.height):
                new_matrix[r_idx].append('-')
            col_b = cols_b[idx_b]
            for r_idx, val in enumerate(col_b):
                new_matrix[profile_a.height + r_idx].append(val)

    return Profile(new_matrix, new_lang_names)