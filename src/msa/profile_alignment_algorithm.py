from data_structures.profile import Profile
from msa import profile_scores
from msa.global_lexstat_model import GlobalLexstatModel
from src.data_structures.models import ScoreMatrix, TracebackMatrix, ScoringParams
from src.environment.config import CONFIG


def align_profiles(profile_a: Profile, profile_b: Profile, free_start_gaps: bool, free_end_gaps: bool,
        custom_params: ScoringParams = None, lexstat_model: GlobalLexstatModel = None) -> tuple[Profile, float, float, ScoreMatrix, TracebackMatrix]:

    scorer = profile_scores.ProfileScorer(custom_params, lexstat_model)

    profile_a.ensure_first_column_is_gap()
    profile_b.ensure_first_column_is_gap()

    matrix_rows = profile_a.width
    matrix_cols = profile_b.width

    score_matrix, trace_matrix = init_matrix(matrix_rows, matrix_cols, free_start_gaps, scorer)

    score_matrix, trace_matrix = fill_alignment(profile_a, profile_b, score_matrix, trace_matrix, scorer)

    abs_final_score, end_cutoff_i, end_cutoff_j = get_final_absolute_score(free_end_gaps, score_matrix)

    rel_final_score = get_relative_score(abs_final_score, profile_a, profile_b)

    merged_profile = build_new_profile(trace_matrix, profile_a, profile_b)

    return merged_profile, abs_final_score, rel_final_score, score_matrix, trace_matrix



def init_matrix(rows: int, columns: int, free_start_gaps: bool, scorer: profile_scores.ProfileScorer) -> tuple[ScoreMatrix, TracebackMatrix]:

    score_matrix = [[0 for _ in range(columns)] for _ in range(rows)]
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

def fill_alignment(profile1: Profile, profile2: Profile, score_matrix: ScoreMatrix, trace_matrix: TracebackMatrix, scorer: profile_scores.ProfileScorer) \
        -> tuple[ScoreMatrix, TracebackMatrix]:

    for i in range(1, len(score_matrix)):
        for j in range(1, len(score_matrix[0])):
            score_matrix[i][j], trace_matrix[i][j] = scorer.calculate_best(score_matrix, profile1, profile2, i, j)

    return score_matrix, trace_matrix

def get_final_absolute_score(free_end_gaps: bool, score_matrix: ScoreMatrix) -> tuple[float, int, int]:

    if free_end_gaps:

        best_score= float("-inf")
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
        return score_matrix[-1][-1], -1, -1

def get_relative_score(raw_score: float, profile1: Profile, profile2: Profile) -> float:

        max_len = max(profile1.width, profile2.width)
        if max_len == 0:
            return 0.0

        max_possible_score = max_len * CONFIG['lingpy']['exact_match_score']
        relative_score = raw_score / max_possible_score

        return relative_score


def build_new_profile(traceback_matrix: TracebackMatrix, profile1: Profile, profile2: Profile) -> Profile:
    ops = CONFIG['operations']

    profile1.ensure_first_column_is_gap()
    profile2.ensure_first_column_is_gap()

    languages = profile1.get_languages()
    languages.extend(profile2.get_languages())

    result = Profile.empty(languages)

    current_i = len(traceback_matrix) - 1
    current_j = len(traceback_matrix[0]) - 1

    while current_i > 0 or current_j > 0:

        current_op = traceback_matrix[current_i][current_j]
        current_cognates = {}

        if current_op == ops['match']:
            current_cognates.update(profile1.get_column(current_i))
            current_cognates.update(profile2.get_column(current_j))
            current_i -= 1
            current_j -= 1

        elif current_op == ops['deletion']:
            current_cognates.update(profile1.get_column(current_i))
            for lang in profile2.get_languages():
                current_cognates[lang] = "-"
            current_i -= 1

        elif current_op == ops['insertion']:
            for lang in profile1.get_languages():
                current_cognates[lang] = "-"
            current_cognates.update(profile2.get_column(current_j))
            current_j -= 1

        result.insert_front_lang_sensitive(current_cognates)

    return result

