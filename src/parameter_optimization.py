import optuna
import pycldf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from language_input import get_word_tuple_samples
from src.cldf_repo import CLDFRepository
from src.clustering import clustering
from src.clustering.clustering import calculate_cluster_impurity
from src.data_structures.models import ScoringParams
from src.lexstat_scores import get_lexstat_score
from src.simple_alignment import match_evaluator
from src.simple_alignment.match_evaluator import match_every_to_distance

ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")
sequences_sample = get_word_tuple_samples(ds, sample_ratio=0.2, seed=100)

def align_objective(trial):

    gap_penalty = trial.suggest_float('GAP_PENALTY', -12.0, -0.2)
    metathesis_penalty = trial.suggest_float('METATHESIS_PENALTY', -12.0, -0.2)
    metathesis_extend_penalty = trial.suggest_float('METATHESIS_PENALTY_EXTEND', -12.0, -0.2)
    fusion_penalty = trial.suggest_float('FUSION_PENALTY', -12.0, -0.2)

    params = ScoringParams.custom_params(gap_penalty, metathesis_penalty, metathesis_extend_penalty, fusion_penalty)

    df_matrix = match_evaluator.match_every_to_distance(sequences_sample, params)

    condensed_distances = squareform(df_matrix.values, checks=False)
    tree = linkage(condensed_distances, method='average')

    cutoff_fraction = trial.suggest_float('CUTOFF_FRACTION', 0.05, 0.95)

    max_tree_distance = tree[:, 2].max()
    actual_cutoff = cutoff_fraction * max_tree_distance

    cluster_labels = fcluster(tree, t=actual_cutoff, criterion='distance')

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = cluster_labels

    base_impurity = calculate_cluster_impurity(results_df)
    total_words = len(results_df)
    cluster_count = results_df['Cluster_ID'].nunique()
    fragmentation_penalty = cluster_count / total_words

    print("Impurity: ", base_impurity)

    final_loss = (0.6 * base_impurity) + (0.4 * fragmentation_penalty)
    return final_loss


def find_best_alignment_params():

    study = optuna.create_study(direction='minimize')

    study.optimize(align_objective, n_trials=50)

    print("Best parameter combination:")
    print(study.best_params)
    print(f"Best loss: {study.best_value}")

def cluster_objective(trial) -> float:

    df_matrix = match_evaluator.load_existing_matrix('distances.dat', sequences_sample)

    epsilon = trial.suggest_float('EPSILON', 0.01, 2)
    results_df = clustering.run_dbscan_clustering(df_matrix, epsilon)

    noise_ratio = clustering.calculate_noise_ratio(results_df)

    if noise_ratio == 1.0:
        return 1.0

    base_impurity = clustering.calculate_cluster_impurity(results_df)

    valid_df = results_df[results_df['Cluster_ID'] != -1]
    valid_cluster_count = valid_df['Cluster_ID'].nunique()
    fragmentation_penalty = valid_cluster_count / len(valid_df)

    final_loss = (0.5 * base_impurity) + (0.25 * fragmentation_penalty) + (0.25 * noise_ratio)

    print(
        f"Impurity: {base_impurity:.4f} | Fragmentation-Penalty: {fragmentation_penalty: 4f} | Noise-Ratio: {noise_ratio:.4f} | Loss: {final_loss:.4f}")

    return final_loss



def find_best_clustering_params():

    df_matrix = match_every_to_distance(sequences_sample)

    study = optuna.create_study(direction='minimize')

    study.optimize(cluster_objective, n_trials=500)

    print("Best parameter combination:")
    print(study.best_params)
    print(f"Best loss: {study.best_value}")


def create_cluster_objective(df_matrix):
    def cluster_objective2(trial) -> float:
        # optimize dbscan epsilon
        epsilon = trial.suggest_float('EPSILON', 0.01, 5)
        results_df = clustering.run_dbscan_clustering(df_matrix, epsilon)

        noise_ratio = clustering.calculate_noise_ratio(results_df)

        if noise_ratio == 1.0:
            return 1.0

        base_impurity = clustering.calculate_cluster_impurity(results_df)

        valid_df = results_df[results_df['Cluster_ID'] != -1]
        valid_cluster_count = valid_df['Cluster_ID'].nunique()
        fragmentation_penalty = valid_cluster_count / len(valid_df)

        final_loss = (0.5 * base_impurity) + (0.25 * fragmentation_penalty) + (0.25 * noise_ratio)

        return final_loss

    return cluster_objective2


def find_best_clustering_params_for_pair(ds, lang1_name: str, lang2_name: str):
    cldf = CLDFRepository(ds)
    words = cldf.get_words_for_language_as_tuples(lang1_name)
    words.extend(cldf.get_words_for_language_as_tuples(lang2_name))

    lexstat_scores = get_lexstat_score(ds, lang1_name, lang2_name)

    df_matrix = match_evaluator.match_every_to_distance(words, None, lexstat_scores)

    study = optuna.create_study(direction='minimize')
    objective = create_cluster_objective(df_matrix)
    study.optimize(objective, n_trials=500)

    print("Best parameter combination:")
    print(study.best_params)
    print(f"Best loss: {study.best_value}")

    return study.best_params

def db_scan_objective(trial) -> float:
    gap_penalty = trial.suggest_float('GAP_PENALTY', -12.0, -0.2)
    metathesis_penalty = trial.suggest_float('METATHESIS_PENALTY', -12.0, -0.2)
    metathesis_extend_penalty = trial.suggest_float('METATHESIS_PENALTY_EXTEND', -12.0, -0.2)
    fusion_penalty = trial.suggest_float('FUSION_PENALTY', -12.0, -0.2)

    params = ScoringParams.custom_params(gap_penalty, metathesis_penalty, metathesis_extend_penalty, fusion_penalty)

    df_matrix = match_evaluator.match_every_to_distance(sequences_sample, params)
    epsilon = trial.suggest_float('EPSILON', 0.01, 0.5)
    results_df = clustering.run_dbscan_clustering(df_matrix, epsilon)

    noise_ratio = clustering.calculate_noise_ratio(results_df)

    if noise_ratio == 1.0:
        return 1.0

    base_impurity = clustering.calculate_cluster_impurity(results_df)

    valid_df = results_df[results_df['Cluster_ID'] != -1]
    valid_cluster_count = valid_df['Cluster_ID'].nunique()
    fragmentation_penalty = valid_cluster_count / len(valid_df)

    final_loss = (0.4 * base_impurity) + (0.2 * fragmentation_penalty) + (0.4 * noise_ratio)

    if noise_ratio > 0.5:
        final_loss += (noise_ratio - 0.5) * 2.0

    if base_impurity > 0.5:
        final_loss += (base_impurity - 0.5) * 2.0

    print(f"Impurity: {base_impurity:.4f} | Fragmentation-Penalty: {fragmentation_penalty: 4f} | Noise-Ratio: {noise_ratio:.4f} | Loss: {final_loss:.4f}")

    return final_loss


def find_best_dbscan_params():

    study = optuna.create_study(direction='minimize')

    study.optimize(db_scan_objective, n_trials=50)

    print("Best parameter combination:")
    print(study.best_params)
    print(f"Best loss: {study.best_value}")


