import optuna
import pycldf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import clustering
import match_evaluator
import scores
from clustering import calculate_cluster_impurity
from language_input import get_all_words_as_tuples
from match_evaluator import match_every

ds = pycldf.Dataset.from_metadata("languages/blumpanotacana/cldf/cldf-metadata.json")
sequences_sample = get_all_words_as_tuples(ds, sample_ratio=0.1, seed=100)

def align_objective(trial):

    gap_penalty = trial.suggest_float('GAP_PENALTY', -12.0, -0.2)
    metathesis_penalty = trial.suggest_float('METATHESIS_PENALTY', -12.0, -0.2)
    metathesis_extend_penalty = trial.suggest_float('METATHESIS_PENALTY_EXTEND', -12.0, -0.2)
    fusion_penalty = trial.suggest_float('FUSION_PENALTY', -12.0, -0.2)

    params = dict(gap_penalty=gap_penalty,
                  metathesis_penalty=metathesis_penalty,
                  metathesis_extend_penalty=metathesis_extend_penalty,
                  fusion_penalty=fusion_penalty)

    scores.override_scoring_params(params)

    df_matrix = match_every(sequences_sample)

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

def cluster_objective(trial):

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

    scores.reset_scoring_params()

    df_matrix = match_every(sequences_sample)

    study = optuna.create_study(direction='minimize')

    study.optimize(cluster_objective, n_trials=500)

    print("Best parameter combination:")
    print(study.best_params)
    print(f"Best loss: {study.best_value}")


def db_scan_objective(trial):
    gap_penalty = trial.suggest_float('GAP_PENALTY', -12.0, -0.2)
    metathesis_penalty = trial.suggest_float('METATHESIS_PENALTY', -12.0, -0.2)
    metathesis_extend_penalty = trial.suggest_float('METATHESIS_PENALTY_EXTEND', -12.0, -0.2)
    fusion_penalty = trial.suggest_float('FUSION_PENALTY', -12.0, -0.2)

    params = dict(gap_penalty=gap_penalty,
                  metathesis_penalty=metathesis_penalty,
                  metathesis_extend_penalty=metathesis_extend_penalty,
                  fusion_penalty=fusion_penalty)
    scores.override_scoring_params(params)

    df_matrix = match_every(sequences_sample)
    epsilon = trial.suggest_float('EPSILON', 0.01, 0.5)
    results_df = clustering.run_dbscan_clustering(df_matrix, epsilon)

    noise_ratio = clustering.calculate_noise_ratio(results_df)

    if noise_ratio == 1.0:
        return 1.0

    base_impurity = clustering.calculate_cluster_impurity(results_df)

    valid_df = results_df[results_df['Cluster_ID'] != -1]
    valid_cluster_count = valid_df['Cluster_ID'].nunique()
    fragmentation_penalty = valid_cluster_count / len(valid_df)

    final_loss = (0.5 * base_impurity) + (0.25 * fragmentation_penalty) + (0.25 * noise_ratio)

    print(f"Impurity: {base_impurity:.4f} | Fragmentation-Penalty: {fragmentation_penalty: 4f} | Noise-Ratio: {noise_ratio:.4f} | Loss: {final_loss:.4f}")

    return final_loss


def find_best_dbscan_params():

    study = optuna.create_study(direction='minimize')

    study.optimize(db_scan_objective, n_trials=50)

    print("Best parameter combination:")
    print(study.best_params)
    print(f"Best loss: {study.best_value}")


