import optuna
import pandas as pd
from src.clustering import clustering
from src.data_structures.models import ScoringParams
from src.simple_alignment import match_evaluator


def evaluate_clustering(results_df: pd.DataFrame) -> tuple[float, float, float, float]:
    # evaluate cluster metrics and compute unified loss
    noise_ratio = clustering.calculate_noise_ratio(results_df)

    if noise_ratio == 1.0:
        return 1.0, 1.0, 1.0, 1.0

    valid_df = results_df[results_df['Cluster_ID'] != -1]
    valid_cluster_count = valid_df['Cluster_ID'].nunique()
    fragmentation_penalty = valid_cluster_count / len(valid_df) if len(valid_df) > 0 else 1.0

    base_impurity = clustering.calculate_cluster_impurity(results_df)

    final_loss = (0.4 * base_impurity) + (0.2 * fragmentation_penalty) + (0.4 * noise_ratio)

    if noise_ratio > 0.5:
        final_loss += (noise_ratio - 0.5) * 2.0
    if base_impurity > 0.5:
        final_loss += (base_impurity - 0.5) * 2.0

    return final_loss, base_impurity, fragmentation_penalty, noise_ratio


def create_unified_objective(sequences, algo_name: str, opt_alignment: bool, opt_clustering: bool,
                             precomputed_matrix=None):
    # dynamic objective function based on requested configuration
    def objective(trial):
        # handle alignment parameters
        if opt_alignment:
            gap = trial.suggest_float('GAP_PENALTY', -12.0, -0.2)
            meta = trial.suggest_float('METATHESIS_PENALTY', -12.0, -0.2)
            meta_ext = trial.suggest_float('METATHESIS_PENALTY_EXTEND', -12.0, -0.2)
            fusion = trial.suggest_float('FUSION_PENALTY', -12.0, -0.2)
            params = ScoringParams.custom_params(gap, meta, meta_ext, fusion)
            df_matrix = match_evaluator.match_every_to_distance(sequences, params)
        else:
            df_matrix = precomputed_matrix

        # handle clustering algorithms and their hyperparameters
        if algo_name == 'hdbscan':
            # min_cluster_size is fixed to 2 by linguistic logic
            results_df = clustering.run_hdbscan_clustering(df_matrix, min_cluster_size=2)

        elif algo_name == 'dynamic_tree':
            deep_split = trial.suggest_int('DEEP_SPLIT', 0, 4) if opt_clustering else 2
            results_df, _ = clustering.run_dynamic_tree_clustering(df_matrix, min_cluster_size=2, method='average')

        elif algo_name == 'infomap':
            threshold = trial.suggest_float('DIST_THRESH', 0.1, 0.9) if opt_clustering else 0.5
            results_df = clustering.run_infomap_clustering(df_matrix, dist_threshold=threshold)

        elif algo_name == 'kmedoids':
            max_k = max(2, len(df_matrix) // 2)
            n_clusters = trial.suggest_int('N_CLUSTERS', 2, max_k) if opt_clustering else max(2, len(df_matrix) // 10)
            results_df = clustering.run_kmedoids_clustering(df_matrix, n_clusters=n_clusters)

        elif algo_name == 'affinity_propagation':
            damping = trial.suggest_float('DAMPING', 0.5, 0.99) if opt_clustering else 0.5
            results_df = clustering.run_affinity_propagation_clustering(df_matrix, damping=damping)

        else:
            raise ValueError(f"unknown algorithm: {algo_name}")

        # evaluation
        loss, impurity, frag, noise = evaluate_clustering(results_df)

        # store real metrics in trial for later analysis
        trial.set_user_attr("impurity", impurity)
        trial.set_user_attr("fragmentation", frag)
        trial.set_user_attr("noise_ratio", noise)

        return loss

    return objective


def run_optimization_suite(sequences, algorithms: list[str], modes: list[str], n_trials: int = 50) -> pd.DataFrame:
    results = []

    # precompute matrix with internal default alignment settings if needed
    standard_matrix = match_evaluator.match_every_to_distance(sequences)

    for algo in algorithms:
        for mode in modes:
            print(f"--- starting study: {algo} | mode: {mode} ---")

            opt_align = mode in ['alignment', 'joint']
            opt_clust = mode in ['clustering', 'joint']
            matrix = None if opt_align else standard_matrix

            study = optuna.create_study(direction='minimize')
            objective = create_unified_objective(sequences, algo, opt_align, opt_clust, matrix)
            study.optimize(objective, n_trials=n_trials)

            best = study.best_trial
            results.append({
                'algorithm': algo,
                'mode': mode,
                'best_loss': best.value,
                'impurity': best.user_attrs.get('impurity'),
                'fragmentation': best.user_attrs.get('fragmentation'),
                'noise_ratio': best.user_attrs.get('noise_ratio'),
                'best_params': best.params
            })

    return pd.DataFrame(results)
