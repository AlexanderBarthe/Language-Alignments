import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN

DEFAULT_TREE_CUTOFF = 0.4

def run_hierarchical_clustering(df: pd.DataFrame, cutoff:float = DEFAULT_TREE_CUTOFF, method: str = 'average'):
    condensed_distances = squareform(df.values, checks=False)
    tree = linkage(condensed_distances, method=method)
    cluster_labels = fcluster(tree, t=cutoff, criterion='distance')
    results_df = df.index.to_frame(index=False)
    results_df['Cluster_ID'] = cluster_labels
    return results_df, tree

def run_dbscan_clustering(df_matrix: pd.DataFrame, eps: float, min_samples: int = 2):

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = dbscan.fit_predict(df_matrix.values)

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = cluster_labels

    return results_df

def get_cluster_count(results_df: pd.DataFrame) -> int:
    return results_df['Cluster_ID'].nunique()

def get_entries_from_cluster(results_df: pd.DataFrame, cluster_id: int):
    return results_df[results_df['Cluster_ID'] == cluster_id]

def calculate_cluster_impurity(clustered_df: pd.DataFrame) -> float:
    valid_df = clustered_df[clustered_df['Cluster_ID'] != -1]

    if valid_df.empty:
        return 1.0

    total_minority_words = 0
    total_valid_words = len(valid_df)

    for cluster_id, cluster_group in valid_df.groupby('Cluster_ID'):
        cluster_size = len(cluster_group)

        majority_concept_count = cluster_group['Concept'].value_counts().iloc[0]

        minority_count = cluster_size - majority_concept_count
        total_minority_words += minority_count

    return total_minority_words / total_valid_words


def calculate_noise_ratio(results_df: pd.DataFrame) -> float:

    total_words = len(results_df)
    if total_words == 0:
        return 0.0

    noise_count = (results_df['Cluster_ID'] == -1).sum()
    return noise_count / total_words