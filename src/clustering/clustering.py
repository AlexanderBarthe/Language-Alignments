import numpy as np
import pandas as pd
from infomap import Infomap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN, AffinityPropagation
from dynamicTreeCut import cutreeHybrid
from sklearn_extra.cluster import KMedoids

def run_dynamic_tree_clustering(df_matrix: pd.DataFrame, min_cluster_size: int = 2, method: str = 'average') -> tuple[pd.DataFrame, np.ndarray]:
    condensed_distances = squareform(df_matrix.values, checks=False)
    tree = linkage(condensed_distances, method=method)

    # apply dynamic tree cut using link param
    cluster_results = cutreeHybrid(
        link=tree,
        distM=df_matrix.values,
        minClusterSize=min_cluster_size,
        deepSplit=2,
        pamStage=False
    )

    labels_array = np.array(cluster_results["labels"])
    labels_array[labels_array == 0] = -1

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = labels_array

    return results_df, tree


def run_hdbscan_clustering(df_matrix: pd.DataFrame, min_cluster_size: int = 2) -> pd.DataFrame:

    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    cluster_labels = hdbscan.fit_predict(df_matrix.values)

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = cluster_labels

    return results_df


def run_infomap_clustering(df_matrix: pd.DataFrame, dist_threshold: float = 0.5) -> pd.DataFrame:

    im = Infomap(silent=True)
    dist_matrix = df_matrix.values
    n_nodes = dist_matrix.shape[0]

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = dist_matrix[i, j]
            if dist <= dist_threshold:
                similarity_weight = 1.0 / (1.0 + dist)
                im.add_link(i, j, similarity_weight)

    im.run()

    # extract node module ids
    labels = np.full(n_nodes, -1)
    for node in im.tree:
        if node.is_leaf:
            labels[node.node_id] = node.module_id

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = labels

    return results_df


def run_kmedoids_clustering(df_matrix: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    # run k-medoids with precomputed distance matrix
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', init='k-medoids++')
    cluster_labels = kmedoids.fit_predict(df_matrix.values)

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = cluster_labels

    return results_df


def run_affinity_propagation_clustering(df_matrix: pd.DataFrame, damping: float = 0.5) -> pd.DataFrame:
    # affinity propagation needs similarity matrix
    similarities = -df_matrix.values

    ap = AffinityPropagation(affinity='precomputed', damping=damping, random_state=42)
    cluster_labels = ap.fit_predict(similarities)

    results_df = df_matrix.index.to_frame(index=False)
    results_df['Cluster_ID'] = cluster_labels

    return results_df


def get_cluster_count(results_df: pd.DataFrame) -> int:
    return results_df['Cluster_ID'].nunique()


def get_entries_from_cluster(results_df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    return results_df[results_df['Cluster_ID'] == cluster_id]


def calculate_cluster_impurity(clustered_df: pd.DataFrame) -> float:
    # calculate ratio of minority concepts within each valid cluster
    valid_df = clustered_df[clustered_df['Cluster_ID'] != -1]

    if valid_df.empty:
        return 1.0

    # compute minority counts per cluster and sum
    total_minority_words = valid_df.groupby('Cluster_ID')['Concept'].apply(
        lambda concepts: len(concepts) - concepts.value_counts().iloc[0]
    ).sum()

    return float(total_minority_words / len(valid_df))


def calculate_noise_ratio(results_df: pd.DataFrame) -> float:
    # return ratio of elements labeled as noise
    if results_df.empty:
        return 0.0

    return float((results_df['Cluster_ID'] == -1).mean())