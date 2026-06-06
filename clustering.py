import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

CUTOFF = 0.4

def run_hierarchical_clustering(df: pd.DataFrame, method: str = 'average'):

    condensed_distances = squareform(df.values, checks=False)

    z = linkage(condensed_distances, method=method)

    cluster_labels = fcluster(z, t=CUTOFF, criterion='distance')

    results_df = df.index.to_frame(index=False)

    results_df['Cluster_ID'] = cluster_labels

    return results_df, z

def calculate_avg_cluster_impurity(results_df: pd.DataFrame) -> float:
    all_words_cnt = len(results_df)
    if all_words_cnt == 0:
        return 0.0

    weighted_error = 0.0

    for cluster_id, cluster_group in results_df.groupby('Cluster_ID'):
        cluster_size = len(cluster_group)

        most_occuring_concept = cluster_group['Concept'].value_counts().iloc[0]

        cluster_error_rate = 1.0 - (most_occuring_concept / cluster_size)

        weighted_error += cluster_error_rate * (cluster_size / all_words_cnt)

    return weighted_error

def get_cluster_count(results_df: pd.DataFrame) -> int:
    return results_df['Cluster_ID'].nunique()

def get_entries_from_cluster(results_df: pd.DataFrame, cluster_id: int):
    return results_df[results_df['Cluster_ID'] == cluster_id]