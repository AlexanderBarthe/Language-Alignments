import numpy as np
import pandas as pd

from src.data_structures.guide_tree import TreeNode


def build_upgma_tree(dist_df: pd.DataFrame) -> TreeNode:
    active_nodes = [TreeNode(name=lang) for lang in dist_df.index]

    cluster_sizes = {node: 1 for node in active_nodes}

    matrix = dist_df.values.astype(float).copy()
    np.fill_diagonal(matrix, np.inf)

    while len(active_nodes) > 1:
        idx_i, idx_j = np.unravel_index(np.argmin(matrix), matrix.shape)
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i

        node_i = active_nodes[idx_i]
        node_j = active_nodes[idx_j]
        dist = matrix[idx_i, idx_j]

        parent_node = TreeNode(left=node_i, right=node_j, distance=dist / 2.0)

        size_i = cluster_sizes[node_i]
        size_j = cluster_sizes[node_j]
        new_size = size_i + size_j
        cluster_sizes[parent_node] = new_size

        new_distances = []
        for k in range(len(active_nodes)):
            if k != idx_i and k != idx_j:
                d_ik = matrix[idx_i, k]
                d_jk = matrix[idx_j, k]

                d_new = (size_i * d_ik + size_j * d_jk) / new_size
                new_distances.append(d_new)

        matrix = np.delete(matrix, [idx_i, idx_j], axis=0)
        matrix = np.delete(matrix, [idx_i, idx_j], axis=1)

        if len(new_distances) > 0:
            new_dist_arr = np.array(new_distances)

            matrix = np.vstack([matrix, new_dist_arr])

            new_row = np.append(new_dist_arr, np.inf)
            matrix = np.hstack([matrix, new_row.reshape(-1, 1)])

        active_nodes.pop(idx_j)
        active_nodes.pop(idx_i)
        active_nodes.append(parent_node)

    return active_nodes[0]