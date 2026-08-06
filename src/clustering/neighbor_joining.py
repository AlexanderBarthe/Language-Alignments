import numpy as np
import pandas as pd

from src.data_structures.guide_tree import TreeNode


def build_tree_nj(dist_df: pd.DataFrame) -> TreeNode:
    active_nodes = [TreeNode(name=lang) for lang in dist_df.index]

    matrix = dist_df.values.astype(float).copy()

    np.fill_diagonal(matrix, 0.0)

    n = len(active_nodes)

    while n > 2:
        col_sums = np.sum(matrix, axis=0)
        q_matrix = (n - 2) * matrix - col_sums[:, None] - col_sums[None, :]

        # ignore diagonal for finding the minimum
        np.fill_diagonal(q_matrix, np.inf)

        idx_i, idx_j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i

        node_i = active_nodes[idx_i]
        node_j = active_nodes[idx_j]
        dist_ij = matrix[idx_i, idx_j]

        # calculate individual branch lengths for the children
        branch_len_i = (dist_ij / 2.0) + ((col_sums[idx_i] - col_sums[idx_j]) / (2.0 * (n - 2)))
        branch_len_j = dist_ij - branch_len_i

        # attach branch lengths directly to the nodes
        node_i.distance = branch_len_i
        node_j.distance = branch_len_j

        # instantiate parent node. passing dist_ij to avoid init errors if distance is required
        parent_node = TreeNode(left=node_i, right=node_j, distance=dist_ij)

        # calculate distances from the new node to all other remaining nodes
        mask = np.ones(n, dtype=bool)
        mask[[idx_i, idx_j]] = False
        new_distances = (matrix[idx_i, mask] + matrix[idx_j, mask] - dist_ij) / 2.0

        # update distance matrix
        matrix = np.delete(matrix, [idx_i, idx_j], axis=0)
        matrix = np.delete(matrix, [idx_i, idx_j], axis=1)

        if len(new_distances) > 0:
            matrix = np.vstack([matrix, new_distances])
            new_row = np.append(new_distances, 0.0)
            matrix = np.hstack([matrix, new_row.reshape(-1, 1)])

        active_nodes.pop(idx_j)
        active_nodes.pop(idx_i)
        active_nodes.append(parent_node)
        n -= 1

    # connect the last two remaining nodes to form a rooted tree for the msa
    node_i = active_nodes[0]
    node_j = active_nodes[1]
    dist_ij = matrix[0, 1]

    node_i.distance = dist_ij / 2.0
    node_j.distance = dist_ij / 2.0

    root_node = TreeNode(left=node_i, right=node_j, distance=0.0)

    return root_node