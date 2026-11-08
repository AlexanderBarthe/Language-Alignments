import pandas as pd

class TreeNode:
    def __init__(self, name: str | None = None, left: "TreeNode | None " = None, right: "TreeNode| None " = None, distance: float = 0.0):
        self.name = name
        self.left = left
        self.right = right
        self.distance = distance

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def post_order_traversal(self):
        # yields child nodes before parent nodes
        if self.left:
            yield from self.left.post_order_traversal()
        if self.right:
            yield from self.right.post_order_traversal()
        yield self

    def __str__(self) -> str:
        lines = []
        self._build_string(lines, "", True, True)
        return "\n".join(lines)

    def _build_string(self, lines: list[str], prefix: str, is_last: bool, is_root: bool):
        # resolve node name and format current line
        node_name = self.name if self.name else "*"

        if is_root:
            lines.append(node_name)
            next_prefix = ""
        else:
            branch = "└── " if is_last else "├── "
            lines.append(f"{prefix}{branch}{node_name}")
            next_prefix = prefix + ("    " if is_last else "│   ")

        # process valid children recursively
        children = [node for node in (self.left, self.right) if node]
        for i, child in enumerate(children):
            is_last_child = i == (len(children) - 1)
            child._build_string(lines, next_prefix, is_last_child, False)

    def compare(self, tree: TreeNode) -> float:

        self_splits, self_leaves = TreeNode.get_unrooted_splits(self)
        other_splits, other_leaves = TreeNode.get_unrooted_splits(tree)

        if self_leaves != other_leaves:
            raise ValueError(
                "Cannot compare trees with differing leaf sets."
            )

        if not self_splits and not other_splits:
            return 0.0

        mismatches = len(self_splits.symmetric_difference(other_splits))

        max_distance = len(self_splits) + len(other_splits)

        return mismatches / max_distance

    def _collect_leaf_names(self) -> set[str]:
        return {node.name for node in self.post_order_traversal() if node.is_leaf()}

    def _collect_clusters(self, all_leaves: frozenset[str]) -> set[frozenset[str]]:

        clusters: set[frozenset[str]] = set()
        for node in self.post_order_traversal():
            if node.is_leaf():
                continue
            cluster = frozenset(node._collect_leaf_names())
            if len(cluster) >= 2 and cluster != all_leaves:
                clusters.add(cluster)
        return clusters

    @staticmethod
    def get_leaf_names(node: TreeNode) -> list[str]:

        if node.is_leaf():
            if node.name is None:
                raise ValueError("Leaf without a name.")
            return [node.name]

        names = []

        if node.left is not None:
            names.extend(TreeNode.get_leaf_names(node.left))

        if node.right is not None:
            names.extend(TreeNode.get_leaf_names(node.right))

        return names

    @staticmethod
    def get_unrooted_splits(root: TreeNode):
        leaf_names = TreeNode.get_leaf_names(root)

        if len(leaf_names) != len(set(leaf_names)):
            raise ValueError("Tree contains duplicate leaf names.")

        all_leaves = frozenset(leaf_names)
        splits = set()

        def traverse(node: TreeNode) -> frozenset[str]:

            if node.is_leaf():
                return frozenset([node.name])

            descendant_leaves = set()

            for child in (node.left, node.right):
                if child is None:
                    continue

                child_leaves = traverse(child)
                other_leaves = all_leaves - child_leaves

                # Ignore trivial splits (single leaf on either side).
                if len(child_leaves) >= 2 and len(other_leaves) >= 2:
                    side_a = tuple(sorted(child_leaves))
                    side_b = tuple(sorted(other_leaves))

                    canonical_side = min(
                        (side_a, side_b),
                        key=lambda side: (len(side), side),
                    )

                    splits.add(frozenset(canonical_side))

                descendant_leaves.update(child_leaves)

            return frozenset(descendant_leaves)

        traverse(root)

        return splits, all_leaves

    @staticmethod
    def calculate_split_support(
            reference_tree: TreeNode,
            bootstrap_trees: list[TreeNode],
    ) -> pd.DataFrame:

        if not bootstrap_trees:
            raise ValueError("No bootstrap trees provided.")

        reference_splits, reference_leaves = TreeNode.get_unrooted_splits(
            reference_tree
        )

        bootstrap_split_sets = []

        # Extract splits from every bootstrap tree.
        for i, tree in enumerate(bootstrap_trees):

            splits, leaves = TreeNode.get_unrooted_splits(tree)

            if leaves != reference_leaves:
                raise ValueError(
                    f"Bootstrap tree {i} has a different set of leaves."
                )

            bootstrap_split_sets.append(splits)

        rows = []

        # Calculate support for every split in the reference tree.
        for split in reference_splits:
            count = sum(
                split in bootstrap_splits
                for bootstrap_splits in bootstrap_split_sets
            )

            other_side = reference_leaves - split

            rows.append({
                "Split": (
                    f"{{{', '.join(sorted(split))}}} | "
                    f"{{{', '.join(sorted(other_side))}}}"
                ),
                "Support count": count,
                "Bootstrap support (%)": (
                        count / len(bootstrap_trees) * 100
                ),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Bootstrap support (%)")
            .reset_index(drop=True)
        )
