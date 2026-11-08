
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

        self_leaves = frozenset(self._collect_leaf_names())
        other_leaves = frozenset(tree._collect_leaf_names())

        missing = self_leaves ^ other_leaves
        if missing:
            raise ValueError(
                f"Cannot compare trees with differing leaf sets: {sorted(missing)}"
            )

        self_clusters = self._collect_clusters(self_leaves)
        other_clusters = tree._collect_clusters(other_leaves)

        mismatches = len(self_clusters - other_clusters) + len(other_clusters - self_clusters)

        max_distance = 2 * (len(self_leaves) - 2)
        if max_distance == 0:
            return 0.0
        return min(mismatches / max_distance, 1.0)

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

