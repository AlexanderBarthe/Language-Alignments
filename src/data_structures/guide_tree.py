
class TreeNode:
    def __init__(self, name: str = None, left: "TreeNode" = None, right: "TreeNode" = None, distance: float = 0.0):
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