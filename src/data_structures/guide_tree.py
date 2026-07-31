
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