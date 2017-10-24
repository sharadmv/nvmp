from .node import Node

class Add(Node):

    def __init__(self, left, right):
        self.left, self.right = left, right

    def expected_statistics(self):
        return list(map(sum, zip(self.left.expected_statistics(), self.right.expected_statistics())))
