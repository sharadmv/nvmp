from deepx import T
from .node import Node

class Add(Node):

    def __init__(self, left, right):
        self.left, self.right = left, right
        self._value = self.left.value() + self.right.value()

    def sample(self, num_samples=1):
        return self.left.sample(num_samples=num_samples) + self.right.sample(num_samples=num_samples)

    def get_stat(self, name, feed_dict={}):
        if name == 'x':
            return self.left.get_stat('x', feed_dict=feed_dict) + self.right.get_stat('x', feed_dict=feed_dict)
        elif name == 'xxT':
            x, xx = self.left.get_stats('x', 'xxT', feed_dict=feed_dict)
            y, yy = self.right.get_stats('x', 'xxT', feed_dict=feed_dict)
            return xx + 2 * T.outer(x, y) + yy
        raise Exception()

    def value(self):
        return self._value
