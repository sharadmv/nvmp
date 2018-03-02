from deepx import T

from .statistics import Stats
from .tensor import Tensor

class Add(Tensor):

    def __init__(self, left, right):
        super(Add, self).__init__()
        self.left, self.right = left, right

    def shape(self):
        return self.left.shape()

    def _sample(self, num_samples=None):
        return self.left._sample(num_samples) + self.right._sample(num_samples)

    def _statistic(self, stat):
        if stat == Stats.X:
            return Stats.X(self.left) + Stats.X(self.right)
        elif stat == Stats.XXT:
            return (
                Stats.XXT(self.left) + Stats.XXT(self.right)
                + T.outer(Stats.X(self.left), Stats.X(self.right))
                + T.outer(Stats.X(self.right), Stats.X(self.left))
            )

class ScalarMul(Tensor):

    def __init__(self, left, right):
        super(ScalarMul, self).__init__()
        self.left, self.right = left, right

    def shape(self):
        return self.right.shape()

    def _sample(self, num_samples):
        return self.left._sample(num_samples) * self.right._sample(num_samples)

    def _statistic(self, stat):
        if stat == Stats.X:
            return Stats.X(self.left) * Stats.X(self.right)
        elif stat == Stats.XXT:
            return (
                Stats.XXT(self.left) * Stats.XXT(self.right)
            )

class MatVecMul(Tensor):

    def __init__(self, left, right):
        super(MatVecMul, self).__init__()
        self.left, self.right = left, right

    def shape(self):
        return self.left.shape()

    def _sample(self, num_samples=None):
        return self.left._sample(num_samples) + self.right._sample(num_samples)

    def _statistic(self, stat):
        if stat == Stats.X:
            return Stats.X(self.left) + Stats.X(self.right)
        elif stat == Stats.XXT:
            return (
                Stats.XXT(self.left) + Stats.XXT(self.right)
                + T.outer(Stats.X(self.left), Stats.X(self.right))
                + T.outer(Stats.X(self.right), Stats.X(self.left))
            )
