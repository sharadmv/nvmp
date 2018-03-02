from deepx import T

from .statistics import Stats
from .tensor import Tensor

class Tile(Tensor):

    def __init__(self, tensor, num):
        super(Tile, self).__init__()
        self.tensor, self.num = tensor, num

    def shape(self):
        return T.concat([[self.num], self.tensor.shape()], 0)

    def sample(self, num_samples=None):
        raise NotImplementedError

    def statistics(self):
        return self.tensor.statistics()

    def _statistic(self, stat):
        return T.pack([stat(self.tensor) for _ in range(self.num)])
