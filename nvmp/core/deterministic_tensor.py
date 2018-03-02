from deepx import T
from .statistics import Stats
from .tensor import Tensor

class DeterministicTensor(Tensor):

    def __init__(self, value):
        super(DeterministicTensor, self).__init__()
        self.value = value

    def shape(self):
        return T.shape(self.value)

    def sample(self, num_samples=None):
        if num_samples is None:
            return self.value
        return T.pack([self.value for _ in range(num_samples)])

    def _statistic(self, stat):
        return stat(self.value)

    def statistics(self):
        return set(Stats)

    def log_likelihood(self):
        return 0.
