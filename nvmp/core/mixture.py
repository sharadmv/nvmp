from deepx import T

from .statistics import Stats
from .random_tensor import RandomTensor

class Mixture(RandomTensor):

    def __init__(self, tensor, weights):
        super(Mixture, self).__init__()
        self.tensor, self.weights = tensor, weights

    def shape(self):
        return self.tensor.shape()[1:]

    def _sample(self, num_samples):
        tensor_samples = self.tensor.sample(num_samples)
        weight_samples = self.weights.sample(num_samples)
        return T.einsum('ia,iab->ib', weight_samples, tensor_samples)

    def log_likelihood(self, x):
        raise Exception()

    def _statistic(self, stat):
        weights = Stats.X(self.weights)
        stats = stat(self.tensor)
        out = T.core.tensordot(weights, stats, 1)
        out.set_shape(T.get_shape(stats)[1:])
        return out

    def statistics(self):
        return {
            Stats.X, Stats.XXT
        }
