import numpy as np
from deepx import T

from ..core import Stats, coerce
from .distribution import Distribution

class Gumbel(Distribution):

    def __init__(self, m, b):
        super(Gumbel, self).__init__()
        self.m, self.b = coerce(m), coerce(b)

    def shape(self):
        return T.shape(Stats.X(self.m))

    def statistics(self):
        return { Stats.X }

    def _statistic(self, stat):
        if stat == Stats.X:
            return Stats.X(self.m) + np.euler_gamma * Stats.X(self.b)
        raise NotImplementedError

    def log_likelihood(self, x):
        m, b = Stats.X(m), Stats.X(b)
        x = Stats.X(x)
        z = (x - m) / b
        return 1 / b * T.exp(-z - T.exp(-z))

    def _sample(self, num_samples):
        shape = self.shape()
        sample_shape = T.concat([[num_samples], shape], 0)
        random_sample = T.random_uniform(sample_shape)
        m, b = Stats.X(self.m), Stats.X(self.b)
        return m[None] - b[None] * T.log(-T.log(random_sample))
