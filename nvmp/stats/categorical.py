import numpy as np
from deepx import T

from ..core import Stats
from .exponential_family import ExponentialFamily
from .gumbel import Gumbel

class Categorical(ExponentialFamily):

    def shape(self):
        return T.shape(self.get_parameters('natural')[Stats.X])

    def _sample(self, num_samples):
        a = self.get_parameters('natural')[Stats.X]
        d = self.shape()[-1]
        gumbel_noise = Gumbel(T.zeros_like(a), T.ones_like(a)).sample(num_samples)
        return T.one_hot(T.argmax(a[None] + gumbel_noise, -1), d)

    def statistics(self):
        return { Stats.X }

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        pi = regular_parameters
        return {
            Stats.X: Stats.LogX(pi)
        }

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        eta = natural_parameters[Stats.X]
        return T.exp(eta)

    def log_z(self, parameter_type='regular', stop_gradient=False):
        if parameter_type == 'regular':
            pi = self.get_parameters('regular', stop_gradient=stop_gradient)
            eta = Stats.LogX(pi)
        else:
            eta = self.get_parameters('natural', stop_gradient=stop_gradient)[Stats.X]
        return T.logsumexp(eta, -1)
