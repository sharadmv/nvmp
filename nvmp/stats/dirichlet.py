import numpy as np
from deepx import T

from ..core import Stats
from .exponential_family import ExponentialFamily

class Dirichlet(ExponentialFamily):

    def shape(self):
        return T.shape(self.get_parameters('natural')[Stats.LogX])

    def _sample(self, num_samples):
        alpha = self.get_parameters('regular')
        d = self.shape()[-1]
        gammas = T.random_gamma([num_samples], alpha, beta=1)
        return gammas / T.sum(gammas, -1)[..., None]

    def statistics(self):
        return { Stats.LogX }

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        alpha = regular_parameters
        return {
            Stats.LogX: Stats.X(alpha) - 1.
        }

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        eta = natural_parameters[Stats.LogX]
        return eta + 1.

    def log_z(self, parameter_type='regular', stop_gradient=False):
        if parameter_type == 'regular':
            alpha = self.get_parameters('regular', stop_gradient=stop_gradient)
        else:
            alpha = self.natural_to_regular(self.get_parameters('natural', stop_gradient=stop_gradient))
        return T.sum(T.gammaln(alpha), -1) - T.gammaln(T.sum(alpha, -1))
