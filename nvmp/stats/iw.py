import numpy as np
from deepx import T
from deepx.stats import NIW

from ..core import Stats
from .exponential_family import ExponentialFamily
from .gaussian import Gaussian

class InverseWishart(ExponentialFamily):

    def shape(self):
        return T.shape(self.get_parameters('natural')[Stats.HSI])

    def _sample(self, num_samples):
        S, nu = self.get_parameters('regular')
        nu_ = T.cast(nu, T.core.int32)
        samples = Gaussian([T.tile(T.matrix_inverse(S)[None],
                                   T.concat([[num_samples], T.ones([self.ndim()], dtype=T.core.int32)], 0)),
                            T.zeros(T.concat([[num_samples], self.shape()[:-1]], 0))]).sample(num_samples=nu_)
        return T.matrix_inverse(T.sum(T.outer(samples, samples), 0))


    def statistics(self):
        return { Stats.HSI, Stats.HLDS }

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        S, nu = regular_parameters
        d = T.to_float(T.shape(S)[-1])
        return {
            Stats.HSI: S,
            Stats.HLDS: nu + d + 1,
        }

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        S = natural_parameters[Stats.HSI]
        d = T.to_float(T.shape(S)[-1])
        return [S, natural_parameters[Stats.HLDS] - d - 1]

    def log_z(self, parameter_type='natural', stop_gradient=False):
        if parameter_type == 'regular':
            S, nu = self.get_parameters('regular', stop_gradient=stop_gradient)
        elif parameter_type == 'natural':
            natparam = self.get_parameters('natural', stop_gradient=stop_gradient)
            S, nu = self.natural_to_regular(natparam)
        d = T.to_float(T.shape(S)[-1])
        return (
            0.5 * nu * (d * T.log(2.) - T.logdet(S))
            + T.multigammaln(nu / 2., d)
        )

IW = InverseWishart
