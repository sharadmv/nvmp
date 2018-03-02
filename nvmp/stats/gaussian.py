import numpy as np
from deepx import T

from ..core import Stats
from .exponential_family import ExponentialFamily

class Gaussian(ExponentialFamily):

    def shape(self):
        return T.shape(self.get_parameters('natural')[Stats.X])

    def _sample(self, num_samples):
        sigma, mu = self.natural_to_regular(self.regular_to_natural(self.get_parameters('regular')))

        L = T.cholesky(sigma)
        sample_shape = T.concat([[num_samples], T.shape(mu)], 0)
        noise = T.random_normal(sample_shape)
        L = T.tile(L[None], T.concat([[num_samples], T.ones([T.rank(sigma)], dtype=np.int32)]))
        return mu[None] + T.matmul(L, noise[..., None])[..., 0]

    def statistics(self):
        return { Stats.X, Stats.XXT }

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        sigma, mu = regular_parameters
        J = Stats.HSI(sigma)
        return {
            Stats.XXT: J,
            Stats.X: (-(2 * J)@Stats.X(mu)[..., None])[..., 0],
        }

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        J, m = natural_parameters[Stats.XXT], natural_parameters[Stats.X]
        sigma = -0.5 * T.matrix_inverse(J)
        mu = T.matmul(sigma, m[..., None])[..., 0]
        return [sigma, mu]

    def log_z(self, parameter_type='regular', stop_gradient=False):
        if parameter_type == 'regular':
            sigma, mu = self.get_parameters('regular', stop_gradient=stop_gradient)
            d = T.to_float(self.shape()[-1])
            hsi, hlds = Stats.HSI(sigma), Stats.HLDS(sigma)
            mmT = Stats.XXT(mu)
            return (
                - T.sum(hsi * mmT, [-1, -2]) - hlds
                + d / 2. * np.log(2 * np.pi)
            )
        else:
            natparam = self.get_parameters('natural', stop_gradient=stop_gradient)
            d = T.to_float(self.shape()[-1])
            J, m = natparam[Stats.XXT], natparam[Stats.X]
            return (
                - 0.25 * (m[..., None, :]@T.matrix_inverse(J)@m[..., None])[..., 0, 0]
                - 0.5 * T.logdet(-2 * J)
                + d / 2. * np.log(2 * np.pi)
            )
