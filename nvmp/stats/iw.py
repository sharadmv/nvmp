import numpy as np
from deepx import T
from deepx.stats import NIW

from .common import Distribution
from ..util import get_stat, get_stats

class InverseWishart(Distribution):

    def get_param_dim(self):
        return 2

    def log_likelihood(self, x, feed_dict={}):
        leading_dim = T.shape(x)[:-1]
        param1, param2 = self.get_parameters('regular')
        eta1, eta4 = get_stats(param1, '-0.5S^-1', '-0.5log|S|', feed_dict=feed_dict)
        mu, mumuT = get_stats(param2, 'x', 'xxT', feed_dict=feed_dict)
        eta2 = -2 * T.matmul(eta1, mu[..., None])[..., 0]
        natparam = NIW.pack([
            eta1,
            eta2,
            T.trace(mumuT * eta1),
            eta4,
        ])
        suff_stats = self.sufficient_statistics(x)
        stats = NIW.pack([
            suff_stats['xxT'],
            suff_stats['x'],
            T.ones(leading_dim),
            T.ones(leading_dim)
        ])
        return self.log_h(x) + T.sum(natparam * stats, axis=[-1, -2])

    def log_h(self, x):
        d = T.to_float(T.shape(x)[-1])
        return -0.5 * d * T.log(2 * np.pi)

    def sufficient_statistics(self, x):
        return {
            'xxT': T.outer(x, x),
            'x': x,
        }

    def expected_sufficient_statistics(self):
        sigma, mu = self.get_parameters('regular')
        eta1 = T.outer(mu, mu) + sigma
        eta2 = mu
        return {
            'xxT': eta1,
            'x': eta2,
        }

    def sample(self, num_samples=1):
        sigma, mu = self.get_parameters('regular')

        L = T.cholesky(sigma)
        sample_shape = T.concat([[num_samples], T.shape(mu)], 0)
        noise = T.random_normal(sample_shape)
        L = T.tile(L[None], T.concat([[num_samples], T.ones([T.rank(sigma)], dtype=np.int32)]))
        return mu[None] + T.matmul(L, noise[..., None])[..., 0]

        neghalfJ, h = self.unpack(self.get_parameters('natural'))
        sample_shape = T.concat([T.shape(h), [num_samples]], 0)
        J = -2 * neghalfJ
        L = T.cholesky(J)
        noise = T.matrix_solve(T.matrix_transpose(L), T.random_normal(sample_shape))
        noise = T.transpose(noise, T.concat([[T.rank(noise) - 1], T.range(0, T.rank(noise) - 1)], 0))
        return T.matrix_solve(J, h[..., None])[..., 0][None] + noise

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        sigma, mu = regular_parameters
        sigma_inv = T.matrix_inverse(sigma)
        eta1 = -0.5 * sigma_inv                              # -\frac{1}{2} \Sigma^{-1}
        eta2 = T.matrix_solve(sigma, mu[..., None])[..., 0] # \Sigma^{-1}\mu
        return Gaussian.pack([eta1, eta2])

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        J, h = Gaussian.unpack(natural_parameters)
        sigma_inv = -2 * J
        mu = T.matrix_solve(sigma_inv, h[..., None])[..., 0]
        return [T.matrix_inverse(sigma_inv), mu]

    @classmethod
    def pack(cls, natural_parameters):
        return natural_parameters

    @classmethod
    def unpack(cls, packed_parameters):
        return packed_parameters

    def log_z(self):
        sigma, mu = self.get_parameters('regular')
        sigma_inv = T.matrix_inverse(sigma)
        mu_shape = T.get_shape(mu)
        if len(mu_shape) == 1:
            return 0.5 * T.einsum('a,ab,b->', mu, sigma_inv, mu) + 0.5 * T.logdet(sigma)
        elif len(mu_shape) == 2:
            return 0.5 * T.einsum('ia,iab,ib->i', mu, sigma_inv, mu) + 0.5 * T.logdet(sigma)
