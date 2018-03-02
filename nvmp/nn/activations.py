from deepx import T
from deepx.nn import Linear

from .util import log1pexp
from .. import stats as dist


__all__ = ['Gaussian', 'Bernoulli', 'Dummy']


class Gaussian(Linear):

    def __init__(self, *args, **kwargs):
        self.cov_type = kwargs.pop('cov_type', 'diagonal')
        super(Gaussian, self).__init__(*args, **kwargs)
        assert not self.elementwise

    def get_dim_out(self):
        return [self.dim_out[0] * 2]

    def activate(self, X):
        if self.cov_type == 'diagonal':
            sigma, mu = T.split(X, 2, axis=-1)
            sigma = T.matrix_diag(log1pexp(sigma))
            return [sigma, mu]
        raise Exception("Undefined covariance type: %s" % self.cov_type)

    def __str__(self):
        return "Gaussian(%s)" % self.dim_out


class Bernoulli(Linear):

    def __init__(self, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)

    def activate(self, X):
        if self.elementwise:
            return dist.Bernoulli(X, parameter_type='regular')
        return dist.Bernoulli(T.sigmoid(X), parameter_type='regular')

    def __str__(self):
        return "Bernoulli(%s)" % self.dim_out


class Dummy(object):

    def __init__(self, dim):
        self.dim = dim

    def initialize(self):
        pass

    def get_parameters(self):
        return []

    def __call__(self, x):
        shape = T.shape(x)[:-1]
        return dist.Gaussian([1e-4 * T.eye(self.dim, batch_shape=shape), x])
