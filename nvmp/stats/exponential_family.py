from functools import lru_cache
from deepx import T

from abc import abstractmethod
from .distribution import Distribution

class ExponentialFamily(Distribution):

    def __init__(self, parameters, parameter_type='regular'):
        self.parameter_cache = {}
        self.parameter_cache[parameter_type, False] = parameters
        self._estats = None
        super(ExponentialFamily, self).__init__()

    def get_parameters(self, parameter_type, stop_gradient=False):
        if (parameter_type, stop_gradient) not in self.parameter_cache:
            if parameter_type == 'natural':
                if stop_gradient:
                    self.parameter_cache[parameter_type, stop_gradient] = {
                        k: T.core.stop_gradient(v)
                        for k, v in self.get_parameters('natural', stop_gradient=False).items()
                    }
                else:
                    self.parameter_cache[parameter_type, stop_gradient] = self.regular_to_natural(self.get_parameters('regular'))
            elif parameter_type == 'regular':
                if stop_gradient:
                    self.parameter_cache[parameter_type, stop_gradient] = [
                        T.core.stop_gradient(v) for v in self.get_parameters('regular', stop_gradient=False)
                    ]
                else:
                    self.parameter_cache[parameter_type, stop_gradient] = self.regular_to_natural(self.get_parameters('regular'))
        return self.parameter_cache[parameter_type, stop_gradient]

    @lru_cache(maxsize=None)
    def sufficient_statistics(self, x):
        return {
            s: s(x) for s in self.statistics()
        }

    def log_likelihood(self, x):
        natparam = self.get_parameters('natural')
        stats = self.sufficient_statistics(x)
        return (
            sum(T.sum(stats[stat] * natparam[stat], list(range(-stat.out_dim(), 0)))
                for stat in self.statistics()
            ) - self.log_z()
        )

    def _statistic(self, stat):
        return self.expected_sufficient_statistics()[stat]

    def expected_sufficient_statistics(self):
        if self._estats is None:
            natparam = self.get_parameters('natural', stop_gradient=True)
            stats = list(self.statistics())
            log_z = self.log_z('natural', stop_gradient=True)
            grads = T.grad(log_z, [natparam[s] for s in stats])
            self._estats = {
                s: g for s, g in zip(stats, grads)
            }
        return self._estats

    def as_variable(self):
        params = self.get_parameters('natural')
        return self.__class__({
            stat: T.variable(params[stat])
            for stat in self.statistics()
        }, 'natural')
