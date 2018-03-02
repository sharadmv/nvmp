from enum import Enum
from abc import ABCMeta, abstractmethod

from deepx import T

from .context import context

__all__ = ['Stats']

class Statistic(object, metaclass=ABCMeta):

    @abstractmethod
    def in_dim(self):
        pass

    @abstractmethod
    def out_dim(self):
        pass

    @abstractmethod
    def compute(self, x):
        pass

class X(Statistic):

    def in_dim(self):
        return 1

    def out_dim(self):
        return 1

    def compute(self, x):
        return x

class XXT(Statistic):

    def in_dim(self):
        return 1

    def out_dim(self):
        return 2

    def compute(self, x):
        return T.outer(x, x)

class HSI(Statistic):

    def in_dim(self):
        return 2

    def out_dim(self):
        return 2

    def compute(self, A):
        return -0.5 * T.matrix_inverse(A)

class HLDS(Statistic):

    def in_dim(self):
        return 2

    def out_dim(self):
        return 0

    def compute(self, A):
        return -0.5 * T.logdet(A)

class LogX(Statistic):

    def in_dim(self):
        return 1

    def out_dim(self):
        return 1

    def compute(self, x):
        return T.log(x)

class Stats(Enum):
    X = X()
    XXT = XXT()
    HSI = HSI()
    HLDS = HLDS()
    LogX = LogX()

    def __call__(self, x):
        from .tensor import Tensor
        if not isinstance(x, Tensor):
            return self.value.compute(x)
        # if x.ndim() != self.value.in_dim():
            # raise Exception("Can't compute stat: incorrect dim")
        return context(x).statistic(self)

    def in_dim(self):
        return self.value.in_dim()

    def out_dim(self):
        return self.value.out_dim()
