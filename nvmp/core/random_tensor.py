from abc import abstractmethod

from .tensor import Tensor

class RandomTensor(Tensor):

    def __init__(self):
        super(RandomTensor, self).__init__()
        self._value = None

    def sample(self, num_samples=None):
        if num_samples is None:
            if self._value is None:
                self._value = self._sample(1)[0]
            return self._value
        else:
            return self._sample(num_samples)

    @abstractmethod
    def log_likelihood(self, x):
        pass

    @abstractmethod
    def _sample(self, num_samples):
        pass

    def get_param_dim(self):
        return {
            s: s.out_dim() for s in self.statistics()
        }
