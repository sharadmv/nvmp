from functools import wraps
import six
from deepx import T
from abc import ABCMeta, abstractmethod

def stats(*names):
    def wrapper(func):
        @wraps(func)
        def foo(*args):
            for arg in args:
                if isinstance(arg, SufficientStatistics):
                    pass
        return foo
    return wrapper

@six.add_metaclass(ABCMeta)
class SufficientStatistics(object):

    def __init__(self, stats):
        self.statistics = stats

    def get_stat(self, name):
        if name not in self.statistics:
            raise Exception("Statistic '%s' missing" % name)
        return self.statistics[name]

    def get_stats(self, *names):
        return [self.statistics[name] for name in names]

class Add(SufficientStatistics):

    def __init__(self, left, right):
        self.left, self.right = left, right

    def get_stat(self, name):
        if name == 'x':
            return self.left.get_stat(name) + self.right.get_stat(name)
        raise Exception()
