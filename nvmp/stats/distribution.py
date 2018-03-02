from functools import lru_cache
from deepx import T

from abc import abstractmethod
from ..core import RandomTensor, get_current_graph

class Distribution(RandomTensor):

    def __init__(self, graph=True):
        if graph and get_current_graph() is not None:
            get_current_graph().add(self)
        super(Distribution, self).__init__()
