import tensorflow as tf
from tensorflow.python.client.session import \
        register_session_run_conversion_functions
from abc import ABCMeta, abstractmethod

from deepx import T

from .util import coerce
from .graph import get_current_graph
from .context import context, current_scope
from .statistics import Stats

class Tensor(object, metaclass=ABCMeta):

    def __init__(self):
        self._node_map = None
        self._graph = get_current_graph()
        self.scope = current_scope()
        self._stats = {}

    @abstractmethod
    def shape(self):
        pass

    def ndim(self):
        return T.core.size(self.shape())

    @abstractmethod
    def sample(self):
        pass

    def statistic(self, stat):
        if stat not in self._stats:
            self._stats[stat] = self._statistic(stat)
        return self._stats[stat]

    @abstractmethod
    def _statistic(self, stat):
        pass

    @abstractmethod
    def statistics(self):
        pass

    def __add__(self, other):
        from .binary_ops import Add
        return Add(self, coerce(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .binary_ops import Add
        return Add(self, -coerce(other))

    def __rsub__(self, other):
        from .binary_ops import Add
        return Add(-self, coerce(other))

    def __mul__(self, other):
        from .binary_ops import ScalarMul
        return ScalarMul(coerce(other), self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def tile(self, num):
        from .unary_ops import Tile
        return Tile(self, num)

    def __getitem__(self, z):
        from ..stats import Categorical
        if isinstance(z, Categorical):
            from .mixture import Mixture
            return Mixture(self, z)
        raise NotImplementedError

    @staticmethod
    def _session_run_conversion_fetch_function(tensor):
        return ([tensor.sample()], lambda val: val[0])

    @staticmethod
    def _session_run_conversion_feed_function(feed, feed_val):
        return [(feed.sample(), feed_val)]

    @staticmethod
    def _session_run_conversion_feed_function_for_partial_run(feed):
        return [feed.sample()]

    @staticmethod
    def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(v.dtype):
            raise ValueError(
                "Incompatible type conversion requested to type '%s' for variable "
                "of type '%s'" % (dtype.name, v.dtype.name))
        return v.sample()

    def node_map(self):
        if self._node_map is None:
            node_map = {}
            for node in self._graph:
                stats = {stat(node) for stat in node.statistics()}
                for stat in stats:
                    node_map[stat] = node
            self._node_map = node_map
        return self._node_map

    def parents(self):
        graph = self._graph
        if graph is None:
            raise Exception("RV not part of a graph")
        node_map = self.node_map()
        parents = set([])
        def explore(v):
            for v_parent in v.op.inputs:
                if v_parent in node_map:
                    parents.add(node_map[v_parent])
                else:
                    explore(v_parent)
        [explore(s(self)) for s in self.statistics()]
        return {context(p) for p in parents}

    def children(self):
        graph = self._graph
        if graph is None:
            raise Exception("RV not part of a graph")
        children = set()
        node_map = self.node_map()
        for node in graph:
            parents = set([])
            def explore(v):
                for v_parent in v.op.inputs:
                    if v_parent in node_map:
                        parents.add(node_map[v_parent])
                    else:
                        explore(v_parent)
            explore(node.sample())
            if self in parents:
                children.add(node)
        return {context(c) for c in children}

    def markov_blanket(self):
        children = self.children()
        parents = self.parents()
        coparents = set([p for node in children for p in node.parents()]) - {self}
        return (children, parents, coparents)

register_session_run_conversion_functions(
    Tensor,
    Tensor._session_run_conversion_fetch_function,
    Tensor._session_run_conversion_feed_function,
    Tensor._session_run_conversion_feed_function_for_partial_run)

tf.register_tensor_conversion_function(
    Tensor, Tensor._tensor_conversion_function)
