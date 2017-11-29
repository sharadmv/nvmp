import tensorflow as tf
import six
from abc import abstractmethod, ABCMeta

from deepx.stats import Distribution

from .graph import get_current_graph

@six.add_metaclass(ABCMeta)
class Node(object):

    def __init__(self, parameters, *args, **kwargs):
        pass

    @abstractmethod
    def value(self):
        pass

    @staticmethod
    def _overload_all_operators():
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            Node._overload_operator(operator)

    def __add__(self, right):
        from .ops import Add
        return Add(self, right)

    @staticmethod
    def _overload_operator(operator):
        def _run_op(a, *args):
            def convert(b):
                if isinstance(b, Node):
                    return b.value()
                else:
                    return b
            return getattr(tf.Tensor, operator)(a.value(), *map(convert, args))
        try:
            _run_op.__doc__ = getattr(tf.Tensor, operator).__doc__
        except AttributeError:
            pass

        setattr(Node, operator, _run_op)

    __array_priority__ = 100

    def parents(self):
        graph = get_current_graph()
        node_map = { node.value() : node for node in graph }
        parents = set([])
        def explore(v):
            for v_parent in v.op.inputs:
                if v_parent in node_map:
                    parents.add(node_map[v_parent])
                else:
                    explore(v_parent)
        explore(self.value())
        return parents

    def children(self):
        graph = get_current_graph()
        children = set()
        for node in graph:
            if self in node.parents():
                children.add(node)
        return children

    def markov_blanket(self):
        children = self.get_children()
        parents = self.parents()
        coparents = set([p for node in children for p in node.parents()]) - {self}
        return (children, parents, coparents)

    def get_stats(self, *names, feed_dict={}):
        return [self.get_stat(name, feed_dict=feed_dict) for name in names]

    @abstractmethod
    def get_stat(self, name):
        pass

    def _as_graph_element(self):
        return self.value()
