from deepx import T

from ..core import Node, get_current_graph

def get_stats(x, *names, feed_dict={}):
    return [get_stat(x, name, feed_dict=feed_dict) for name in names]

def get_stat(x, name, feed_dict={}):
    node = get_current_graph().get_node(x)
    print(x, name)
    if node is not None:
        return node.get_stat(name, feed_dict=feed_dict)
    if name == 'x':
        return x
    elif name == 'xxT':
        return T.outer(x, x)
    elif name == '-0.5S^-1':
        return -0.5 * T.matrix_inverse(x)
    elif name == '-0.5log|S|':
        return -0.5 * T.logdet(x)
    raise Exception()
