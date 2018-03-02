from contextlib import contextmanager

class Graph(object):

    def __init__(self, graph=None):
        if graph is not None:
            self.graph = graph.copy()
        else:
            self.graph = set([])

    def get_node(self, x):
        for node in self.graph:
            if x == node.value():
                return node

    def add(self, node):
        self.graph.add(node)

    def __iter__(self):
        return iter(self.graph)

    def __repr__(self):
        return 'Graph(%s)' % repr(self.graph)

    def __str__(self):
        return str(self.graph)

_DEFAULT_GRAPH = Graph()

GRAPH_STACK = []

@contextmanager
def graph():
    GRAPH_STACK.append(Graph())
    yield get_current_graph()
    GRAPH_STACK.pop(-1)

def in_graph():
    return len(GRAPH_STACK) > 0

def get_current_graph():
    if len(GRAPH_STACK) == 0:
        return None
    return GRAPH_STACK[-1]
