from contextlib import contextmanager

class Graph(object):

    def __init__(self, graph=None):
        if graph is not None:
            self.graph = graph.copy()
        else:
            self.graph = set([])

    def add(self, node):
        self.graph.add(node)

    def __iter__(self):
        return iter(self.graph)

    def __repr__(self):
        return 'Graph(%s)' % repr(self.graph)

    def __str__(self):
        return str(self.graph)

_DEFAULT_GRAPH = Graph()

GRAPH_STACK = [_DEFAULT_GRAPH]

def get_current_graph():
    return GRAPH_STACK[-1]

@contextmanager
def graph(**kwargs):
    graph = Graph(**kwargs)
    push_graph(graph)
    yield graph
    pop_graph(graph)

def push_graph(graph):
    GRAPH_STACK.append(graph)

def pop_graph(graph):
    GRAPH_STACK.pop(-1)
