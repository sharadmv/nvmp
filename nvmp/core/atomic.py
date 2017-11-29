from .node import Node
from .graph import get_current_graph

class AtomicNode(Node):

    def __init__(self, num_samples=None):
        if num_samples is None:
            self._value = self.sample(1)[0]
        else:
            self._value = self.sample(num_samples=num_samples)
        get_current_graph().add(self)
