from deepx import T
from ..core import get_current_graph, graph_context

def complete_conditional(var, data={}):
    graph = get_current_graph()
    children, parents, coparents = var.markov_blanket()
    if not(len(parents.intersection(data)) == len(parents) \
        and len(children.intersection(data)) == len(children) \
        and len(coparents.intersection(data)) == len(coparents)):
        raise Exception('markov blanket not specified')

    stats = var.statistics()
    child_messages = [get_child_message(var, c, data=data) for c in children]
    parent_message = var.get_parameters('natural')
    return var.__class__({
        s: parent_message[s] + sum([child_message[s] for child_message in child_messages]) for s in stats
    }, 'natural', graph=False)

    return child_messages

def get_child_message(x, y, data={}):
    y_ = data[y]
    stats = x.statistics()
    log_likelihood = y.log_likelihood(y_)
    param = T.grad(T.sum(log_likelihood), [x.get_statistic(s) for s in stats])
    return {
        s:param[i] for i, s in enumerate(stats)
    }
