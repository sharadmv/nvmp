import tensorflow as tf
from deepx import T

from .conditional import complete_conditional
from ..core import get_current_graph, RandomVariable, graph_context, context
from ..stats import kl_divergence
from .util import top_sort

def message_passing(hidden, visible):
    elbo = 0.0
    for var in top_sort(hidden)[::-1]:
        child_messages = [get_child_message(var, c, hidden={k:v for k, v in hidden.items() if
                                                        k != var}, visible=visible) for c in var.children()]
        stats = var.statistics()
        parent_message = var.get_parameters('natural')
        e_p = var.__class__(parent_message, 'natural', graph=False)
        natparam = {
            s: parent_message[s] + sum([child_message[s] for child_message in child_messages]) for s in stats
        }
        q = var.__class__(natparam, 'natural', graph=False)
        elbo -= kl_divergence(q, e_p)
        hidden[var] = q
    for var in visible:
        with graph_context(hidden):
            elbo += T.sum(var.log_likelihood(visible[var]))
    return hidden, elbo

def get_child_message(x, y, hidden={}, visible={}):
    with graph_context({**hidden, **visible}):
        data = context(y)
        log_likelihood = y.log_likelihood(data)
    stats = x.statistics()
    param = T.grad(T.sum(log_likelihood), [x.get_statistic(s) for s in stats])
    return {
        s:param[i] for i, s in enumerate(stats)
    }
