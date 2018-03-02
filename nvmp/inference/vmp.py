from deepx import T
from ..stats import Gaussian, IW
from .util import top_sort
from .message_passing import message_passing

def initialize_node(node, children):
    if isinstance(node, Gaussian):
        d = T.shape(node)
        return Gaussian([T.eye(d[-1], batch_shape=d[:-1]), T.random_normal(d)])
    elif isinstance(node, IW):
        d = T.shape(node)
        return IW([(T.to_float(d[-1]) + 1) * T.eye(d[-1], batch_shape=d[:-2]), T.to_float(d[-1]) + 1])


def vmp(graph, data, max_iter=100, tol=1e-4):
    q, visible = {}, {}
    for node in top_sort(graph)[::-1]:
        if node in data:
            visible[node] = T.to_float(data[node])
        else:
            q[node] = initialize_node(node, {})

    ordering = list(q.keys())
    params = [q[var].get_parameters('natural') for var in ordering]
    prev_elbo = T.constant(float('inf'))
    def cond(i, elbo, prev_elbo, q):
        return T.logical_and(i < max_iter, abs(elbo - prev_elbo) > tol)
    def step(i, elbo, prev_elbo, q):
        prev_elbo = elbo
        q_vars = {var:var.__class__(param, 'natural') for var, param in zip(ordering, q)}
        q, elbo = message_passing(q_vars, visible)
        return i + 1, elbo, prev_elbo, [q[var].get_parameters('natural') for var in ordering]
    i, elbo, prev_elbo, q = T.while_loop(cond, step,
                                 [0, float('inf'), 0.0, params])
    return {var:var.__class__(param, 'natural') for var, param in zip(ordering, q)}, elbo
