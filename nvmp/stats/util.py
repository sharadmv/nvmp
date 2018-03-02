from deepx import T

def kl_divergence(p, q):
    assert p.statistics() == q.statistics()
    param_dim = p.get_param_dim()
    dist = p.__class__
    p_param, q_param = p.get_parameters('natural'), q.get_parameters('natural')
    stats = p.statistics()
    p_stats = p.expected_sufficient_statistics()
    return (
        sum([T.sum((p_param[s] - q_param[s]) * p_stats[s], list(range(-param_dim[s], 0))) for s in stats])
        - p.log_z() + q.log_z()
    )
