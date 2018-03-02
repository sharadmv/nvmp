import numpy as np
from deepx import T

def coerce(x, shape=None):
    from .deterministic_tensor import DeterministicTensor
    if isinstance(x, float) or isinstance(x, int):
        return DeterministicTensor(T.constant(x))
    if isinstance(x, np.ndarray):
        return DeterministicTensor(T.constant(x))
    if isinstance(x, T.core.Tensor):
        return DeterministicTensor(x)
