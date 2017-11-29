import inspect
from functools import wraps

from deepx import T

class Trace(object):

    def __init__(self, func, inputs):
        self.func = func.__name__
        self.inputs = inputs

def wrap_func(func):
    @wraps(func)
    def foo(*args, **kwargs):
        output = func(*args, **kwargs)
        output._trace = Trace(func, args)
        return output
    return foo

WRAPPED_METHODS = {
    'abs', 'sum', 'log'
}

for method in WRAPPED_METHODS:
    print("Modifying", method)
    setattr(T, method, wrap_func(getattr(T, method)))

# def modify_method(func):
    # from .node import Node
    # def new_func(*args, **kwargs):
        # args = list(map(lambda x: x.value() if isinstance(x, Node) else x, args))
        # kwargs = {k: v.value() if isinstance(v, Node) else v for k, v in kwargs.items()}
        # return func(*args, **kwargs)
    # return new_func

# for method_name in dir(T):
    # method = getattr(T, method_name)
    # if inspect.ismethod(method):
        # setattr(T, method_name, modify_method(method))
