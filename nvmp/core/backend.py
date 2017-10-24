import inspect
from deepx import T

def modify_method(func):
    from .node import Node
    def new_func(*args, **kwargs):
        args = list(map(lambda x: x.value() if isinstance(x, Node) else x, args))
        kwargs = {k: v.value() if isinstance(v, Node) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return new_func

for method_name in dir(T):
    method = getattr(T, method_name)
    if inspect.ismethod(method):
        setattr(T, method_name, modify_method(method))
