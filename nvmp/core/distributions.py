import inspect

from .. import stats
from ..core import Node

# def convert_class(cls):
    # def new_init(self, *args, **kwargs):
        # Node.__init__(self, *args, **kwargs)
        # cls.__init__(self, *args, **kwargs)
    # cls.__init__ = new_init
    # return cls

for var_name in dir(stats):
    var = getattr(stats, var_name)
    if inspect.isclass(var):
        mro = inspect.getmro(var)
        if stats.Distribution in mro:
            # var = convert_class(var)
            globals()[var_name] = type(var_name, (Node, var), {})
