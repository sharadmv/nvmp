from functools import lru_cache
from tensorflow.python.framework import tensor_util

@lru_cache()
def trace(x):
    op_type = x.op.type
    parents = x.op.inputs
    parent_traces = list(map(trace, parents))
    if op_type == 'Const':
        return "(Const %s)" % parse_dimensions(x.op.get_attr('value'))
    if len(parent_traces) == 0:
        return op_type
    return "(%s %s)" % (
        op_type,
        " ".join(map(trace, parents))
    )

def parse_dimensions(value_proto):
    # shape = value_proto.tensor_shape.dim
    tensor_content = tensor_util.MakeNdarray(value_proto)
    # if value_proto.dtype == 1:
        # value = value_proto.float_val
    # else:
        # value = value_proto.tensor_content
    return tensor_content#, "%s %s" % (value, [s.size for s in shape])
