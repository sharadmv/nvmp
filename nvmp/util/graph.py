import tensorflow as tf

def recompute_node(x, stop_nodes=set(), feed_dict={}):
    if x in feed_dict:
        return feed_dict[x]
    if x in stop_nodes:
        return x
    if x.op.node_def.op in {'VariableV2', 'Placeholder'}:
        return x
    idx = x.op.outputs.index(x)
    graph = tf.get_default_graph()
    attr_names = [name for name in x.op.node_def.attr.keys()]
    attr_dict = {attr: x.op.node_def.attr[attr] for attr in attr_names}
    inputs = [recompute_node(input, stop_nodes=stop_nodes, feed_dict=feed_dict) for input in x.op.inputs]
    return graph.create_op(x.op.node_def.op, inputs, x.op._output_types, attrs=attr_dict).outputs[idx]
