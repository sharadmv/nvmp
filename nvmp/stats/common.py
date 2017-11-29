from ..core import Node, AtomicNode

def convert_param(param):
    if isinstance(param, Node):
        return param.value()
    return param

class Distribution(AtomicNode):

    def __init__(self, parameters, parameter_type='regular', num_samples=None, **kwargs):
        self.parameter_cache = {}
        if isinstance(parameters, list):
            parameters = list(map(convert_param, parameters))
        else:
            parameters = convert_param(parameters)

        self.parameter_cache[parameter_type] = parameters
        super(Distribution, self).__init__(**kwargs)
        self._value = self.sample()[0]

    def get_parameters(self, parameter_type):
        if parameter_type not in self.parameter_cache:
            if parameter_type == 'regular':
                self.parameter_cache['regular'] = self.natural_to_regular(self.get_parameters('natural'))
            elif parameter_type == 'natural':
                self.parameter_cache['natural'] = self.regular_to_natural(self.get_parameters('regular'))
            elif parameter_type == 'packed':
                self.parameter_cache['packed'] = self.natural_to_packed(self.get_parameters('natural'))
        return self.parameter_cache[parameter_type]

    def get_stat(self, name, feed_dict={}):
        if self in feed_dict:
            return feed_dict[self][name]
        raise Exception('statistic not available')
        stats = self.expected_sufficient_statistics()
        if name in stats:
            return stats[name]
        raise Exception()

    def value(self):
        return self._value
