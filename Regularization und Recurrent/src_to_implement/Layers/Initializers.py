import numpy as np


class Constant:
    def __init__(self, weight_initialization=0.1):
        self.weight_initialization = weight_initialization

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.weight_initialization)


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        standard_deviation = np.sqrt(2/(fan_in + fan_out))
        return np.random.normal(loc=0, scale=standard_deviation, size=weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        standard_deviation = np.sqrt(2/fan_in)
        return np.random.normal(loc=0, scale=standard_deviation, size=weights_shape)





