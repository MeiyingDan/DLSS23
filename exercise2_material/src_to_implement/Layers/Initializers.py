import numpy as np

class Constant:
    def __init__(self, inital_weight = 0.1):
        self.inital_weight = inital_weight

    def initialize(weights_shape, fan_in, fan_out):
        return np.full((fan_in, fan_out), weights_shape)

class UniformRandom:
    def initialize(weights_shape, fan_in=0, fan_out=1):
        return np.random.uniform(low=fan_in, high=fan_out, size=weights_shape)

class Xavier:
    def initialize(weights_shape, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=weights_shape)

class He(object):
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        return np.random.randn(*weights_shape) * sigma