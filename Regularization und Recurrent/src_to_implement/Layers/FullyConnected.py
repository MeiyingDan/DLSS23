import numpy as np
from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        # self.gradient_weights = None
        self.optimizer = None
        self.input_tensor = None

    def forward(self, input_tensor):

        input_bias = np.ones((input_tensor.shape[0], 1))
        allinput_tensor = np.concatenate((input_tensor, input_bias), axis=1)
        self.input_tensor = allinput_tensor
        return np. dot(self.input_tensor, self.weights)            #output = input tensor for the next layer



    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)    #gradient
        if self.optimizer is not None:
            # self.optimizer = Sgd(0.025)
            updated_weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.weights = updated_weights
        return np.dot(error_tensor, self.weights.T)[:, : -1]        # error tensor for the previous layer


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size,), self.input_size, self.output_size)
        self.weights = np.concatenate((self.weights, self.bias), axis=0)



    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value






