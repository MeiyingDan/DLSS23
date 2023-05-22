from Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        weights = np.random.uniform(low=0, high= 1,size=(input_size, output_size))
        self.optimizer = None
        self.gradient_weights = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.dot(input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        if self.optimizer is not None:
            self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor, self.weights.T)
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    