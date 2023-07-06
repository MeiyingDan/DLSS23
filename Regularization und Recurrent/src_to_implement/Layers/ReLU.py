import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        relu_grad = np.where(self.input_tensor > 0, 1, 0)     #np.where(condition, x, y)
        return error_tensor * relu_grad















