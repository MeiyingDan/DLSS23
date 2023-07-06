import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        flattened_tensor = input_tensor.reshape(self.input_shape[0], -1)       # batch_size, *spatial_dim, channels
        return flattened_tensor

    def backward(self, error_tensor):
        reshaped_error = error_tensor.reshape(self.input_shape)
        return reshaped_error