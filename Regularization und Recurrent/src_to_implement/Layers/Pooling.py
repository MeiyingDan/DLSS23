import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        # self.mask = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, input_channels, input_height, input_width = self.input_tensor.shape
        stride_height, stride_width = self.stride_shape
        pooling_height, pooling_width = self.pooling_shape

        output_height = (input_height - pooling_height)//stride_height + 1
        output_width = (input_width - pooling_width)//stride_width + 1
        output_tensor = np.zeros((batch_size, input_channels, output_height, output_width))

        # self.mask = np.zeros_like(input_tensor)

        for b in range(batch_size):
            for c in range(input_channels):
                for y in range(output_height):
                    for x in range(output_width):
                        input_window = self.input_tensor[b, c, y * stride_height: y * stride_height + pooling_height,
                                x * stride_width: x * stride_width + pooling_width]
                        output_tensor[b, c, y, x] = np.max(input_window)
        return output_tensor

    def backward(self, error_tensor):
        batch_size, input_channels, output_height, output_width = error_tensor.shape

        error_input = np.zeros_like(self.input_tensor)

        for b in range(batch_size):
            for c in range(input_channels):
                for y in range(output_height):
                    for x in range(output_width):
                        mask_window = self.input_tensor[b, c,
                                      y * self.stride_shape[0]:y * self.stride_shape[0] + self.pooling_shape[0],
                                      x * self.stride_shape[1]:x * self.stride_shape[1] + self.pooling_shape[1]]
                        error = error_tensor[b, c, y, x]
                        max_value = np.max(mask_window)
                        max_pos = np.where(mask_window == max_value)
                        error_input[b, c, y * self.stride_shape[0] + max_pos[0][0],
                        x * self.stride_shape[1] + max_pos[1][0]] += error

        return error_input

