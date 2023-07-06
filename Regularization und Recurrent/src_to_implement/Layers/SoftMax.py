import numpy as np
from Layers.Base import BaseLayer



class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        max_val = np.max(input_tensor)
        if max_val > 0:
            shift_val = abs(max_val) + 1e-6          # Translation constant, adding a small offset to avoid dividing by zero error
            input_tensor = input_tensor - shift_val

        exp_vals = np.exp(input_tensor)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        #exponentiated values along the axis 1 (rows),keeping the dimensions intact with the  argument
        self.output = exp_vals / sum_exp
        return self.output

    def backward(self, error_tensor):
        error_tensor = self.output*(error_tensor - np.sum(error_tensor * self.output, axis=1)[:, np.newaxis])
        # error_tensor = self.output * (error_tensor - np.sum(np.multiply(error_tensor, self.output), axis=1)[:, np.newaxis])
        return error_tensor
