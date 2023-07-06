import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.alpha = 0.8
        self.epsilon = np.finfo(float).eps
        self.running_mean = np.zeros(self.channels)
        self.running_variance = np.zeros(self.channels)
        self.optimizer = None

        self.weights = np.ones((self.channels))
        self.bias = np.zeros((self.channels))
        self.gradient_weights = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.channels,), self.channels, self.channels)
        self.bias = bias_initializer.initialize((self.channels,), self.channels, self.channels)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 4:
            self.B, self.H, self.M, self.N = input_tensor.shape
            self.input_tensor = self.reformat(input_tensor)

        if self.testing_phase == False:
            self.mean = np.mean(self.input_tensor, axis=0)
            self.variance = np.var(self.input_tensor, axis=0)
            self.running_mean = np.copy(self.mean)
            self.running_variance = np.copy(self.variance)

        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_variance = self.alpha * self.running_variance + (1 - self.alpha) * self.variance

        if self.testing_phase:
            self.x_hat = (self.input_tensor - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
        else:
            self.x_hat = (self.input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)

        self.output_tensor = self.weights * self.x_hat + self.bias

        if len(input_tensor.shape) == 4:
            self.output_tensor = self.reformat(self.output_tensor)
            self.input_tensor = self.reformat(self.input_tensor)
        return self.output_tensor


    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
        else:
            input_tensor = self.input_tensor

        self.gradient_weights = np.sum(error_tensor * self.x_hat, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        gradient_x = compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean, self.variance, self.epsilon)

        if len(self.input_tensor.shape) == 4:
            gradient_x = self.reformat(gradient_x)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        return gradient_x

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            self.B, self.H, self.M, self.N = tensor.shape
            reshaped_tensor = np.reshape(tensor, (self.B, self.H, self.M * self.N))
            transposed_tensor = np.transpose(reshaped_tensor, (0, 2, 1))
            final_tensor = np.reshape(transposed_tensor, (-1, self.H))
        else:
            reshaped_tensor = np.reshape(tensor, (self.B, -1, self.H))
            transposed_tensor = np.transpose(reshaped_tensor, (0, 2, 1))
            final_tensor = np.reshape(transposed_tensor, (self.B, self.H, self.M, self.N))
        return final_tensor
