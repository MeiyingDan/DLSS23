import numpy as np
from scipy.signal import convolve, convolve2d, correlate2d, correlate


class Conv(): # wo ist die Inheritance definiert ?? 
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        # super().__init__()
        self.stride_shape = stride_shape
        if type(stride_shape) == int:
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        if (len(self.convolution_shape) == 3):
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
        
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, size=(self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, (self.num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None
        self.trainable = True
        self.optimizer_weights = None   # 优化器对象，用于优化权重
        self.optimizer_bias = None

    def forward(self, input_tensor):
        if len(input_tensor.shape) == 3:
            batch_size, input_channels, input_height = input_tensor.shape
            input_width = 1
        else:
            batch_size, input_channels, input_height, input_width = input_tensor.shape


        if len(self.stride_shape) == 1:
            stride_height = self.stride_shape[0]
            stride_width = 1
        else:
            stride_height, stride_width = self.stride_shape


        if len(self.convolution_shape) == 2:
            cov_channels, conv_height = self.convolution_shape
            conv_width = 1
        else:
            cov_channels, conv_height, conv_width = self.convolution_shape

        # padding_height = (conv_height - 1) // 2
        # padding_width = (conv_width - 1) // 2
        # Perform zero-padding on input tensor
        if (conv_height - 1) % 2 == 0:
            padding_heightL, padding_heightR = (conv_height - 1) // 2, (conv_height - 1) // 2
        else:
            padding_heightL, padding_heightR = (conv_height - 1) // 2, (conv_height - 1) // 2 + 1

        if (conv_width - 1) % 2 == 0:
            padding_widthL, padding_widthR = (conv_width - 1) // 2, (conv_width - 1) // 2
        else:
            padding_widthL, padding_widthR = (conv_width - 1) // 2, (conv_width - 1) // 2 + 1


        padded_input = np.pad(input_tensor,
                              ((0, 0), (0, 0), (padding_heightL, padding_heightR), (padding_widthL, padding_widthR)),
                              mode='constant')

        output_height = (input_height - conv_height + (padding_heightL + padding_widthR)) // stride_height + 1
        output_width = (input_width - conv_width + (padding_widthL + padding_widthR)) // stride_width + 1

        output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

        for b in range(batch_size):
            for k in range(self.num_kernels):
                kernel = self.weights[k]
                if len(self.convolution_shape) == 2:
                    output_tensor[b, k, :, :] = correlate2d(padded_input[b], kernel, mode='valid')
                    # output_tensor[b, k, :, :] = correlate2d(input_tensor[b], kernel, mode='same')
                else:
                    output_tensor[b, k, :, :] = correlate(padded_input[b], kernel, mode='valid')
        output_tensor += self.bias
        return output_tensor



    def backward(self, error_tensor):
        batch_size, _, output_height, output_width = error_tensor.shape
        # input_channels, input_height, input_width = self.input_shape
        #
        # # Update weights and bias using the optimizer
        # if self.optimizer is not None:
        #     self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        #     self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        #
        # gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        # if len(self.convolution_shape) == 2:
        #     # 2D convolution
        #     # Perform correlation between error tensor and flipped kernels
        #     flipped_kernels = np.flip(self.weights, axis=(1, 2))
        #     error_tensor = correlate(error_tensor, flipped_kernels, mode='full')
        # else:
        #     flipped_kernels = np.flip(self.weights, axis=(1, 2, 3))
        #     error_tensor = correlate(error_tensor, flipped_kernels, mode='full')
        #
        # return error_tensor

        # if len(self.stride_shape) == 1:
        #     stride_height = self.stride_shape[0]
        #     stride_width = 1
        # else:
        #     stride_height, stride_width = self.stride_shape

        if len(self.convolution_shape) == 2:
            cov_channels, conv_height = self.convolution_shape
            conv_width = 1
        else:
            cov_channels, conv_height, conv_width = self.convolution_shape

        padding_height = conv_height // 2
        padding_width = conv_width // 2

        padded_input = np.pad(self.input_tensor,
                              ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
                              mode='constant')
        padded_error = np.pad(error_tensor,
                              ((0, 0), (0, 0), (conv_height - 1, conv_height - 1), (conv_width - 1, conv_width - 1)),
                              mode='constant')

        self.gradient_weights = np.zeros_like(self.weights)
        gradient_input = np.zeros_like(padded_input)

        for b in range(batch_size):
            for k in range(self.num_kernels):
                if len(self.convolution_shape) == 2:
                    self.gradient_weights[k, :, :] += convolve(padded_input[b, :, :, :], padded_error[b, k, :, :], mode='valid')
                    gradient_input[b, :, :, :] += convolve(padded_error[b, k, :, :], self.weights[k, :, :][np.newaxis, :, :], mode='full')
                else:
                    self.gradient_weights[k, :, :, :] += convolve(padded_input[b, :, :, :], padded_error[b, k, :, :], mode='valid')
                    gradient_input[b, :, :, :] += convolve(padded_error[b, k, :, :], self.weights[k, :, :, :][np.newaxis, :, :, :], mode='full')



        return gradient_input[:, :, padding_height:-padding_height, padding_width:-padding_width]

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape),
                                                      np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.num_kernels, self.num_kernels, self.num_kernels)

    # @property
    # def optimizer(self):
    #     # return self.optimizer_weights, self.optimizer_bias
    #     return self._optimizer
    #
    # @optimizer.setter
    # def optimizer(self, optimizer):
    #     # self.optimizer_weights = optimizer.copy()
    #     # self.optimizer_bias = optimizer.copy()
    #     self._optimizer = optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):    #return the gradient with respect to the weights and bias, after they have been calculated in the backward-pass.
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property                   #return the gradient with respect to the weights and bias, after they have been calculated in the backward-pass.
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

