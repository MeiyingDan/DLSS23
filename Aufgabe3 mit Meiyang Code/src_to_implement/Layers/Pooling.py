import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.lastShape = input_tensor.shape
        input_height = np.ceil((input_tensor.shape[2] - self.pooling_shape[0] + 1) / self.stride_shape[0]) #claculating size of the pooling layer in 
        input_width = np.ceil((input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1])  #to given input tensor 
        output_tensor = np.zeros((*input_tensor.shape[0:2], int(input_height), int(input_width)))
        self.x_s = np.zeros((*input_tensor.shape[0:2], int(input_height), int(input_width)), dtype=int) #making placeholders
        self.y_s = np.zeros((*input_tensor.shape[0:2], int(input_height), int(input_width)), dtype=int) #to store max. values 
        
        a = -1
        # implimenting typical two loop strucutre to iterate over a (mXn) matrix 
        for i in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0]):
            a += 1
            b = -1
            for j in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1]):
                b += 1
                temp = input_tensor[:, :, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]].reshape(*input_tensor.shape[0:2], -1)
                output_pos = np.argmax(temp, axis = 2)
                x = output_pos // self.pooling_shape[1]
                y = output_pos % self.pooling_shape[1]
                self.x_s[:, :, a, b] = x
                self.y_s[:, :, a, b] = y
                output_tensor[:, :, a, b] = np.choose(output_pos, np.moveaxis(temp, 2, 0))         

        return output_tensor
    
    def backward(self, error_tensor):
        return_tensor = np.zeros(self.lastShape) # This tensor will hold the gradient values 
                                                # that will be passed back to the previous layer during the backward pass
        for a in range(self.x_s.shape[0]):
            for b in range(self.x_s.shape[1]):
                for i in range(self.x_s.shape[2]):
                    for j in range(self.y_s.shape[3]): #using the update formula 
                        return_tensor[a, b, i*self.stride_shape[0]+self.x_s[a, b, i, j], j*self.stride_shape[1]+self.y_s[a, b, i, j]] += error_tensor[a, b, i, j]
        return return_tensor
