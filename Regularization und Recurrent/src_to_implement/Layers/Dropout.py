import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase == False:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape) / self.probability
            #np.random.binomial返回每次事件发生的次数（不是1 就是0，因为试验次数为1），随机生成一个0、1的向量    并且最后进行缩放（rescale）
            output_tensor = input_tensor * self.mask
        else:
            output_tensor = input_tensor
        return output_tensor


    def backward(self, error_tensor):
        if self.testing_phase == False:
            error_tensor *= self.mask
        return error_tensor