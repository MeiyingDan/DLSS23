from Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self):
        super.__init__()

    def forward(self,input_tensor):
        '''retruns a tensor that serves as a input tensor for next layer'''
        self.input_tensor = input_tensor
        output_tensor = input_tensor*(input_tensor > 0)
        return output_tensor

    def backward(self,error_tensor):
        '''retruns a tensor that serves as a error tensor for previous layer'''
        return error_tensor*(self.input_tensor > 0)

