import copy
from Optimization.Loss import CrossEntropyLoss
#form Layers import Initializers


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None



    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output_tensor = self.loss_layer.forward(input_tensor, self.label_tensor)
        return output_tensor

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            Loss = self.forward()
            self.loss.append(Loss)
            errer = self.backward()

    def test(self, input_tensor):
        output = input_tensor

        for layer in self.layers:
            output = layer.forward(output)

        return output
        
def append_layer(layer):
    layer.initalize()

