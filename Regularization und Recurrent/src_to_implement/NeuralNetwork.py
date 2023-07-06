import copy
from Layers.Base import BaseLayer


class NeuralNetwork(BaseLayer):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__()
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.regularization_norm = 0
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer


    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = self.phase

            input_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer.regularizer is not None:
                self.regularization_norm += layer.optimizer.regularizer.norm(layer.weights)
        output = self.loss_layer.forward(input_tensor, self.label_tensor)
        output += self.regularization_norm
        return output

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        self.regularization_norm = 0
        return error_tensor


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for i in range(iterations):
            Loss = self.forward()
            self.loss.append(Loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        output = input_tensor
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = self.phase
            output = layer.forward(output)
            # layer.testing_phase = False
        return output

    @property
    def phase(self):
        return self.testing_phase

    @phase.setter
    def phase(self, value):
        self.testing_phase = value

