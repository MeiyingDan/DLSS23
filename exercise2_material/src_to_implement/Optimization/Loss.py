import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        # Compute the loss value according to the CrossEntropy Loss formula
        # batch_size = prediction_tensor.shape[0]
        self.prediction_tensor = prediction_tensor
        self.epsilon = np.finfo(float).eps
        loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon))
        return loss

    def backward(self, label_tensor):
        # Return the error tensor for the previous layer
        error_tensor = -label_tensor / (self.prediction_tensor + self.epsilon)
        return error_tensor
