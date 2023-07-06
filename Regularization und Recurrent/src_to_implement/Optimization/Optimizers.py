import numpy as np

class Optimizer:
    def __init__(self):
        # self.learning_rate = learning_rate
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * regularizer_gradient
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * regularizer_gradient
        if self.v is None:
            self.v = np.zeros_like(gradient_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.v
        return updated_weights


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * regularizer_gradient
        self.k += 1

        if self.v is None:
            self.v = np.zeros_like(gradient_tensor)
        if self.r is None:
            self.r = np.zeros_like(gradient_tensor ** 2)
        self.v = self.mu * self.v + ((1-self.mu) * gradient_tensor)
        self.r = self.rho * self.r + (1-self.rho) * (gradient_tensor ** 2)
        v_hat = self.v/(1-(self.mu ** self.k))
        r_hat = self.r/(1-(self.rho ** self.k))
        updated_weights = weight_tensor - self.learning_rate * v_hat/(np.sqrt(r_hat) + np.finfo(float).eps)
        return updated_weights

