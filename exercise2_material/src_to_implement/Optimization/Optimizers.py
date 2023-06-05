import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights

class SgdWithMomentum:
    def __init__(self,learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
            velocity_tensor = self.momentum_rate * velocity_tensor - self.learning_rate * gradient_tensor
            weight_tensor += velocity_tensor
            return weight_tensor, velocity_tensor

class Adam:
    def __init__(self,learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.mu * v + (1 - self.mu) * gradient_tensor
        r = self.rho * r + (1 - self.rho) * np.square(gradient_tensor)
        m_hat = v / (1 - self.mu)
        v_hat = r / (1 - self.rho)
        weight_tensor = self.learning_rate * m_hat / (np.sqrt(v_hat) + np.epsilon)
        return weight_tensor, v, r