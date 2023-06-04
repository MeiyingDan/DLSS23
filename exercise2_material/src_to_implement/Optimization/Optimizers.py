import numpy as np

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
        m_tensor = self.mu * m_tensor + (1 - self.mu) * gradient_tensor
        v_tensor = self.rho * v_tensor + (1 - self.rho) * np.square(gradient_tensor)
        m_hat = m_tensor / (1 - self.mu ** t)
        v_hat = v_tensor / (1 - self.rho ** t)
        weight_tensor = self.learning_rate * m_hat / (np.sqrt(v_hat) + np.epsilon)
        return weight_tensor, m_tensor, v_tensor