class Sgd:
    def __init__(self,learning_rate:float):
        self.learning_rate = learning_rate 

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''Function returns the updated weights according to the basic gradient descent update scheme.
            required/usage: function(arg, arg) -> weight tensor'''
        updated_weights = weight_tensor + (-self.learning_rate*gradient_tensor)
        return updated_weights

