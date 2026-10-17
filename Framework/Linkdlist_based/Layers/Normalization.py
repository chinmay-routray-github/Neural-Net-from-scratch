import numpy as np
from Framework.Linkdlist_based.Model import Layer

class BatchNorm_layer(Layer):
    def __init__(self, out_dim: int, is_training: bool = False):
        super().__init__("BatchNorm")
        self.is_training = is_training
        self.var = 0
        self.running_mean = 0
        self.running_var = 0
        self.epsilon = 1e-6
        self.momentum = 0.9
        self.rng = np.random.default_rng()
        self.gamma = self.rng.standard_normal(out_dim)
        self.beta = self.rng.standard_normal(out_dim)
        self.x_hat = None
        self.grad_gamma = None
        self.grad_beta = None
        self.grad_x = None 
        self.params = [self.gamma, self.beta, self.grad_gamma, self.grad_beta, self.grad_x]

    def forward(self, X_in: np.ndarray):
        if self.is_training:
            self.X_hat = self.normalize(X_in)
        else:
            self.X_hat = (X_in - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.X_hat @ self.gamma + self.beta


    def normalize(self, X_in: np.ndarray):
       x_mean = np.mean(X_in, axis = 0, keepdims = True)
       self.var = np.var(X_in, axis = 0, keepdims = True)

       X_hat = (X_in - x_mean) / np.sqrt(var + self.epsilon)

       self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
       self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

       return X_hat


    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad_beta = np.mean(grad_in, axis=0, keepdims = True)
        self.grad_gamma = np.mean(grad_in * self.X_hat, axis=0, keepdims = True)

        B = self.X_hat.shape[0]
        grad_diff = B * grad_in - self.grad_beta - self.x_hat * self.grad_gamma
        scaling_factor = 1 /(B * np.sqrt(self.var + self.epsilon)) * self.gamma
        
        self.grad_x = scaling_factor * grad_diff
        
        return self.grad_x
