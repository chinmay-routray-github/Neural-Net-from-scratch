import numpy as np
from Framework.Linkdlist_based.Model import Layer

class BatchNorm_layer(Layer):
    def __init__(self, out_dim: int, is_training: bool = False):
        super().__init__("BatchNorm")
        self.is_training = is_training
        self.has_learnable_param = True
        self.var = 0
        self.running_mean = 0
        self.running_var = 0
        self.eps = 1e-6
        self.momentum = 0.9
        self.rng = np.random.default_rng()
        self.gamma = self.rng.standard_normal(out_dim)
        self.beta = self.rng.standard_normal(out_dim)
        self.x_hat = None
        self.grad_gamma = None
        self.grad_beta = None
        self.grad_x = None 

        @property
        def parameters(self):
            return {
                "gamma" : self.gamma,
                "beta" : self.beta
            }


        @property
        def gradients(self):
            return {
                "gamma" : self.grad_gamma,
                "beta" : self.grad_beta
            }


        @property
        def hper_parameters(self):
            return {
                "running_mean" : self.running_mean,
                "running_var" : self.running_var,
                "momentum_coef" : self.momentum,
                "eps" : self.eps
            }

    def forward(self, X_in: np.ndarray):
        if self.is_training:
            self.X_hat = self.normalize(X_in)
        else:
            self.X_hat = (X_in - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.X_hat @ self.gamma + self.beta


    def normalize(self, X_in: np.ndarray):
       x_mean = np.mean(X_in, axis = 0, keepdims = True)
       self.var = np.var(X_in, axis = 0, keepdims = True)

       X_hat = (X_in - x_mean) / np.sqrt(self.var + self.eps)

       self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
       self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var

       return X_hat


    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad_beta = np.mean(grad_in, axis=0, keepdims = True)
        self.grad_gamma = np.mean(grad_in * self.X_hat, axis=0, keepdims = True)

        B = self.X_hat.shape[0]
        grad_diff = B * grad_in - self.grad_beta - self.x_hat * self.grad_gamma
        scaling_factor = 1 /(B * np.sqrt(self.var + self.eps)) * self.gamma
        
        self.grad_x = scaling_factor * grad_diff
        
        return self.grad_x
