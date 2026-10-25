import numpy as np
from Framework.Linkdlist_based.Model import Layer

class Dropout_layer(Layer):
    def __init__(self, threshold: float = 0.5, is_training: bool = False):
        super().__init__("Dropout")
        self.is_training = is_training
        self.has_learnable_param = False
        self.p = threshold      # threshold - p
        self.M = None           # dropout_mask - M
        self.grad = None        # dL/dA = M/(1-p) * dL/dZ
        

        @property
        def parameters(self):
            return { 
                "M" : self.M 
            }

        
        @property
        def gradients(self):
            return { 
                "dx" : self.grad 
            }

        @property
        def hyper_parameters(self):
            return {
                "p" : self.p
            }
        

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        if self.is_training:
            rng = np.random.default_rng
            self.M = (np.random(X_in.shape) >= self.threshold)
            scale = 1 / (1 - self.p)
            return scale * X_in * self.M
        else:
            return X_in


    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad = (self.M / (1 - self.p)) * grad_in 
        return self.grad
                


