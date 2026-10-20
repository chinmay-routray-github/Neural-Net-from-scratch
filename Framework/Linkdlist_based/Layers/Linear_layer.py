import numpy as np
from Framework.Linkdlist_based.Model import Layer

class Linear_layer(Layer):
    def __init__(self, d_in: int, d_out: int):
        super().__init__("Linear")
        self.d_in = d_in
        self.d_out = d_out
        rng = np.random.default_rng()
        self.W = rng.standard_normal(self.d_in, self.d_out)
        self.b = rng.standard_normal(self.d_out)
        self.grad_w = None
        self.grad_b = None
        self.grad_x = None
        self.params = [self.W, self.b]
        

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return X_in @ self.W + self.b

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad_w = np.dot(self.x_in.T, grad_in) 
        self.grad_b = np.mean(grad_in, axis=0, keepdims = True)
        self.grad_x = np.dot(grad_in, self.W.T)

