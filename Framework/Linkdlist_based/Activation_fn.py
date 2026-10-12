import numpy as np
from Framework.Linkdlist_based.Model import Layer

class Activation_fn(Layer):
    def __init__(self, activation_type: str):
        super().__init__("Activation")
        self.activation_type = activation_type
        self.x_in = None
        self.grad = None
        

class ReLU(Activation_fn):
    def __init__(self):
        super().__init__("ReLU")
        
    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return np.maximum(0, X_in)

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad = np.where(self.x_in > 0, 1, 0) * grad_in

    
class Sigmoid(Activation_fn):
    def __init__(self):
        super().__init__("Sigmoid")
        
    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return 1/( 1 + np.exp(-X_in))

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        sigma = 1 /(1 + np.exp(-self.x_in))
        self.grad = sigma(1 - sigma) * grad_in


class Tanh(Activation_fn):
    def __init__(self):
        super().__init__("Tanh")
        
    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return np.tanh(X_in)

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad = (1 - np.square(np.tanh(self.x_in))) * grad_in


class Softmax(Activation_fn):
    def __init__(self):
        super().__init__("Softmax")

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        exp = np.exp(X_in - np.max(X_in, axis=-1, keepdims=True))
        return exp /np.sum(exp, axis=-1, keepdims=True) 

     
class LeakyReLU(Activation_fn):
    def __init__(self, alpha: float = 0.01):
        super().__init__("LeakyReLU")
        self.alpha = alpha
        
    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return np.where(X_in > 0, X_in, self.alpha * X_in)

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad = np.where(self.x_in > 0, 1, self.alpha) * grad_in
    
class ELU(Activation_fn):
    def __init__(self, alpha: float = 1.0):
        super().__init__("ELU")
        self.alpha = alpha

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return np.where(X_in > 0, X_in, self.alpha * (np.exp(X_in) - 1)) 

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        self.grad = np.where(self.x_in > 0, 1, self.alpha * np.exp(self.x_in)) * grad_in
    
class Swish(Activation_fn):
    def __init__(self, beta: float = 1.0):
        super().__init__("Swish")
        self.beta = beta

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return X_in * (1 / (1 + np.exp(-self.beta * X_in)))

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        swish = self.forward(self.x_in)
        sigma = 1 / (1 + np.exp(-self.X_in))
        self.grad = self.beta * (swish + sigma(1 - swish)) * grad_in


class SILU(Activation_fn):
    def __init__(self):
        super().__init__("SILU")
        self.beta = 1.0

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return X_in * (1 / (1 + np.exp(-self.beta * X_in)))

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        swish = self.forward(self.x_in)
        sigma = 1 / (1 + np.exp(-self.X_in))
        self.grad = self.beta * (swish + sigma(1 - swish)) *grad_in


class GELU(Activation_fn):
    def __init__(self):
        super().__init__("GELU")
        self.k = 0.044715

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        self.x_in = X_in
        return 0.5 * X_in * (1 + np.tanh(np.sqrt(2 / np.pi) * (X_in + self.k * np.power(X_in, 3))))

    def backward(self, grad_in: np.ndarray) -> np.ndarray:
        phi_x = np.tanh(np.sqrt(2 / np.pi) * (self.X_in + self.k * np.power(self.X_in, 3)))
        g_x = np.sqrt(2 / np.pi) * (1 + 3 * self.k * np.square(self.x_in))

        self.grad = 0.5 * (1 + phi_x + self.x_in * (1 - np.square(phi_x)) * g_x)

    
