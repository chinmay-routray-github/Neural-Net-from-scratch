import numpy as np

class Layer:
    def __init__(self, layer_type: str = "Layer"):
        self.next = None
        self.prev = None
        self.layer_type = layer_type
        rng = np.random.default_rng()
        self.layer_id = rng.integers(0, 1e6)

    def __call__(self, input_layer):
        self.prev = input_layer
        input_layer.next = self


class Linear_layer(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__("Linear")
        self.input_dim = input_dim
        self.output_dim = output_dim
        rng = np.random.default_rng()
        self.W = rng.standard_normal(self.input_dim, self.output_dim)
        self.b = rng.standard_normal(self.output_dim)
        self.params = [self.W, self.b]
        

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        return X_in @ self.W + self.b
