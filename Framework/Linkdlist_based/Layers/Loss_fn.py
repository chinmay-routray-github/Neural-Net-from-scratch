import numpy as np
from Framework.Linkdlist_based.Model import Layer

class Loss_fn(Layer):
    def __init__(self, loss_type: str):
        super().__init__("Output_layer")
        self.loss_type = loss_type
        self.grad = None
        self.prediction = None


class MAE(Loss_fn):
    def __init__(self):
        super().__init__("MAE")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        self.prediction = y_hat
        return np.mean(np.abs(y - y_hat))

class MSE(Loss_fn):
    def __init__(self):
        super().__init__("MSE")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        self.prediction = y_hat
        return np.mean(np.square(y - y_hat))

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        B = 1 / y.shape[0]          #batch_size B
        self.grad = (2 / B) * (y - y_hat)
        return self.grad

# For one-hot encoded labels
# Valid for Multi-Label Classification, each class output is independent.
# The targets (y) are vectors like [1, 0, 1].
class BCE(Loss_fn):
    def __init__(self):
        super().__init__("BCE")

    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        eps = 1e-6
        self.prediction = y_hat
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        loss_per_sample = np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped), axis=-1)
        return - np.mean(loss_per_sample)

# Valid for Multi-Class Classification, a sample belongs to exactly one mutually exclusive class 
# The classes are dependent.
# The targets (y) are strictly one-hot encoded like [0, 1, 0].
# With advance indexing, y like [2, 0, 1]
class CrossEntropy(Loss_fn):
    def __init__(self):
        super().__init__("CrossEntropy")

    
    def forward(self, y: np.ndarray, y_hat: np.ndarray, advance_indexing: bool = True) -> float:
        eps = 1e-6
        self.prediction = y_hat
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        if(advance_indexing):
            #For Advance Indexing 
            # y - shape (b,)

            predicted_class = y_hat_clipped[np.arange(y.shape[0]), y]
            return - np.mean(np.log(predicted_class + eps))
        else:
            #For one-hot encoded labels
            # y - shape (b X d)

            loss_per_sample = np.mean(y * np.log(y_hat_clipped), axis=-1)
            return - np.mean(loss_per_sample)


class SoftmaxFusedCrossEntropy(Loss_fn):
    def __init__(self):
        super().__init__("Softmax + CrossEntropy")

    # Combinined simple approach
    # def forward(self, y: np.ndarray, logits: np.ndarray) -> float:
    #     logits_exp = np.exp(logits - np.max(logits, axis=-1, keepdims = True))
    #     y_hat = logits_exp / np.sum(logits_exp, axis=-1, keepdims = True)

    #     eps = 1e-6
    #     y_hat_clipped = np.clip(y_hat, eps, 1.0 - eps)
    #     loss_per_sample = np.sum(y * np.log(y_hat_clipped), axis=-1)
    #     return - np.mean(loss_per_sample)

    # Softmax fused with cross entropy
    # log(y_hat_i) = (z_i - z_i_max) - log(sigma(e^z_i))   Obtained by substitution for numerical stability
    # substituting y_hat in log(y_hat)
    def forward(self, y: np.ndarray, logits: np.ndarray) -> float:
        stable_logits = np.exp(logits - np.max(logits, axis=-1, keepdims = True))
        self.prediction = stable_logits
        log_y_hat = stable_logits - np.log(np.sum(np.exp(logits)), axis=-1, keepdims = True)
    
        loss_per_sample = np.sum(y * log_y_hat, axis=-1)
        return - np.mean(loss_per_sample)

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        B = 1 / y.shape[0]          #batch_size B
        self.grad = (1 / B) * (y_hat - y)
        return self.grad

# BCEWithLogitsLoss 
# Loss = max(x, 0) - xy + log(1 + e^-|x|)  Obtained by substitution for numerical stability
# substituting y_hat in log(y_hat) and log(1 - y_hat)
class SigmoidFusedBCE(Loss_fn):
    def __init__(self):
        super().__init__("Sigmoid + BCE")


    def forward(self, y: np.ndarray, logits: np.ndarray) -> float:
        self.prediction = logits
        loss_per_sample = np.max(logits, 0, axis=-1, keepdims = True) - logits * y + np.log(1 + np.exp(-np.abs(logits)))
        return np.mean(loss_per_sample)

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        B = 1 / y.shape[0]           #batch_size B
        self.grad = (1 / B) * (y_hat - y)
        return self.grad

