import numpy as np
from Framework.Linkdlist_based.Layers.Dropout import Dropout_layer
from Framework.Linkdlist_based.Layers.Activation_fn import *
from Framework.Linkdlist_based.Layers.Normalization import BatchNorm_layer
from Framework.Linkdlist_based.Layers import Linear_layer


class Layer:

    _id_counter = 0    # to ensure unique layer ids

    def __init__(self, layer_type: str = "Layer"):
        self.next = None
        self.prev = None
        self.layer_type = layer_type
        self.has_learnable_param = True
        # rng = np.random.default_rng()
        Layer._id_counter += 1
        self.layer_id = f"{layer_type}_{Layer._id_counter}"

    def __call__(self, next_layer):
        self.next = next_layer
        next_layer.prev = self


class Hidden_layer(Layer):
    def __init__(self, d_in: int, d_out: int,  dropout_threshold: int, activation_fn: str = "LeakyReLU", is_training: bool = True):
        super().__init__("Hidden_layer")

        self.has_learnable_param = True

        self.linear = Linear_layer(d_in, d_out)

        activation_map = {
            "relu" : ReLU(),
            "sigmoid" : Sigmoid(),
            "tanh" : Tanh(),
            "leakyrelu" : LeakyReLU(),
            "softmax" : Softmax(),
            "elu" : ELU(),
            "gelu" : GELU(),
            "swish" : Swish(),
            "silu" : SILU()
        }
        
        self.activation = activation_map[activation_fn.lower()]
        
        self.BatchNorm = BatchNorm_layer(d_out, is_training)
        
        self.Dropout = Dropout_layer(dropout_threshold, is_training)

        # Arranging the layers
        self.linear(self.activation(self.BatchNorm(self.Dropout)))


    @property
    def sub_layers(self):
        
        all_sub_layers = [self.linear, self.activation, self.BatchNorm, self.Dropout]
        
        return {
            sub_layer.layer_id : sub_layer 
            for sub_layer in all_sub_layers 
            if sub_layer is None
        } 


    @property
    def parameters(self):
        hidden_layer_params = {}
        for layer_id in self.sub_layers:
            layer = self.sub_layers[layer_id]
            if (hasattr(layer, "parameters") and layer.has_learnable_param):
                for param_name, param in layer.parameters.items():
                    hidden_layer_params[param_name] = param 
        return hidden_layer_params


    @property
    def gradients(self):
        hidden_layer_grads = {}
        for layer_id in self.sub_layers:
            layer = self.sub_layers[layer_id]
            if (hasattr(layer, "gradients") and self.has_learnable_param):
                for grad_name, grad in layer.gradients.items():
                    hidden_layer_grads[grad_name] = grad
        return hidden_layer_grads


    def forward(self, X_in: np.ndarray, is_training: bool = True):
        curr_layer = self.linear
        x_in = X_in

        while(curr_layer):
            if(curr_layer.layer_type == "BatchNorm"):
                out = curr_layer.forward(x_in, is_training)
            else:
                out = curr_layer.forward(x_in)
            
            curr_layer = curr_layer.next
            x_in = out

    def backward(self, grad_in: np.ndarray):
        curr_layer = self.Dropout
        grad_in = grad_in

        while(curr_layer):
            grad_out = curr_layer.backward(grad_in)
            
            curr_layer = curr_layer.prev
            grad_in = grad_out



class Model:

    _id_counter = 0

    def __init__(self, first_layer: Layer):
        self.head = first_layer
        self.tail = first_layer
        self.model_type = "FeedForward-NN"
        rng = np.random.default_rng()
        self.model_id = f"{rng.integers(0, 1e6)}{Model._id_counter}"
        self.layer_sequence = {1 : first_layer}

    # To add individual layers sequnetially
    def __call__(self, next_layer: Layer):
        
        self.tail.next = next_layer
        next_layer.prev = self.tail
        self.tail = next_layer

        self.layer_sequence[max(self.layer_sequence) + 1] = next_layer
        

    @ property
    def parameters(self):
        for sequence, layer in self.layer_sequence.items():
            if (hasattr(layer, "parameters") and layer.has_learnable_param):
                return {
                    f"{layer.layer_id}_{param_name}" : param 
                    
                    for param_name, param in layer.parameters.items()
                }


    @property
    def gradients(self):
        for sequence, layer in self.layer_sequence.items():
            if (hasattr(layer, "gradients") and layer.has_learnable_param):
                return {
                    f"{layer.layer_id}_{grad_name}" : grad 
                    
                    for grad_name, grad in layer.gradients.items()
                }


    # To add n number of hidden layers sequentially
    def add_HiddenLayers(self, n: int, d_in: int, d_out: int, dropout_threshold: float, activation_fn: str):
        for i in range(n):
            if (n == 1):
                self.__call__(Hidden_layer(d_in, d_out, dropout_threshold, activation_fn))
            else:
                self.__call__(Hidden_layer(d_out, d_out, dropout_threshold, activation_fn))


    def forward(self, X_in: np.ndarray):
        cur_layer = self.head
        x_in = X_in
        
        while (cur_layer):
            out = cur_layer.forward(x_in)

            cur_layer = cur_layer.next

            x_in = out

        return out


    def backward(self, y: np.ndarray):
        cur_layer = self.tail
        grad_in = y

        while (cur_layer):
            grad_out = cur_layer.backward(grad_in)

            cur_layer = cur_layer.prev

            grad_in = grad_out

