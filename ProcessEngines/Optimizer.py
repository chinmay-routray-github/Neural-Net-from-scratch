
from Framework.Linkdlist_based.Model import *

class Optimizer:
    def __init__(self, lr: float = 0.01, optimizer: str = "gradient descent"):
        self.lr = lr
        self.algorithm = optimizer
        self.t = 0  # iteration counter
        self.model = None
        

    def __call__(self, model: Model):
        self.model = model


    def update_param(self, *args, ** kwargs):
        NotImplementedError("Please choose a valid Optimizer. Optimizer is just generic class " \
        "with update_para method not implemented.")

    
    def step(self):
        
        parameters = self.model.parameters

        gradients = self.model.gradients

        # 1. Condition for first & second moment-based optimizers (e.g., Adam, Nadam)
        if (hasattr(self, "v") and hasattr(self, "m")):
            for param_name in self.model.parameters:
                parameters[param_name], self.v[param_name] , self.m[param_name] = self.update_param(self.v[param_name], 
                                                                                                    self.m[param_name], 
                                                                                                    parameters[param_name], 
                                                                                                    gradients[param_name])
        
        # 2. Condition for first moment-based optimizers (e.g., RMSprop, Momentum, NAG)
        elif (hasattr(self, "v")):
            for param_name in self.model.parameters:
                parameters[param_name], self.v[param_name] = self.update_param(self.v[param_name], 
                                                                               parameters[param_name], 
                                                                               gradients[param_name])

        else:
            for param_name in self.model.parameters:
                parameters[param_name] = self.update_param(parameters[param_name], gradients[param_name])

        # Increment the iteration counter if all updates are successful,
        # else rollback
        self.t += 1


class Gradient_Descent(Optimizer):
    def __init__(self, lr: float):
        super().__init__(lr)


    def update_param(self, w_t: np.ndarray, dw: np.ndarray) -> np.ndarray:
        return w_t - self.lr * dw

# Nesterov's Accelerated gradient descent
class NAG(Optimizer):
    def __init__(self, lr: float = 0.01, momentum_coef: float = 0.9):
        super().__init__(lr, "NAG")
        self.eta = momentum_coef
        self.v = {}


    def __call__(self, model: Model):
        # self.model = model

        self.v = {
                
            param_name : np.zeros_like(param) 
                
            for param_name, param in model.parameters.items()
        }
            

    
	# \(v_{t+1} = \gamma v_t + \eta \nabla_\theta L(\theta_t)\)
	# \(\theta_{t+1} = \theta_t - (\gamma v_{t+1} + \eta \nabla_\theta L(\theta_t))\)

    def update_param(self, v_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray) -> np.ndarray:
        
        v_t_next = self.eta * v_t + self.lr * dw

        return w_t - (self.eta * v_t_next + self.lr * dw), v_t_next

# Adaptive gradient
# It accumulated the squared gradients and uses this running sum to modify the update step
class AdaGrad(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__(lr, "AdaGrad")
        self.eps = 1e-8
        self.v = {}


    def __call__(self, model: Model):

        self.v = {
                
            param_name : np.zeros_like(param) 
                
            for param_name, param in model.parameters.items()
        }

        
    def update_param(self, v_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray) -> np.ndarray:
        # square of gradients accumulation
        v_t_next = v_t + np.square(dw)

        # update learnable parameters
        return w_t - (self.lr * dw) / (np.sqrt(v_t_next) + self.eps), v_t_next

# RMSprop
class RMSprop(Optimizer):
    def __init__(self, lr = 0.01, smoothing_coef: float = 0.9):
        super().__init__(lr, "RMSprop")
        self.beta = smoothing_coef
        self.eps = 1e-8
        self.v = {}


    def __call__(self, model: Model):

        self.v = {
                
            param_name : np.zeros_like(param) 
                
            for param_name, param in model.parameters.items()
        }


    def update_param(self, v_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray):
        # Update uncentered variance tracker by moving averages of square of gradients
        v_t_next = self.beta * v_t + (1 - self.beta) * np.square(dw)

        # update learnable parameter
        return w_t - (self.lr * dw) / (np.sqrt(v_t_next) + self.eps), v_t_next

# Highly popular in LSTMs, RNNs, and Reinforcement Learning
class RMSprop_Momentum(Optimizer):
    def __init__(self, lr = 0.01, smoothing_coef: float = 0.9, momentum_coef: float = 0.9):
        super().__init__(lr, "RMSprop + Momentum")
        self.beta = smoothing_coef
        self.eta = momentum_coef
        self.eps = 1e-8
        self.v = {}
        self.m = {}


    def __call__(self, model: Model):

        self.v = {
        
            param_name : np.zeros_like(param) 
        
            for param_name, param in model.parameters.items()
        }
        
        self.m = {
        
            param_name : np.zeros_like(param)
        
            for param_name, param in model.parameters.items()
        }  

    
    def update_param(self, v_t: np.ndarray, m_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray):
        # Update uncentered variance tracker by moving averages of square of gradients
        v_t_next = self.beta * v_t + (1 - self.beta) * np.square(dw)

        # update momentum 
        m_t_next = self.eta * m_t + (self.lr * dw) / (np.sqrt(v_t_next) + self.eps)

        # update learnable parameter
        return w_t - m_t_next, v_t_next, m_t_next

# Adaptive Moment Estimation 
# Transformers, GANs, and deep CNNs. Excellent all-rounder
class Adam(Optimizer):
    def __init__(self, lr: float = 0.01, first_moment_coef: float = 0.9,  second_moment_coef: float = 0.99):
        super().__init__(lr, "Adam")
        self.beta_1 = first_moment_coef
        self.beta_2 = second_moment_coef
        self.eps = 1e-8
        self.v = {}
        self.m = {}


    def __call__(self, model: Model):
    
        self.v = {
            
            param_name : np.zeros_like(param) 
            
            for param_name, param in model.parameters.items()
        }
            
        self.m = {
            
            param_name : np.zeros_like(param)
            
            for param_name, param in model.parameters.items()
        } 


    def update(self, v_t: np.ndarray, m_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray):
        # 1. Standard Adam moment accumulation updates
        m_t_next = self.beta_1 * m_t + (1 - self.self.beta_1) * dw

        v_t_next = self.beta_2 * v_t + (1 - self.beta_2) * np.square(dw)

        # 2. Bias correction step
        m_t_next_hat = m_t_next / (1 - self.beta_1 ** self.t)

        v_t_next_hat = v_t_next / (1 - self.beta_2 ** self.t)

        # update learnable parameter
        return w_t - (self.lr * m_t_next_hat) / (np.sqrt(v_t_next_hat) + self.eps), v_t_next_hat, m_t_next_hat


class AdamW(Optimizer):
    def __init__(self, lr: float = 0.01, first_momentum_coef: float = 0.9, second_momentum_coef: float = 0.999, weight_decay_rate: float = 0.01):
        super().__init__(lr, "AdamW")
        self.beta_1 = first_momentum_coef
        self.beta_2 =  second_momentum_coef
        self.eta = weight_decay_rate
        self.eps = 1e-8
        self.v = {}
        self.m = {}

    def __call__(self, model: Model):
    
        self.v = {
            
            param_name : np.zeros_like(param) 

            for param_name, param in model.parameters.items()
        }
            
        self.m = {
            
            param_name : np.zeros_like(param)
            
            for param_name, param in model.parameters.items()
        } 

    def update_param(self, v_t: np.ndarray, m_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray):
        # 1. Standard Adam moment accumulation updates
        m_t_next = self.beta_1 * m_t + (1 - self.beta_1) * dw

        v_t_next = self.beta_2 * v_t + (1 - self.beta_2) * np.square(dw)

        # 2. Bias correction step
        m_t_next_hat = m_t_next / (1 - self.beta_1 ** self.t)

        v_t_next_hat = v_t_next / (1 - self.beta_2 ** self.t)

        # 3. update learnable parameters
        return w_t - self.lr * self.eta * w_t - (self.lr * m_t_next_hat) / (np.sqrt(v_t_next_hat) + self.eps), v_t_next_hat, m_t_next_hat


# Nesterov-accelerated Adam
# Tasks where you want Adam but need faster training convergence
class NAdam(Optimizer):
    def __init__(self, lr: float = 0.01, first_moment_coef: float = 0.9, second_moment_coef: float = 0.99):
        super().__init__(lr, "NAdam")
        self.beta_1 = first_moment_coef
        self.beta_2 = second_moment_coef
        self.eps = 1e-8
        self.v = {}
        self.m = {}
        
        
    def __call__(self, model: Model):
            
        self.v = {
                    
            param_name : np.zeros_like(param) 
                    
            for param_name, param in model.parameters.items()
        }
                    
        self.m = {
                    
            param_name : np.zeros_like(param)
                    
            for param_name, param in model.parameters.items()
        }

    
    def update_param(self, v_t: np.ndarray, m_t: np.ndarray, w_t: np.ndarray, dw: np.ndarray):
        # 1. Standard Adam moment accumulation updates
        m_t_next = self.beta_1 * m_t + (1 - self.beta_1) * dw

        v_t_next = self.beta_2 * v_t + (1 - self.beta_2) * np.square(dw)

        # 2. Bias correction for the second moment (variance)
        v_t_next_hat = v_t_next / (1 - self.beta_2 ** self.t)

        # 3. Nesterov Look-Ahead correction step combined with first moment bias correction
        m_look_ahead = self.beta_1/(1  - self.beta_1 ** (self.t + 1)) * m_t_next

        dw_look_ahead = (1 - self.beta_1) / (1 - self.beta_1 ** self.t) * dw

        m_t_next_hat = m_look_ahead + dw_look_ahead

        # update learnable parameter
        return w_t - (self.lr * m_t_next_hat) / (np.sqrt(v_t_next_hat) + self.eps), v_t_next_hat, m_t_next_hat


