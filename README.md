# Neural-Net-from-scratch (Custom Modular Neural Network Engine)
My own Perceptron Network - My attempt at custom implementation of state-of-art deep learning methods and architectures mimicing the Pytorch frameworks.


A lightweight, object-oriented Deep Learning framework built entirely from scratch using **Python and NumPy**. This framework implements a dynamic doubly-linked list graph execution architecture with professional, decoupled parameter optimization pipelines that mimic modern production deep learning environments like **PyTorch**.

---

## 🚀 Key Architectural Features

| Component | Engineering Implementation Strategy |
| :--- | :--- |
| **Graph Architecture** | **Doubly-Linked List:** Every layer acts as an isolated execution block tracking `.next` and `.prev` pointers to automatically assemble computational pipelines. |
| **Encapsulation** | **Scoped Flat Dictionaries:** Unique, collision-proof string keys (e.g., `Linear_1_W`, `BatchNorm_2_gamma`) manage parameter and gradient vectors dynamically without hardcoded class type checking. |
| **Memory Blueprint** | **Pass-by-Assignment Correction:** Updates target root container dictionary spaces (`parameters[name] = ...`) to enforce true in-place parameter optimizations across compound sub-layers. |
| **Optimizers** | **Polymorphic Step Routines:** Fully isolated update calculus loops supporting Vanilla SGD, Nesterov Accelerated Gradient (NAG), Adam, AdamW, Nadam, and RMSprop. |

---

## 🛠️ Layer Sequence & Topology

The core structural pipeline for processing inputs inside a hidden composite block is wired as follows:

```
Input Vector ➔ [ Linear Layer ] ➔ [ Activation Layer ] ➔ [ BatchNorm Layer ] ➔ [ Dropout Layer ] ➔ [ Softmax + CE Loss ]
```

### 1. Unified Layer & Parameter Hierarchy
Leaves (like `Linear_layer` and `BatchNorm_layer`) expose local names (`W`, `b`, `gamma`, `beta`). Parent wrappers (like `Hidden_layer` and `Model`) automatically scope these keys with unique IDs up the execution tree to ensure perfect dictionary symmetry:

```python
# Parameters dictionary and Gradients dictionary maintain absolute key symmetry:
print(model.parameters.keys())
# Outputs: ['Linear_1_W', 'Linear_1_b', 'Linear_3_W', 'Linear_3_b', 'BatchNorm_4_gamma', 'BatchNorm_4_beta']
```

---

## 💻 Code Examples

### 1. Defining a Model Pipeline
```python
import numpy as np

# Initialize individual standalone layers or isolated combo layers
layer1 = Hidden_layer(d_in=784, d_out=256, dropout_threshold=0.2, activation_fn="LeakyReLU")
layer2 = Hidden_layer(d_in=256, d_out=10, dropout_threshold=0.0, activation_fn="Softmax")

# Instantiate the FeedForward network graph via standard execution hooks
model = Model(first_layer=layer1)
model(layer2)  # Dynamically hooks layer1.next = layer2 and layer2.prev = layer1
```

### 2. The Universal Training Step Loop
```python
# Initialize a polymorphic optimizer
optimizer = AdamOptimizer(lr=0.001)
optimizer(model)  # Automatically binds the network and hooks memory buffers (v, m)

# Forward pass
predictions = model.forward(X_batch, is_training=True)
loss, loss_grad = criterion(predictions, Y_batch)

# Backward pass
model.backward(loss_grad)

# Optimize parameters uniformly across the graph
optimizer.step()
```

---

## 📉 Activation Function Blueprint Matrix

| Activation Function | Mathematical Formula | Optimal Placement / Use Cases |
| :--- | :--- | :--- |
| **ReLU** | $f(x) = \max(0, x)$ | Standard default for hidden layers in baseline MLPs. |
| **LeakyReLU** | $f(x) = \max(0.01x, x)$ | Prevents the "Dying ReLU" bottleneck; widely used in GANs. |
| **GELU** | $f(x) = x \cdot \Phi(x)$ | State-of-the-art default for Modern LLMs and Transformers (GPT/BERT). |
| **Swish / SiLU** | $f(x) = x \cdot \sigma(x)$ | Smooth, non-monotonic curve; popular in advanced CNNs (YOLO). |
| **Softmax** | $f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$ | Applied exclusively at the final layer for Multi-Class Output mapping. |

---

## ⚡ Mathematical Derivative Cheat Sheet

### Softmax Fused Cross Entropy Loss (Backward Pass)
$$\frac{\partial L}{\partial X} = \frac{1}{N} (P - T)$$

### Batch Normalization (Backward Gradient Vectorization)
$$\frac{\partial L}{\partial X_i} = \frac{\gamma}{N\sqrt{\sigma^2 + \epsilon}} \bigg[ N\frac{\partial L}{\partial Y_i} - d\beta - \hat{X}_i \odot d\gamma \bigg]$$

### Nesterov Accelerated Gradient (NAG Change of Variables Formulation)
$$v_{t+1} = \gamma v_t + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - (\gamma v_{t+1} + \eta \nabla L(\theta_t))$$
