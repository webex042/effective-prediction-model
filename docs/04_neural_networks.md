# Neural Networks

> **Stage 3 — Deep Learning Foundations: Architecture, Non-Linearity & Forward Propagation**

---

## 1. Intuition

A neural network is a parameterized function approximator $f_\theta(x)$ composed of layered affine transformations interleaved with non-linear activation functions.

### Biological Analogy vs Mathematical Reality
- **Biological inspiration**: Neurons receive dendrite inputs, weigh them across synapses, sum them at the soma, and fire an electrical spike down the axon if a threshold is exceeded.
- **Mathematical reality**: Each layer is high-dimensional matrix multiplication followed by an element-wise non-linear scalar function:
  $$z = Wx + b$$
  $$a = \sigma(z)$$

### Why Deep Networks? (The Collapse of Linear Stacks)
Suppose we construct a 3-layer network without activation functions:
$$z_1 = W_1 x + b_1$$
$$z_2 = W_2 z_1 + b_2$$
$$z_3 = W_3 z_2 + b_3$$

Substituting directly:
$$z_3 = W_3 (W_2 (W_1 x + b_1) + b_2) + b_3 = (W_3 W_2 W_1) x + (W_3 W_2 b_1 + W_3 b_2 + b_3)$$

Letting $W_{equiv} = W_3 W_2 W_1$ and $b_{equiv} = W_3 W_2 b_1 + W_3 b_2 + b_3$:
$$z_3 = W_{equiv} x + b_{equiv}$$

A 100-layer purely linear network is mathematically equivalent to a single linear layer! Non-linear activations are the **sole mechanism** that gives neural networks the power to bend, fold, and carve non-linear decision boundaries.

---

## 2. Mathematical Formulation

### 2.1 The Two-Layer Perceptron (MLP)
Given an input vector $x \in \mathbb{R}^{D_{in}}$:

1. **Hidden Linear Pre-Activation**:
   $$z^{[1]} = W^{[1]} x + b^{[1]}$$
   where $W^{[1]} \in \mathbb{R}^{H \times D_{in}}$ and $b^{[1]} \in \mathbb{R}^H$.

2. **Hidden Non-Linear Activation (ReLU)**:
   $$a^{[1]} = \text{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$

3. **Output Linear Pre-Activation**:
   $$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$$
   where $W^{[2]} \in \mathbb{R}^{C \times H}$ and $b^{[2]} \in \mathbb{R}^C$.

4. **Output Probabilities (Softmax)**:
   $$\hat{y}_c = P(y = c \mid x) = \frac{e^{z_c^{[2]}}}{\sum_{k=1}^C e^{z_k^{[2]}}}$$

### 2.2 Activation Functions
| Activation | Equation | Derivative | Key Property |
|------------|----------|------------|--------------|
| **ReLU** | $\max(0, z)$ | $\mathbb{I}(z > 0)$ | Constant gradient 1 for $z>0$; fast, avoids saturation |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | Squashes to $(0, 1)$; vanishes for $\|z\| \gg 0$ |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | Zero-centered $(-1, 1)$; vanishes at tails |

### 2.3 Weight Initialization (He / Kaiming)
If weights are initialized too large, activations explode. If too small, signals vanish to zero.
For ReLU activations, Kaiming Normal draws weights from:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{D_{in}}}\right)$$
This preserves activation variance across forward layers: $\text{Var}(z^{[l]}) = \text{Var}(z^{[l-1]})$.

---

## 3. Assumptions & Universal Approximation

1. **Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991)**:
   A standard feedforward network with a single hidden layer and non-linear activation can approximate any continuous function on a compact subset of $\mathbb{R}^D$ to arbitrary precision $\epsilon > 0$, provided sufficient hidden units $H$.
2. **Smooth Loss Landscape**: Gradient-based optimization assumes the loss function surface is locally differentiable.

---

## 4. Computational Complexity

For batch size $N$, input dimension $D$, hidden units $H$, and classes $C$:

| Layer | Forward FLOPs | Backward FLOPs | Parameter Count |
|---|---|---|---|
| **Linear 1** ($D \to H$) | $2 \cdot N \cdot D \cdot H$ | $4 \cdot N \cdot D \cdot H$ | $D \cdot H + H$ |
| **ReLU** | $N \cdot H$ | $N \cdot H$ | $0$ |
| **Linear 2** ($H \to C$) | $2 \cdot N \cdot H \cdot C$ | $4 \cdot N \cdot H \cdot C$ | $H \cdot C + C$ |
| **Softmax + Loss** | $3 \cdot N \cdot C$ | $N \cdot C$ | $0$ |

Total Time per step: $\mathcal{O}(N(DH + HC))$.

---

## 5. From-Scratch Implementation (NumPy)

```python
import numpy as np

class TwoLayerMLP:
    def __init__(self, in_features, hidden_dim, num_classes):
        # Kaiming initialization
        self.W1 = np.random.randn(in_features, hidden_dim) * np.sqrt(2.0 / in_features)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(num_classes)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        # Numerically stable softmax
        shifted = self.z2 - np.max(self.z2, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs
```

---

## 6. Limitations

- **Interpretability**: Unlike Decision Trees or Linear Regression with feature weights, individual hidden activations lack direct semantic interpretation ("black-box" nature).
- **Sample Efficiency**: Deep neural networks typically require more data than regularized classical models to avoid overfitting.
- **Hyperparameter Sensitivity**: Learning rate, batch size, and weight initialization strongly influence convergence.
