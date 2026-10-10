# Backpropagation & Gradient Calculus

> **Stage 3 — Deep Learning Foundations: Matrix Gradients, Reverse-Mode Autodiff & Gradient Checking**

---

## 1. Intuition

Training a neural network requires adjusting millions of parameters $\theta = \{W_1, b_1, W_2, b_2, \dots\}$ such that a scalar loss $\mathcal{L}$ decreases:
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

**Backpropagation** is an efficient implementation of **Reverse-Mode Automatic Differentiation**. It applies the multivariable **Chain Rule of Calculus** recursively backward from the scalar loss to the input features.

### Forward Pass vs Backward Pass
- **Forward Pass**: Data flows forward through the computational graph, computing activations $Z^{[1]} \to A^{[1]} \to Z^{[2]} \to \hat{Y}$ and caching intermediate values.
- **Backward Pass**: Error signals flow backward, computing parameter gradients $\frac{\partial \mathcal{L}}{\partial W}$ and input gradients $\frac{\partial \mathcal{L}}{\partial X}$ in a single pass of time complexity proportional to the forward pass!

---

## 2. Mathematical Derivation of Matrix Gradients

Consider a batch of $N$ samples with design matrix $X \in \mathbb{R}^{N \times D_{in}}$ passing through a Linear layer:
$$Z = X W + b$$
where $W \in \mathbb{R}^{D_{in} \times D_{out}}$ and $b \in \mathbb{R}^{D_{out}}$.

Suppose the downstream layers have already computed the upstream gradient tensor:
$$\delta = \frac{\partial \mathcal{L}}{\partial Z} \in \mathbb{R}^{N \times D_{out}}$$

### 2.1 Gradient with respect to Weights $W$
By the scalar chain rule for element $W_{j, k}$:
$$\frac{\partial \mathcal{L}}{\partial W_{j, k}} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial Z_{i, k}} \frac{\partial Z_{i, k}}{\partial W_{j, k}} = \sum_{i=1}^N \delta_{i, k} X_{i, j}$$

In matrix notation:
$$\frac{\partial \mathcal{L}}{\partial W} = X^T \delta = X^T \left( \frac{\partial \mathcal{L}}{\partial Z} \right)$$
*Dimension check*: $(D_{in} \times N) \times (N \times D_{out}) = (D_{in} \times D_{out})$ — matches $W$ exactly!

### 2.2 Gradient with respect to Bias $b$
$$\frac{\partial \mathcal{L}}{\partial b_k} = \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial Z_{i, k}} \frac{\partial Z_{i, k}}{\partial b_k} = \sum_{i=1}^N \delta_{i, k} (1)$$

In vector notation:
$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^N \delta_{i, :} = \mathbf{1}^T \delta$$
*Dimension check*: $(1 \times N) \times (N \times D_{out}) = (1 \times D_{out})$ — matches $b$!

### 2.3 Gradient with respect to Input $X$ (Propagated Backwards)
$$\frac{\partial \mathcal{L}}{\partial X_{i, j}} = \sum_{k=1}^{D_{out}} \frac{\partial \mathcal{L}}{\partial Z_{i, k}} \frac{\partial Z_{i, k}}{\partial X_{i, j}} = \sum_{k=1}^{D_{out}} \delta_{i, k} W_{j, k}$$

In matrix notation:
$$\frac{\partial \mathcal{L}}{\partial X} = \delta W^T = \left( \frac{\partial \mathcal{L}}{\partial Z} \right) W^T$$
*Dimension check*: $(N \times D_{out}) \times (D_{out} \times D_{in}) = (N \times D_{in})$ — matches $X$!

---

## 3. Activation Gradients

### 3.1 ReLU
$$A = \max(0, Z)$$
$$\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial A} \odot \mathbb{I}(Z > 0)$$

### 3.2 Fused Softmax Cross-Entropy
Given logits $Z \in \mathbb{R}^{N \times C}$ and one-hot ground-truth $Y$:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{i, c} \ln(P_{i, c})$$
$$\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{N} (P - Y)$$

---

## 4. Numerical Gradient Checking

To verify that analytical gradients $\nabla_{\text{analytical}}$ are 100% bug-free, we compute the two-sided central difference approximation:

$$g_{\text{numerical}}(\theta_i) \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}$$
with $\epsilon = 10^{-5}$.

We then evaluate the normalized relative error:
$$\text{Relative Error} = \frac{\|g_{\text{analytical}} - g_{\text{numerical}}\|_2}{\|g_{\text{analytical}}\|_2 + \|g_{\text{numerical}}\|_2}$$

- $\text{Relative Error} < 10^{-5}$: **Gradient calculation is correct.**
- $\text{Relative Error} > 10^{-3}$: **Bug in analytical backpropagation!**

---

## 5. Summary of Backpropagation Algorithm

```
Algorithm: Backpropagation for 2-Layer MLP
------------------------------------------
1. Forward:
     Z1 = X @ W1 + b1
     A1 = relu(Z1)
     Z2 = A1 @ W2 + b2
     P  = softmax(Z2)
     L  = cross_entropy(P, Y)

2. Backward:
     dZ2 = (P - Y_onehot) / N
     dW2 = A1.T @ dZ2
     db2 = sum(dZ2, axis=0)

     dA1 = dZ2 @ W2.T
     dZ1 = dA1 * (Z1 > 0)
     dW1 = X.T @ dZ1
     db1 = sum(dZ1, axis=0)

3. Update:
     W1 -= lr * dW1
     b1 -= lr * db1
     W2 -= lr * dW2
     b2 -= lr * db2
```
