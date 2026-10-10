# Linear Algebra for Machine Learning

> **Stage 1 — Mathematical Foundation**

## Overview

Linear algebra is the mathematical backbone of machine learning. Nearly every ML algorithm involves operations on vectors and matrices. **If you understand one thing about neural networks, understand this: they are sequences of matrix multiplications.**

---

## 1. Vectors

### Intuition
A vector is an ordered list of numbers. In ML, vectors represent:
- **Data points**: a sample with n features is a vector in Rⁿ
- **Weights**: each neuron stores a weight vector
- **Embeddings**: words, images, and concepts are represented as vectors
- **Gradients**: direction to update parameters

### Mathematical Notation
A vector **v** ∈ Rⁿ:

```
v = [v₁, v₂, ..., vₙ]
```

### Implementation
```python
import numpy as np

# A 3-dimensional vector
v = np.array([1.0, 2.0, 3.0])

# In ML: a data point with 3 features
# e.g., [temperature, pressure, vibration]
sensor_reading = np.array([87.5, 1.2, 0.05])
```

---

## 2. Dot Product

### Intuition
The dot product measures how much two vectors "agree" in direction:
- **Same direction** → large positive value
- **Perpendicular** → zero
- **Opposite direction** → large negative value

### Math
```
a · b = Σᵢ aᵢ × bᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

### Numerical Example
```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

### Implementation
```python
from src.math_foundations.linear_algebra import dot_product

result = dot_product([1, 2, 3], [4, 5, 6])  # = 32.0
```

### ML Application
| Where | How |
|---|---|
| Neuron computation | Each neuron computes `w · x + b` |
| Attention mechanism | Similarity = `query · key` |
| Cosine similarity | `(a · b) / (‖a‖ × ‖b‖)` |
| Loss computation | Often involves dot products |

---

## 3. Norms

### Intuition
Norms measure the "size" or "length" of a vector.

- **L1 norm** (Manhattan): Sum of absolute values. Like walking on a grid.
- **L2 norm** (Euclidean): Straight-line distance. The "natural" length.

### Math
```
L1: ‖v‖₁ = Σᵢ |vᵢ|
L2: ‖v‖₂ = √(Σᵢ vᵢ²)
```

### Numerical Example
```
v = [3, 4]
L1: |3| + |4| = 7
L2: √(3² + 4²) = √(9 + 16) = √25 = 5
```

### ML Application
| Norm | ML Use |
|---|---|
| L1 | Lasso regularization → encourages sparsity (some weights = 0) |
| L2 | Ridge regularization → prevents large weights (weight decay) |
| L2 | Gradient clipping → `if ‖∇‖ > threshold: ∇ = ∇ × threshold/‖∇‖` |

---

## 4. Matrix Multiplication

### Intuition
Matrix multiplication is THE core operation of neural networks. When you compute `z = Wx + b`, you're multiplying a weight matrix by an input vector.

Each element of the result is a dot product between a row of the first matrix and a column of the second.

### Math
If **A** is (m × n) and **B** is (n × p), then **AB** is (m × p):

```
(AB)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ
```

### Numerical Example
```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]

AB = [[1×5+2×7, 1×6+2×8],    = [[19, 22],
      [3×5+4×7, 3×6+4×8]]       [43, 50]]
```

### Implementation
```python
from src.math_foundations.linear_algebra import matrix_multiply

result = matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
# array([[19., 22.], [43., 50.]])
```

### Complexity
- **Time**: O(m × n × p) — this is why large models are expensive!
- **Space**: O(m × p) for the output

### ML Application
A neural network layer with 768 inputs and 3072 outputs requires:
- **2.36 million** multiplications per forward pass per sample
- This is why GPUs (massively parallel matrix multipliers) are essential

---

## 5. Transpose

### Math
Swap rows and columns: `(Aᵀ)ᵢⱼ = Aⱼᵢ`

### ML Application
- **Backpropagation**: gradients flow through Wᵀ (transposed weights)
- **Attention**: `QKᵀ` requires transposing the key matrix
- **Covariance**: `Cov = (1/n) × XᵀX`

---

## 6. Covariance Matrix

### Intuition
The covariance matrix tells you how features move together:
- **Positive covariance**: when feature A goes up, B tends to go up
- **Negative covariance**: when A goes up, B tends to go down
- **Zero covariance**: A and B are unrelated

### Math
```
C = (1/(n-1)) × X_centeredᵀ × X_centered
```

where `X_centered = X - mean(X)`.

The diagonal elements `Cᵢᵢ = Var(feature_i)`.

### Implementation
```python
from src.math_foundations.linear_algebra import covariance_matrix

X = [[1, 2], [3, 4], [5, 6]]
cov = covariance_matrix(X)
# Diagonal = variance, off-diagonal = covariance
```

---

## 7. Eigendecomposition

### Intuition
Eigendecomposition finds the "natural axes" of a matrix.

For a covariance matrix:
- **Eigenvectors** = directions of maximum variance (principal components)
- **Eigenvalues** = amount of variance in each direction

### Math
Find **v** and **λ** such that: `Av = λv`

- **v** is the eigenvector (direction that A only scales, doesn't rotate)
- **λ** is the eigenvalue (the scaling factor)

### ML Application
- **PCA**: Keep eigenvectors with largest eigenvalues → dimensionality reduction
- **Spectral clustering**: Eigendecompose the graph Laplacian
- **Condition number**: `max(λ)/min(λ)` → numerical stability indicator

### Implementation
```python
from src.math_foundations.linear_algebra import eigendecomposition

eigenvalues, eigenvectors = eigendecomposition([[2, 1], [1, 2]])
# eigenvalues = [3.0, 1.0]  (sorted descending)
```

---

## Why This All Matters

| ML Concept | Linear Algebra Foundation |
|---|---|
| Neural network layers | Matrix multiplication: `z = Wx + b` |
| Loss gradients | Vectors of partial derivatives |
| PCA | Eigendecomposition of covariance matrix |
| Embeddings | Vectors in high-dimensional space |
| Attention | `softmax(QKᵀ / √d) V` — matrix operations throughout |
| Batch processing | Matrix multiply handles entire batches at once |
| Backpropagation | Transposed weight matrices propagate gradients |

---

## Verification

All implementations are verified against NumPy:

```python
from src.math_foundations.linear_algebra import verify_implementations
results = verify_implementations()
# dot_product: ✓ PASS
# norm_l2: ✓ PASS
# transpose: ✓ PASS
# matrix_multiply: ✓ PASS
# covariance_matrix: ✓ PASS
```

---

## References

- Gilbert Strang, *Linear Algebra and Its Applications*
- 3Blue1Brown, *Essence of Linear Algebra* (YouTube series — highly recommended for visual intuition)
- Ian Goodfellow et al., *Deep Learning*, Chapter 2: Linear Algebra
