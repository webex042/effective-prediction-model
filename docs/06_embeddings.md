# Dense Word & Token Embeddings

> **Stage 4 — NLP Foundations: Vector Semantics, Lookup Tables & Geometry**

---

## 1. Intuition

In classical natural language processing, words were treated as atomic symbols represented by **one-hot encodings**:
$$\text{"bearing"} = [1, 0, 0, \dots, 0]^T, \quad \text{"vibration"} = [0, 1, 0, \dots, 0]^T$$

### Why One-Hot Encoding Fails
1. **Sparsity & Extreme Dimensionality**: With a vocabulary of size $|V| = 50,000$, every word vector has 49,999 zeros and a single one.
2. **Orthogonality (Zero Semantic Similarity)**:
   $$\text{"vibration"} \cdot \text{"oscillation"} = 0$$
   $$\text{"vibration"} \cdot \text{"banana"} = 0$$
   The dot product between any two distinct one-hot vectors is identically zero. The model cannot infer that an algorithm that knows how to handle "vibration" should behave similarly when encountering "oscillation".

### The Distributional Hypothesis
> *"You shall know a word by the company it keeps."* — J.R. Firth (1957)

Words that appear in similar textual contexts share semantic meaning. **Dense Embeddings** project words into a continuous low-dimensional vector space $\mathbb{R}^D$ (where $D \ll |V|$, e.g., $D \in [32, 256]$) such that:
$$\cos(\mathbf{e}_{\text{vibration}}, \mathbf{e}_{\text{oscillation}}) \approx 1.0$$

---

## 2. Mathematical Formulation

### 2.1 The Embedding Lookup Operation
Let $E \in \mathbb{R}^{|V| \times D}$ be the embedding matrix (the lookup table), where row $i$ contains the $D$-dimensional representation of token $i$.

Given a sequence of token indices $t = (t_1, t_2, \dots, t_L)$ with $t_i \in \{0, \dots, |V|-1\}$:
$$X = E[t, :] \in \mathbb{R}^{L \times D}$$

Mathematically, this is identical to multiplying a one-hot vector $\mathbf{v} \in \mathbb{R}^{|V|}$ by matrix $E$:
$$\mathbf{e} = \mathbf{v}^T E$$
However, indexing directly is $\mathcal{O}(1)$ time, bypassing the expensive matrix multiplication with sparse zeros.

### 2.2 Bag of Embeddings (Sentence Representation)
To pool token representations into a fixed-size vector for classification:
$$\mathbf{h}_{\text{doc}} = \frac{1}{L} \sum_{i=1}^L \mathbf{e}_{t_i}$$

---

## 3. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| **Lookup ($L$ tokens)** | $\mathcal{O}(L)$ | $\mathcal{O}(L \cdot D)$ |
| **Mean Pooling** | $\mathcal{O}(L \cdot D)$ | $\mathcal{O}(D)$ |
| **Backward Gradient Accumulation** | $\mathcal{O}(L \cdot D)$ | $\mathcal{O}(|V| \cdot D)$ |

---

## 4. From-Scratch Implementation (NumPy)

```python
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.W = np.random.randn(vocab_size, embedding_dim) * np.sqrt(1.0 / embedding_dim)
        self.dW = np.zeros_like(self.W)

    def forward(self, token_ids):
        self.last_input = token_ids
        return self.W[token_ids]

    def backward(self, grad_output):
        self.dW.fill(0.0)
        np.add.at(self.dW, self.last_input, grad_output)
```

---

## 5. Limitations

- **Polysemy & Context Insensitivity**: Static embeddings assign the exact same vector to a word regardless of context. For example, "bearing" in "faulty ball bearing" (mechanical hardware) vs "bearing the load" (verb) receives the identical vector.
- Solution: **Self-Attention & Transformers**, which dynamically compute contextualized representations.
