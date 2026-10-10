# The Self-Attention Mechanism

> **Stage 4 — NLP / Stage 6 — Transformers: Scaled Dot-Product, Multi-Head & Attention Maps**

---

## 1. Intuition: The Information Retrieval Analogy

Before attention, sequence models (RNNs, LSTMs) compressed an entire sentence of arbitrary length into a single fixed-size bottleneck vector $h_T$. Long-range relationships were lost, and sentences could not dynamically focus on relevant tokens.

### Query, Key, Value Framework
Inspired by database retrieval:
- **Query ($Q$)**: *"What kind of information am I looking for?"*
- **Key ($K$)**: *"What content or attributes do I contain?"*
- **Value ($V$)**: *"What actual information do I communicate if matched?"*

In **Self-Attention**, every token in the sequence acts as a Query, Key, and Value simultaneously:
- A token queries all other tokens in the sentence.
- It computes a compatibility score with every Key.
- It extracts a weighted sum of their Values.

---

## 2. Mathematical Formulation

### 2.1 Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

Given sequence length $L$ and key dimension $d_k$:
1. **Dot-Product Similarity Matrix**:
   $$S = Q K^T \in \mathbb{R}^{L \times L}$$
   $S_{i, j} = q_i \cdot k_j$ measures the alignment between token $i$ and token $j$.

2. **Scaling Factor $\frac{1}{\sqrt{d_k}}$**:
   Why is this scale critical?
   Assume components of $q$ and $k$ are independent random variables with zero mean and variance 1:
   $$\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$
   $$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
   As $d_k$ increases (e.g. $d_k = 64$), the variance of the dot product grows to 64, meaning dot products frequently exceed $\pm 15$.
   At large inputs, the $\text{softmax}$ function produces extreme distributions (e.g., $0.9999$ and $0.0001$) where gradients vanish ($\frac{\partial \text{softmax}}{\partial z} \approx 0$).
   Dividing by $\sqrt{d_k}$ normalizes the variance back to 1.0:
   $$\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(q \cdot k) = 1.0$$

3. **Softmax Normalization**:
   $$A_{i, j} = \frac{\exp(S_{i, j} / \sqrt{d_k})}{\sum_{m=1}^L \exp(S_{i, m} / \sqrt{d_k})}$$
   $A \in \mathbb{R}^{L \times L}$ is the **Attention Weights Matrix**, where row $i$ is a probability distribution over all tokens $j$ ($\sum_j A_{i, j} = 1.0$).

4. **Weighted Value Aggregation**:
   $$\text{Output}_i = \sum_{j=1}^L A_{i, j} v_j \in \mathbb{R}^{d_v}$$

---

## 3. Multi-Head Attention

A single attention head can only attend to a single weighted subspace at once. **Multi-Head Attention** projects queries, keys, and values $h$ times with separate parameter matrices:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

- Allows simultaneous tracking of different linguistic and physical phenomena (e.g., Head 1 tracks adjective-noun modifiers; Head 2 tracks causal fault indicators).

---

## 4. Computational Complexity

| Metric | Self-Attention | Recurrent (RNN/LSTM) |
|---|---|---|
| **Complexity per Layer** | $\mathcal{O}(L^2 \cdot d)$ | $\mathcal{O}(L \cdot d^2)$ |
| **Sequential Operations** | $\mathcal{O}(1)$ (fully parallel) | $\mathcal{O}(L)$ (must iterate step-by-step) |
| **Maximum Path Length** | $\mathcal{O}(1)$ (direct connection) | $\mathcal{O}(L)$ (gradient passes through $L$ steps) |

Notice the trade-off: **Self-Attention is $\mathcal{O}(L^2)$ in sequence length**, which will motivate our research into **Efficient Attention (Stage 11)**!

---

## 5. From-Scratch Implementation (NumPy)

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    # Numerically stable softmax
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights
```
