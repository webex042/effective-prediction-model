# The Transformer Architecture

> **Stage 4 — NLP / Stage 6 — Transformers: Encoder Architecture, Positional Encoding & Classification**

---

## 1. Overview

The **Transformer** (Vaswani et al., 2017) completely eliminated recurrence and convolutions, relying entirely on self-attention to model dependencies between input and output tokens.

```
                  ┌───────────────────────────────┐
                  │      Classification Head      │
                  └──────────────▲────────────────┘
                                 │
                  ┌──────────────┴────────────────┐
                  │    Masked Average Pooling     │
                  └──────────────▲────────────────┘
                                 │
                  ┌──────────────┴────────────────┐
                  │   Transformer Encoder Block   │
                  │  ┌─────────────────────────┐  │
                  │  │ Add & LayerNorm (MLP)   │  │
                  │  ├─────────────────────────┤  │
                  │  │ Add & LayerNorm (MHA)   │  │
                  │  └─────────────────────────┘  │
                  └──────────────▲────────────────┘
                                 │
                  ┌──────────────┴────────────────┐
                  │  Embedding + Positional Enc   │
                  └──────────────▲────────────────┘
                                 │
                            Token IDs
```

---

## 2. Key Components

### 2.1 Positional Encodings
Because Self-Attention operates on sets of vectors without inherent ordering:
$$\text{Attention}(P X) = P \text{Attention}(X)$$
The Transformer requires explicit positional information injected into the embeddings:
$$X_{\text{input}} = \text{Embedding}(t) + PE(pos)$$

Using sinusoidal functions of varying frequencies:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
This geometric formulation enables the network to attend to relative distances because for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear rotation of $PE_{pos}$.

### 2.2 Residual Connections & Layer Normalization
Deep Transformer architectures rely on Pre-LN or Post-LN residual layers:
$$x_1 = \text{LayerNorm}(x + \text{Dropout}(\text{MultiHeadAttention}(x, x, x)))$$
$$x_2 = \text{LayerNorm}(x_1 + \text{Dropout}(\text{FFN}(x_1)))$$

---

## 3. Computational Complexity Analysis

For sequence length $L$ and hidden dimension $d$:
- **Self-Attention**: $\mathcal{O}(L^2 d)$ time, $\mathcal{O}(L^2)$ memory for the attention matrix.
- **Feedforward MLP**: $\mathcal{O}(L d^2)$ time.
- **Total per layer**: $\mathcal{O}(L^2 d + L d^2)$.

---

## 4. Empirical Performance on Text Classification

On our industrial maintenance incident reports dataset (300 logs across 3 severity classes):

| Model Architecture | Accuracy | Inference Latency ($\mu s$) | Key Strength |
|---|---|---|---|
| **TF-IDF + Logistic Regression** | 88.33% | **1.2 $\mu s$** | Blazing fast, zero parameter tuning |
| **Bag-of-Embeddings Neural Net** | 90.00% | 12.4 $\mu s$ | Captures semantic synonymy |
| **Self-Attention Transformer** | **95.00%** | 35.8 $\mu s$ | Contextual interaction, highest accuracy |
