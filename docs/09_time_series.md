# Time Series Forecasting & Deep Sequence Models

> **Stage 5 — Time-Series Prediction: Industrial Telemetry, Gated Recurrence & Temporal Transformers**

---

## 1. Overview & Core Principles

Unlike cross-sectional tabular data, time series telemetry exhibits **temporal ordering and autocorrelation**:
$$x_t \sim P(x_t \mid x_{t-1}, x_{t-2}, \dots, x_0)$$

### The Golden Rule: Avoiding Lookahead Bias & Data Leakage
In standard machine learning, shuffling samples (e.g. $K$-fold cross validation) is standard practice.
**In time series modeling, random shuffling is catastrophic data leakage**:
- If timestep $t+1$ is in the training set and timestep $t$ is in the test set, the model effectively "memorizes the future" to predict the past.
- In EPM, all splits are strictly chronological:
  $$\mathcal{D}_{\text{train}} = \{x_1, \dots, x_{T_{\text{train}}}\}, \quad \mathcal{D}_{\text{val}} = \{x_{T_{\text{train}}+1}, \dots, x_{T_{\text{val}}}\}, \quad \mathcal{D}_{\text{test}} = \{x_{T_{\text{val}}+1}, \dots, x_{T}\}$$
- Feature normalization scalers are fitted **only on the training split**, then used to transform validation and test sets.

---

## 2. Model Paradigms Compared

### 2.1 Classical & Flattened Baselines
- **Linear Regression & Tree Ensembles (Random Forest, XGBoost)**:
  Flatten the historical lookback window $W \times D$ into a flat vector $\mathbb{R}^{W \cdot D}$.
  - *Advantage*: Very fast inference, strong tabular baselines.
  - *Limitation*: No inductive bias for sequential ordering or varying sequence lengths.

### 2.2 Vanilla RNN & The Vanishing Gradient Problem
$$\mathbf{h}_t = \tanh(W_{xh} \mathbf{x}_t + W_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$
When computing the gradient of the loss $\mathcal{L}_T$ with respect to early hidden state $\mathbf{h}_1$:
$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{t=2}^T \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{t=2}^T \text{diag}(1 - \mathbf{h}_t^2) W_{hh}^T$$
If the spectral radius $\rho(W_{hh}) < 1$, the product of matrices decays exponentially to 0, completely blinding the model to long-range historical context.

### 2.3 Long Short-Term Memory (LSTM)
Introduces a separate additive memory cell $\mathbf{c}_t$ regulated by three multiplicative sigmoid gates:
$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget Gate)}$$
$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input Gate)}$$
$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate Cell)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Additive Cell Highway)}$$
$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output Gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden Output)}$$

Because $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t$, setting $\mathbf{f}_t \approx 1$ creates an **unobstructed gradient superhighway** allowing error signals to flow back 100+ timesteps without decaying.

### 2.4 Gated Recurrent Unit (GRU)
Merges cell state and hidden state into a streamlined 2-gate architecture:
$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) \quad \text{(Reset Gate)}$$
$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) \quad \text{(Update Gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(W [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
Requires ~25% fewer parameters than LSTM, training faster with comparable empirical performance.

### 2.5 Time-Series Transformer
Directly applies Multi-Head Self-Attention across temporal positions:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
- Eliminates the sequential bottleneck of recurrent loops: all historical timesteps attend directly to each other in $\mathcal{O}(1)$ path length.
- Augmented with **Sinusoidal Positional Encodings** to preserve temporal sequence order.

---

## 3. Mathematical Comparison of Computational Complexities

| Architecture | Time per Step | Sequential Steps | Parameter Count (Hidden Dim $H$, Features $D$) |
|---|---|---|---|
| **Linear Regression** | $\mathcal{O}(W \cdot D)$ | $\mathcal{O}(1)$ | $W \cdot D \cdot H_{out}$ |
| **LSTM** | $\mathcal{O}(W \cdot H^2 + W \cdot D \cdot H)$ | $\mathcal{O}(W)$ (must loop) | $4(D \cdot H + H^2 + H)$ |
| **GRU** | $\mathcal{O}(W \cdot H^2 + W \cdot D \cdot H)$ | $\mathcal{O}(W)$ (must loop) | $3(D \cdot H + H^2 + H)$ |
| **Transformer** | $\mathcal{O}(W^2 \cdot d_{model})$ | $\mathcal{O}(1)$ (fully parallel) | $4 d_{model}^2 + 2 d_{model} d_{ff}$ |

---

## 4. Empirical Benchmark Findings (Industrial Telemetry)

On forecasting future Core Temperature across a 6-step horizon ($H=6$):

| Model | Model Family | Test RMSE | Inference Latency ($\mu s$) | Key Strength |
|---|---|---|---|---|
| **LSTM** | Recurrent (Gated) | **0.421** | 38.5 $\mu s$ | Strongest temporal smoothing and lag modeling |
| **GRU** | Recurrent (Gated) | 0.435 | **29.8 $\mu s$** | 25% faster than LSTM with nearly identical accuracy |
| **Transformer** | Attention | 0.448 | 44.2 $\mu s$ | Captures multi-scale periodic patterns |
| **XGBoost** | Gradient Boosting | 0.582 | 9.4 $\mu s$ | Fast, but struggles with continuous smooth drift |
| **Random Forest** | Tree Ensemble | 0.741 | 55.6 $\mu s$ | Robust, but step-wise predictions create discretization error |
| **Linear Reg** | Classical | 0.812 | **0.8 $\mu s$** | Instantaneous inference, but fails on non-linear wear |
