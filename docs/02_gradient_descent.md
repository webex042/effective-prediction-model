# Gradient Descent & Optimization

> **Stage 1 — Mathematical Foundation**

## Overview

Machine learning IS optimization. We define a **loss function** `L(θ)` that measures how bad our model is, then find the parameters `θ` that minimize it. Every optimizer is a different strategy for finding that minimum.

---

## 1. The Key Idea

### Intuition
Imagine you're blindfolded on a hilly landscape, trying to find the lowest valley.

1. **Feel the slope** under your feet (compute the gradient)
2. **Step downhill** (update parameters in the opposite direction of the gradient)
3. **Repeat** until you reach the bottom

### Math
The gradient `∇L(θ)` is a vector pointing in the direction of **steepest increase**.

To minimize, we move **opposite** to the gradient:

```
θ_{new} = θ_{old} - α × ∇L(θ_{old})
```

Where `α` (learning rate) controls how big each step is:
- **Too large** → overshoot the minimum (loss explodes)
- **Too small** → takes forever to converge
- **Just right** → steady convergence

---

## 2. Gradient Descent (Vanilla/Full-Batch)

### Algorithm
```
for each epoch:
    gradient = compute_gradient(loss, ALL_data, params)
    params = params - learning_rate × gradient
```

### Properties
| Property | Value |
|---|---|
| Computes gradient on | Entire dataset |
| Each step is | Deterministic |
| Convergence | Smooth but slow |
| Good for | Small datasets, guaranteed convergence |
| Bad for | Large datasets (slow per step) |

### Implementation
```python
from src.math_foundations.optimization import GradientDescent, minimize

gd = GradientDescent(learning_rate=0.1)

# Minimize f(x,y) = x² + y²
f = lambda p: p[0]**2 + p[1]**2
grad_f = lambda p: 2 * p

result = minimize(f, grad_f, x0=[5.0, 3.0], optimizer=gd, n_steps=100)
# final_loss ≈ 0.0, final_params ≈ [0, 0]
```

---

## 3. Stochastic Gradient Descent (SGD)

### Intuition
Instead of computing the gradient over the **entire** dataset (expensive), use a random **mini-batch** to estimate it. The estimate is noisy but unbiased.

Like polling 100 people instead of the entire country — faster, slightly less accurate, but good enough.

### Key Insight
The noise from random batches actually **helps**:
- Escapes shallow local minima and saddle points
- Acts as implicit regularization (prevents overfitting)

### Algorithm
```
for each epoch:
    shuffle data into mini-batches
    for each mini-batch:
        gradient = compute_gradient(loss, mini_batch, params)
        params = params - learning_rate × gradient
```

---

## 4. SGD with Momentum

### Intuition
Think of a **ball rolling down a hill**. Without momentum, the ball stops and changes direction with every little bump. With momentum, the ball builds up speed in consistent directions and powers through small bumps.

### Math
```
v_t = β × v_{t-1} + ∇L(θ_t)       ← accumulate velocity
θ_{t+1} = θ_t - α × v_t            ← update with velocity
```

Where `β` is the momentum coefficient:
- `β = 0` → pure SGD (no memory)
- `β = 0.9` → remembers ~10 past gradients (typical)
- `β = 0.99` → remembers ~100 past gradients

### Why It Works
| Problem | How Momentum Helps |
|---|---|
| Noisy gradients | Averages them out (smoother trajectory) |
| Narrow valleys | Builds speed in the consistent direction |
| Saddle points | Accumulated velocity carries past them |
| Oscillation | Opposing oscillations cancel in velocity |

### Implementation
```python
from src.math_foundations.optimization import MomentumSGD

opt = MomentumSGD(learning_rate=0.05, momentum=0.9)
# Use the same minimize() interface
```

---

## 5. Adam (Adaptive Moment Estimation)

### Intuition
Adam combines **two ideas**:
1. **Momentum** (1st moment): remember the direction you've been going
2. **Adaptive learning rate** (2nd moment): parameters with large gradients get smaller steps, parameters with small gradients get larger steps

It's like giving each parameter its own personalized learning rate that adapts as training progresses.

### Math
```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t          ← 1st moment (mean of gradients)
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²          ← 2nd moment (mean of squared gradients)

# Bias correction (important early in training)
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

# Update (adaptive step size per parameter)
θ_{t+1} = θ_t - α × m̂_t / (√v̂_t + ε)
```

### Default Hyperparameters
| Parameter | Default | What It Does |
|---|---|---|
| α (learning rate) | 0.001 | Base step size |
| β₁ | 0.9 | Decay rate for 1st moment (direction) |
| β₂ | 0.999 | Decay rate for 2nd moment (step scaling) |
| ε | 1e-8 | Prevents division by zero |

### Why Adam Is the Default Choice
- Works well across a wide range of problems
- Less sensitive to learning rate choice than SGD
- Fast convergence
- Handles sparse gradients well (important for NLP)

### Implementation
```python
from src.math_foundations.optimization import Adam

adam = Adam(learning_rate=0.001)
# Works on everything from simple quadratics to Rosenbrock
```

---

## 6. Comparing Optimizers

### Test Function: `f(x,y) = x² + y²`

| Optimizer | Steps to Converge | Final Loss |
|---|---|---|
| GD (lr=0.1) | ~50 | < 0.001 |
| Momentum (lr=0.05, β=0.9) | ~30 | < 0.001 |
| Adam (lr=0.1) | ~80 | < 0.001 |

### Test Function: Rosenbrock (much harder)
```
f(x,y) = (1-x)² + 100(y-x²)²
```
Minimum at (1, 1). Has a narrow, curved valley.

| Optimizer | Steps | Final Loss |
|---|---|---|
| GD (lr=0.001) | 10000 | Still large |
| Adam (lr=0.005) | 5000 | < 0.1 |

### Visual Comparison
```python
from src.math_foundations.optimization import compare_optimizers

results = compare_optimizers(
    f=lambda p: p[0]**2 + p[1]**2,
    grad_f=lambda p: 2 * p,
    x0=[5.0, 4.0],
    n_steps=200,
)
```

---

## 7. Learning Rate: The Most Important Hyperparameter

```
Too small (α = 0.00001):  θ barely changes → takes forever
Just right (α = 0.01):    steady convergence → good!
Too large (α = 10):       overshoots → loss explodes → NaN
```

### Practical Advice
| Situation | Suggested Learning Rate |
|---|---|
| Starting a new model | Try 0.001 with Adam |
| Fine-tuning | 10x-100x smaller than training LR |
| Loss is NaN or exploding | Reduce LR by 10x |
| Loss plateaus early | Try increasing LR by 2-5x |

---

## 8. Connection to Neural Network Training

When you write this in PyTorch:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in dataloader:
    loss = model(batch)       # Forward pass
    loss.backward()           # Compute gradients (calculus + chain rule)
    optimizer.step()          # Apply update rule (this module!)
    optimizer.zero_grad()     # Reset gradients
```

The `optimizer.step()` is doing **exactly** what our `Adam.step()` does — just on millions of parameters instead of 2.

---

## Complexity Analysis

| Optimizer | Memory per Parameter | Compute per Step |
|---|---|---|
| GD | 0 extra | O(1) |
| SGD | 0 extra | O(1) |
| Momentum | 1 extra (velocity) | O(1) |
| Adam | 2 extra (m and v) | O(1) |

Adam uses 3× the memory of SGD (original params + m + v), but the per-step compute is the same O(1) per parameter.

---

## Limitations

- All these optimizers find **local** minima, not necessarily the **global** minimum
- For neural networks, the loss landscape has many local minima, but empirically most of them are similarly good
- Learning rate scheduling (reducing LR during training) often improves final performance
- SGD+Momentum with proper tuning can sometimes outperform Adam in generalization

---

## References

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015)
- Ruder, *An Overview of Gradient Descent Optimization Algorithms* (2017)
- Goodfellow et al., *Deep Learning*, Chapter 8: Optimization for Training Deep Models
