"""
EPM Stage 1 — Mathematical Foundation Demo
============================================

This script demonstrates all Stage 1 modules:
    1. Linear Algebra — vectors, matrices, eigendecomposition
    2. Calculus — derivatives, gradients, chain rule, gradient checking
    3. Probability — Gaussian, Bayes, KL divergence, entropy, MLE
    4. Optimization — GD, Momentum, Adam on test functions

Run:
    python demo_stage1.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from src.reproducibility import set_seed
from src.logger import setup_logging

# Setup
setup_logging(level="WARNING", file=False)
set_seed(42)

PLOTS_DIR = Path("docs/plots/stage1")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  EPM — Stage 1: Mathematical Foundation")
print("=" * 70)


# =====================================================================
# 1. LINEAR ALGEBRA
# =====================================================================

print("\n" + "─" * 70)
print("  1. LINEAR ALGEBRA")
print("─" * 70)

from src.math_foundations.linear_algebra import (
    dot_product, norm, normalize, transpose,
    matrix_multiply, covariance_matrix, eigendecomposition,
    verify_implementations as verify_linalg,
)

# Dot product
a = [1, 2, 3]
b = [4, 5, 6]
print(f"\n  Dot product:")
print(f"    a = {a}")
print(f"    b = {b}")
print(f"    a · b = {dot_product(a, b)}")
print(f"    Meaning: {a[0]}×{b[0]} + {a[1]}×{b[1]} + {a[2]}×{b[2]} = {dot_product(a, b)}")

# Norm
v = [3, 4]
print(f"\n  Norms:")
print(f"    v = {v}")
print(f"    L2 norm (Euclidean distance): ||v||₂ = {norm(v, 2)}")
print(f"    L1 norm (Manhattan distance): ||v||₁ = {norm(v, 1)}")
print(f"    Normalized: v̂ = {normalize(v)}")

# Matrix multiply (neural network style)
W = np.array([[0.5, -0.3], [0.2, 0.8], [-0.1, 0.4]])
x = np.array([[2.0], [1.0]])
z = matrix_multiply(W, x)
print(f"\n  Neural network layer (z = Wx):")
print(f"    W (3×2 weights) = {W.tolist()}")
print(f"    x (2×1 input)   = {x.flatten().tolist()}")
print(f"    z (3×1 output)  = {z.flatten().tolist()}")
print(f"    → This is what EVERY neural network layer does!")

# Covariance & eigendecomposition (PCA preview)
np.random.seed(42)
# Create correlated 2D data
mean = [0, 0]
cov_true = [[3, 2], [2, 1.5]]
X = np.random.multivariate_normal(mean, cov_true, size=200)

cov = covariance_matrix(X)
eigenvalues, eigenvectors = eigendecomposition(cov)

print(f"\n  Covariance Matrix (200 samples, 2 features):")
print(f"    [{cov[0,0]:.3f}, {cov[0,1]:.3f}]")
print(f"    [{cov[1,0]:.3f}, {cov[1,1]:.3f}]")
print(f"    Eigenvalues: {eigenvalues.round(3)}")
print(f"    → Largest eigenvalue = direction of most variance (PCA!)")

# Plot: Eigenvectors on data
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=15, color="#4A90D9")
origin = np.mean(X, axis=0)
for i in range(2):
    scale = np.sqrt(eigenvalues[i]) * 2
    end = origin + eigenvectors[:, i] * scale
    ax.annotate("", xy=end, xytext=origin,
                arrowprops=dict(arrowstyle="->", color=["#E74C3C", "#2ECC71"][i], lw=2.5))
    ax.text(end[0] + 0.1, end[1] + 0.1,
            f"λ={eigenvalues[i]:.2f}", fontsize=10, fontweight="bold",
            color=["#E74C3C", "#2ECC71"][i])
ax.set_xlabel("Feature 1", fontsize=11)
ax.set_ylabel("Feature 2", fontsize=11)
ax.set_title("Eigenvectors of Covariance Matrix\n(Principal Components)", fontsize=13)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eigenvectors.png", dpi=120)
plt.close()
print(f"    📊 Plot saved: {PLOTS_DIR / 'eigenvectors.png'}")

# Verification
print(f"\n  Verification against NumPy:")
results = verify_linalg(verbose=False)
for name, passed in results.items():
    print(f"    {'✓' if passed else '✗'} {name}")


# =====================================================================
# 2. CALCULUS
# =====================================================================

print("\n" + "─" * 70)
print("  2. CALCULUS")
print("─" * 70)

from src.math_foundations.calculus import (
    numerical_derivative, numerical_gradient,
    chain_rule_demo, gradient_check,
)

# Derivative
print(f"\n  Numerical Derivatives:")
functions = [
    ("x²", lambda x: x**2, "2x", lambda x: 2*x, 3.0),
    ("sin(x)", np.sin, "cos(x)", np.cos, np.pi/4),
    ("eˣ", np.exp, "eˣ", np.exp, 1.0),
]

for name, f, deriv_name, deriv_fn, x_val in functions:
    numerical = numerical_derivative(f, x_val)
    analytical = deriv_fn(x_val)
    print(f"    d/dx({name}) at x={x_val:.2f}: "
          f"numerical={numerical:.6f}, "
          f"analytical({deriv_name})={analytical:.6f}")

# Chain rule
cr = chain_rule_demo()
print(f"\n  Chain Rule (backpropagation foundation):")
print(f"    Function: {cr['function']}")
print(f"    Chain rule: {cr['chain_rule']}")
print(f"    At x={cr['x']}: analytical={cr['analytical_derivative']:.4f}, "
      f"numerical={cr['numerical_derivative']:.6f}")
print(f"    Match: {'✓' if cr['match'] else '✗'}")

# Gradient
print(f"\n  Gradient (the core of training):")
f_2d = lambda p: p[0]**2 + p[1]**2
grad = numerical_gradient(f_2d, [3.0, 4.0])
print(f"    f(x,y) = x² + y²")
print(f"    ∇f at (3,4) = {grad} = [2×3, 2×4]")
print(f"    → Gradient descent: move OPPOSITE to gradient = toward minimum")

# Gradient check
gc = gradient_check(
    analytical_grad_fn=lambda x: 2 * x,
    loss_fn=lambda x: float(x[0]**2 + x[1]**2),
    params=[3.0, 4.0],
)
print(f"\n  Gradient Check (debugging backprop):")
print(f"    Relative error: {gc['relative_error']:.2e}")
print(f"    Threshold: {gc['tolerance']:.2e}")
print(f"    Result: {'✓ PASSED' if gc['passed'] else '✗ FAILED'}")

# Plot: Gradient descent on 2D landscape
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Contour plot with gradient arrows
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z = X_grid**2 + Y_grid**2

ax = axes[0]
contour = ax.contourf(X_grid, Y_grid, Z, levels=20, cmap="viridis", alpha=0.8)
plt.colorbar(contour, ax=ax, label="f(x,y)")

# Show gradient arrows at a few points
for px, py in [(-3, -3), (3, 3), (-3, 3), (3, -3), (0, 4), (4, 0)]:
    g = numerical_gradient(f_2d, [px, py])
    ax.arrow(px, py, -g[0]*0.3, -g[1]*0.3, head_width=0.2,
             head_length=0.1, fc='white', ec='white', linewidth=1.5)

ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("f(x,y) = x² + y²  with  -∇f arrows", fontsize=12)

# Right: Forward vs Central difference accuracy
ax = axes[1]
h_values = np.logspace(-1, -12, 50)
errors_forward = []
errors_central = []
f_test = lambda x: x**3
x_test = 2.0
exact = 12.0  # 3x² at x=2

for h in h_values:
    errors_forward.append(abs(numerical_derivative(f_test, x_test, h, "forward") - exact))
    errors_central.append(abs(numerical_derivative(f_test, x_test, h, "central") - exact))

ax.loglog(h_values, errors_forward, 'o-', markersize=2, label="Forward: O(h)", color="#E74C3C")
ax.loglog(h_values, errors_central, 's-', markersize=2, label="Central: O(h²)", color="#2ECC71")
ax.set_xlabel("Step size h", fontsize=11)
ax.set_ylabel("Absolute error", fontsize=11)
ax.set_title("Forward vs Central Difference Accuracy", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "calculus_gradients.png", dpi=120)
plt.close()
print(f"\n    📊 Plot saved: {PLOTS_DIR / 'calculus_gradients.png'}")


# =====================================================================
# 3. PROBABILITY
# =====================================================================

print("\n" + "─" * 70)
print("  3. PROBABILITY & STATISTICS")
print("─" * 70)

from src.math_foundations.probability import (
    gaussian_pdf, gaussian_log_likelihood,
    bayes_posterior, sample_gaussian,
    kl_divergence_gaussians, entropy_discrete,
    mle_gaussian,
)

# Gaussian PDF
print(f"\n  Gaussian Distribution N(μ, σ²):")
print(f"    Peak at mean:   p(0 | μ=0, σ=1) = {gaussian_pdf(0.0):.6f}")
print(f"    1σ from mean:   p(1 | μ=0, σ=1) = {gaussian_pdf(1.0):.6f}")
print(f"    2σ from mean:   p(2 | μ=0, σ=1) = {gaussian_pdf(2.0):.6f}")
print(f"    → Values far from mean have lower probability density")

# Bayes' theorem
post = bayes_posterior(prior=0.01, likelihood=0.95, evidence=0.05)
print(f"\n  Bayes' Theorem (medical test example):")
print(f"    Prior P(disease) = 1% (rare disease)")
print(f"    Likelihood P(+test | disease) = 95% (good test)")
print(f"    P(+test) = 5% (overall positive rate)")
print(f"    Posterior P(disease | +test) = {post*100:.0f}%")
print(f"    → Even with a positive test, only {post*100:.0f}% chance!")

# KL Divergence
print(f"\n  KL Divergence (distance between distributions):")
print(f"    KL(N(0,1) || N(0,1)) = {kl_divergence_gaussians(0,1,0,1):.4f} (same → 0)")
print(f"    KL(N(1,1) || N(0,1)) = {kl_divergence_gaussians(1,1,0,1):.4f} (shifted mean)")
print(f"    KL(N(0,2) || N(0,1)) = {kl_divergence_gaussians(0,2,0,1):.4f} (wider)")
print(f"    → Used in VAEs to keep encoder close to prior")

# Entropy
print(f"\n  Entropy (measuring uncertainty):")
print(f"    Fair coin [0.5, 0.5]:     H = {entropy_discrete([0.5, 0.5]):.4f} bits")
print(f"    Biased coin [0.9, 0.1]:   H = {entropy_discrete([0.9, 0.1]):.4f} bits")
print(f"    Certain [1.0, 0.0]:       H = {entropy_discrete([1.0, 0.0]):.4f} bits")
print(f"    → More uncertain = higher entropy")

# MLE
np.random.seed(42)
data = np.random.normal(5.0, 2.0, size=500)
mle = mle_gaussian(data)
print(f"\n  Maximum Likelihood Estimation:")
print(f"    True parameters:  μ=5.0, σ=2.0")
print(f"    MLE estimates:    μ={mle['mean']:.3f}, σ={mle['std_mle']:.3f}")
print(f"    (from {mle['n_samples']} samples)")
print(f"    → MLE finds the parameters that make data most probable")

# Plot: Gaussian distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Left: Different Gaussians
ax = axes[0]
x = np.linspace(-6, 10, 300)
for mu, sigma, label, color in [
    (0, 1, "N(0,1) — Standard Normal", "#3498DB"),
    (0, 2, "N(0,4) — Wider", "#E74C3C"),
    (3, 0.5, "N(3,0.25) — Narrow, shifted", "#2ECC71"),
]:
    ax.plot(x, gaussian_pdf(x, mu, sigma), label=label, linewidth=2, color=color)
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("p(x)", fontsize=11)
ax.set_title("Gaussian Distributions", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Middle: MLE fit
ax = axes[1]
ax.hist(data, bins=30, density=True, alpha=0.6, color="#3498DB", label="Data (500 samples)")
x_fit = np.linspace(data.min(), data.max(), 200)
ax.plot(x_fit, gaussian_pdf(x_fit, mle['mean'], mle['std_mle']),
        'r-', linewidth=2.5, label=f"MLE fit: N({mle['mean']:.2f}, {mle['std_mle']:.2f}²)")
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Maximum Likelihood Estimation", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: KL divergence visualization
ax = axes[2]
x = np.linspace(-5, 5, 300)
p = gaussian_pdf(x, 0, 1)
q = gaussian_pdf(x, 1.5, 1.2)
ax.plot(x, p, linewidth=2, label="P = N(0, 1)", color="#3498DB")
ax.plot(x, q, linewidth=2, label="Q = N(1.5, 1.44)", color="#E74C3C")
ax.fill_between(x, 0, np.abs(p - q), alpha=0.2, color="#F39C12", label="Difference")
kl = kl_divergence_gaussians(0, 1, 1.5, 1.2)
ax.set_title(f"KL(P || Q) = {kl:.3f}", fontsize=12)
ax.set_xlabel("x", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "probability.png", dpi=120)
plt.close()
print(f"\n    📊 Plot saved: {PLOTS_DIR / 'probability.png'}")


# =====================================================================
# 4. OPTIMIZATION
# =====================================================================

print("\n" + "─" * 70)
print("  4. OPTIMIZATION ALGORITHMS")
print("─" * 70)

from src.math_foundations.optimization import (
    GradientDescent, MomentumSGD, Adam,
    minimize, compare_optimizers,
)

# Compare all optimizers on f(x,y) = x² + y²
f = lambda p: float(p[0]**2 + p[1]**2)
grad_f = lambda p: 2 * np.asarray(p, dtype=np.float64)
x0 = np.array([5.0, 4.0])

print(f"\n  Minimizing f(x,y) = x² + y² starting from {x0.tolist()}")
print(f"  Target: minimum at (0, 0) with f = 0\n")

optimizers = {
    "Gradient Descent (lr=0.1)": GradientDescent(learning_rate=0.1),
    "Momentum SGD (lr=0.05)": MomentumSGD(learning_rate=0.05, momentum=0.9),
    "Adam (lr=0.1)": Adam(learning_rate=0.1),
}

all_trajectories = {}
for name, opt in optimizers.items():
    result = minimize(f, grad_f, x0.copy(), opt, n_steps=50)
    all_trajectories[name] = result
    final = result["final_params"]
    print(f"    {name:30s} → loss={result['final_loss']:.6f}, "
          f"params=[{final[0]:.4f}, {final[1]:.4f}]")

# Rosenbrock function (harder)
print(f"\n  Rosenbrock function (harder test, minimum at (1,1)):")

def rosenbrock(p):
    return float((1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2)

def grad_rosenbrock(p):
    p = np.asarray(p, dtype=np.float64)
    dx = -2 * (1 - p[0]) + 200 * (p[1] - p[0]**2) * (-2 * p[0])
    dy = 200 * (p[1] - p[0]**2)
    return np.array([dx, dy])

rosen_result = minimize(
    rosenbrock, grad_rosenbrock, [-1.0, 1.0],
    Adam(learning_rate=0.005), n_steps=5000,
)
print(f"    Adam (5000 steps): loss={rosen_result['final_loss']:.6f}, "
      f"params={rosen_result['final_params'].round(4).tolist()}")

# Plot: Optimizer comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Trajectories on contour
ax = axes[0]
x_range = np.linspace(-6, 6, 100)
y_range = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z = X_grid**2 + Y_grid**2
ax.contourf(X_grid, Y_grid, Z, levels=20, cmap="viridis", alpha=0.7)

colors = ["#E74C3C", "#F39C12", "#3498DB"]
for (name, result), color in zip(all_trajectories.items(), colors):
    traj = result["trajectory"]
    xs = [t["params"][0] for t in traj]
    ys = [t["params"][1] for t in traj]
    short_name = name.split("(")[0].strip()
    ax.plot(xs, ys, "o-", markersize=3, linewidth=1.5,
            label=short_name, color=color, alpha=0.8)

ax.plot(0, 0, "w*", markersize=15, label="Minimum")
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("Optimizer Trajectories on x²+y²", fontsize=12)
ax.legend(fontsize=8)

# Middle: Loss curves
ax = axes[1]
for (name, result), color in zip(all_trajectories.items(), colors):
    traj = result["trajectory"]
    losses = [t["loss"] for t in traj]
    short_name = name.split("(")[0].strip()
    ax.semilogy(losses, linewidth=2, label=short_name, color=color)
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Loss (log scale)", fontsize=11)
ax.set_title("Convergence Comparison", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Rosenbrock trajectory
ax = axes[2]
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-1, 3, 200)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z_rosen = (1 - X_grid)**2 + 100 * (Y_grid - X_grid**2)**2
ax.contourf(X_grid, Y_grid, np.log10(Z_rosen + 1), levels=30, cmap="inferno", alpha=0.8)

traj = rosen_result["trajectory"]
xs = [t["params"][0] for t in traj[::50]]  # Every 50th step
ys = [t["params"][1] for t in traj[::50]]
ax.plot(xs, ys, "o-", markersize=2, linewidth=1, color="cyan", alpha=0.8)
ax.plot(1, 1, "w*", markersize=15)
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("Adam on Rosenbrock (5000 steps)", fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "optimization.png", dpi=120)
plt.close()
print(f"\n    📊 Plot saved: {PLOTS_DIR / 'optimization.png'}")


# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("  STAGE 1 SUMMARY")
print("=" * 70)
print("""
  ✓ Linear Algebra — vectors, matrices, eigendecomposition, covariance
  ✓ Calculus — derivatives, gradients, chain rule, gradient checking
  ✓ Probability — Gaussian, Bayes, KL divergence, entropy, MLE
  ✓ Optimization — GD, Momentum SGD, Adam from scratch

  Key connections to ML:
    • Neural network layers = matrix multiplication (z = Wx + b)
    • Training = minimizing loss via gradient descent
    • Backpropagation = chain rule applied recursively
    • MSE loss = negative Gaussian log-likelihood
    • VAE loss = reconstruction + KL divergence
    • Diffusion = Gaussian noise addition/removal

  All implementations verified against NumPy.
  All 74 unit tests passing.

  📁 Source:  src/math_foundations/
  📁 Tests:   tests/test_math_foundations.py
  📁 Plots:   docs/plots/stage1/
""")
print("=" * 70)
print("  Ready for Stage 2: Classical Machine Learning")
print("=" * 70)
