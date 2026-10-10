"""
Optimization Algorithms — From Scratch
========================================

Stage 1: Mathematical Foundation

WHY OPTIMIZATION?
    Machine learning IS optimization. We define a loss function L(θ)
    that measures how bad our model is, then we find the parameters θ
    that minimize L. Every optimizer here is a different strategy for
    finding that minimum.

WHAT'S IN THIS MODULE:
    - GradientDescent: vanilla full-batch gradient descent
    - SGD: stochastic gradient descent (mini-batch)
    - MomentumSGD: SGD with momentum (like a ball rolling downhill)
    - Adam: adaptive learning rates + momentum (the default choice)

    Each optimizer is implemented as a class with a simple interface:
        optimizer.step(params, gradients) → updated params

ANALOGY:
    Imagine you're blindfolded on a hilly landscape, trying to find
    the lowest valley.

    - GD: Feel the slope everywhere, take a step downhill. Careful but slow.
    - SGD: Feel the slope at a random spot, step that way. Noisy but fast.
    - Momentum: Remember which direction you've been going. Like a ball
      that gains speed going downhill.
    - Adam: Adjust step size per parameter. Steep slopes → small steps.
      Flat regions → bigger steps.

USAGE:
    from src.math_foundations.optimization import Adam

    optimizer = Adam(learning_rate=0.001)

    for epoch in range(100):
        grads = compute_gradients(params, data)
        params = optimizer.step(params, grads)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Base Optimizer
# =============================================================================

class BaseOptimizer:
    """
    Base class for all optimizers.

    Every optimizer follows the same pattern:
    1. Receive current parameters and their gradients
    2. Apply an update rule
    3. Return updated parameters

    The only difference between optimizers is the UPDATE RULE.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        self.learning_rate = learning_rate
        self.step_count = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one optimization step. Override in subclasses."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset optimizer state (momentum, etc.)."""
        self.step_count = 0


# =============================================================================
# Gradient Descent (Vanilla)
# =============================================================================

class GradientDescent(BaseOptimizer):
    """
    Vanilla (full-batch) gradient descent.

    INTUITION:
        The simplest optimizer. At each step:
        1. Compute the gradient of the loss over the ENTIRE dataset
        2. Move parameters in the opposite direction of the gradient
        3. The learning rate controls how big each step is

    MATH:
        θ_{t+1} = θ_t - α × ∇L(θ_t)

        Where:
        - θ: model parameters
        - α: learning rate (step size)
        - ∇L: gradient of loss with respect to θ

    PROPERTIES:
        + Deterministic: same result every time (no randomness)
        + Guaranteed to decrease loss if learning rate is small enough
        - Slow: must process ALL data before each update
        - Can get stuck in saddle points
        - Fixed learning rate for all parameters

    WHEN TO USE:
        - Small datasets (< 10,000 samples)
        - When you need guaranteed convergence
        - For understanding — start here before moving to SGD/Adam

    Parameters
    ----------
    learning_rate : float
        Step size α. Typical range: [0.0001, 0.1].

    Examples
    --------
    >>> gd = GradientDescent(learning_rate=0.1)
    >>> params = np.array([5.0, 3.0])
    >>> grads = np.array([10.0, 6.0])
    >>> gd.step(params, grads)
    array([4., 2.4])
    """

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Apply one gradient descent step.

        θ_{t+1} = θ_t - α × ∇L(θ_t)
        """
        params = np.asarray(params, dtype=np.float64)
        grads = np.asarray(grads, dtype=np.float64)
        self.step_count += 1

        return params - self.learning_rate * grads


# =============================================================================
# Stochastic Gradient Descent (SGD)
# =============================================================================

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent.

    INTUITION:
        Instead of computing the gradient over the ENTIRE dataset
        (expensive for large datasets), use a random mini-batch
        to ESTIMATE the gradient. The estimate is noisy but unbiased.

        It's like polling 100 people instead of the entire country —
        faster, slightly less accurate, but good enough.

    MATH:
        Same update rule as GD, but ∇L is estimated from a mini-batch:

        θ_{t+1} = θ_t - α × ∇L_batch(θ_t)

        The noise from random batches actually HELPS:
        - Escapes shallow local minima and saddle points
        - Acts as implicit regularization (prevents overfitting)

    PROPERTIES:
        + Fast: processes only a small batch per update
        + Noise helps escape bad local minima
        + Good generalization (implicit regularization)
        - Noisy convergence (loss curve is jagged)
        - Learning rate is critical and hard to tune

    NOTE:
        In our implementation, we don't handle batching here.
        The gradient should already be computed on a mini-batch.
        This class just applies the update rule.

    Parameters
    ----------
    learning_rate : float
        Step size. For SGD, often needs to be smaller than for GD
        because gradients are noisier.
    """

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Same update rule as GD. The 'stochastic' part is in how grads are computed."""
        params = np.asarray(params, dtype=np.float64)
        grads = np.asarray(grads, dtype=np.float64)
        self.step_count += 1

        return params - self.learning_rate * grads


# =============================================================================
# SGD with Momentum
# =============================================================================

class MomentumSGD(BaseOptimizer):
    """
    SGD with Momentum.

    INTUITION:
        Think of a ball rolling down a hill. Without momentum, the ball
        stops and changes direction with every little bump. With momentum,
        the ball builds up speed in consistent directions and powers through
        small bumps.

        Momentum accumulates past gradients to smooth out the noisy
        SGD updates and accelerate convergence.

    MATH:
        v_t = β × v_{t-1} + ∇L(θ_t)          (accumulate velocity)
        θ_{t+1} = θ_t - α × v_t               (update parameters)

        Where:
        - v: velocity (accumulated gradient direction)
        - β: momentum coefficient (typically 0.9)
          β=0 → pure SGD (no memory)
          β=0.9 → remembers ~10 past gradients
          β=0.99 → remembers ~100 past gradients

    PROPERTIES:
        + Faster convergence than vanilla SGD
        + Smooths out noisy gradients
        + Helps escape narrow local minima
        - Introduces one extra hyperparameter (β)
        - Can overshoot if momentum is too high

    Parameters
    ----------
    learning_rate : float
        Step size α.
    momentum : float
        Momentum coefficient β. Default: 0.9.

    Examples
    --------
    >>> opt = MomentumSGD(learning_rate=0.01, momentum=0.9)
    >>> params = np.array([1.0, 2.0])
    >>> grads = np.array([0.5, -0.3])
    >>> params = opt.step(params, grads)  # First step
    >>> params = opt.step(params, grads)  # Velocity builds up
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity: Optional[np.ndarray] = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Apply one momentum SGD step.

        v_t = β × v_{t-1} + ∇L
        θ_{t+1} = θ_t - α × v_t
        """
        params = np.asarray(params, dtype=np.float64)
        grads = np.asarray(grads, dtype=np.float64)
        self.step_count += 1

        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # Accumulate velocity
        self.velocity = self.momentum * self.velocity + grads

        # Update parameters
        return params - self.learning_rate * self.velocity

    def reset(self) -> None:
        super().reset()
        self.velocity = None


# =============================================================================
# Adam (Adaptive Moment Estimation)
# =============================================================================

class Adam(BaseOptimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    INTUITION:
        Adam combines TWO ideas:
        1. Momentum: remember the direction you've been going (1st moment)
        2. Adaptive learning rate: parameters with large gradients get
           smaller steps, parameters with small gradients get larger steps
           (2nd moment)

        It's like giving each parameter its own personalized learning rate
        that adapts as training progresses.

    MATH:
        m_t = β₁ × m_{t-1} + (1-β₁) × g_t         (1st moment: mean of gradients)
        v_t = β₂ × v_{t-1} + (1-β₂) × g_t²         (2nd moment: mean of squared grads)

        Bias correction (important at the start when m,v are near zero):
        m̂_t = m_t / (1 - β₁ᵗ)
        v̂_t = v_t / (1 - β₂ᵗ)

        Update:
        θ_{t+1} = θ_t - α × m̂_t / (√v̂_t + ε)

        Where:
        - α: learning rate (typically 0.001)
        - β₁: decay rate for 1st moment (typically 0.9)
        - β₂: decay rate for 2nd moment (typically 0.999)
        - ε: small constant to prevent division by zero (typically 1e-8)

    WHY ADAM IS THE DEFAULT:
        - Works well across a wide range of problems
        - Less sensitive to learning rate choice than SGD
        - Fast convergence
        - Handles sparse gradients well (e.g., NLP embeddings)

    LIMITATIONS:
        - Can converge to suboptimal solutions in some cases
        - May generalize slightly worse than well-tuned SGD+Momentum
        - More memory: stores m and v for every parameter

    Parameters
    ----------
    learning_rate : float
        Step size α. Default: 0.001.
    beta1 : float
        Exponential decay for 1st moment. Default: 0.9.
    beta2 : float
        Exponential decay for 2nd moment. Default: 0.999.
    epsilon : float
        Small constant for numerical stability. Default: 1e-8.

    Examples
    --------
    >>> adam = Adam(learning_rate=0.001)
    >>> params = np.array([1.0, 2.0, 3.0])
    >>> for _ in range(100):
    ...     grads = compute_gradients(params)
    ...     params = adam.step(params, grads)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[np.ndarray] = None  # 1st moment (mean)
        self.v: Optional[np.ndarray] = None  # 2nd moment (variance)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Apply one Adam optimization step.

        m_t = β₁ m_{t-1} + (1-β₁) g_t
        v_t = β₂ v_{t-1} + (1-β₂) g_t²
        m̂ = m_t / (1 - β₁ᵗ)
        v̂ = v_t / (1 - β₂ᵗ)
        θ = θ - α × m̂ / (√v̂ + ε)
        """
        params = np.asarray(params, dtype=np.float64)
        grads = np.asarray(grads, dtype=np.float64)
        self.step_count += 1

        # Initialize moments on first step
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_hat = self.v / (1 - self.beta2 ** self.step_count)

        # Parameter update
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self) -> None:
        super().reset()
        self.m = None
        self.v = None


# =============================================================================
# Optimization Benchmark: Minimize a Test Function
# =============================================================================

def minimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    optimizer: BaseOptimizer,
    n_steps: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Minimize a function using a given optimizer.

    This function runs the optimization loop and records the trajectory,
    which is useful for visualization and comparing optimizers.

    Parameters
    ----------
    f : callable
        Objective function f: Rⁿ → R.
    grad_f : callable
        Gradient function ∇f: Rⁿ → Rⁿ.
    x0 : array-like
        Initial parameters.
    optimizer : BaseOptimizer
        The optimizer to use.
    n_steps : int
        Number of optimization steps.
    verbose : bool
        If True, print progress every 10 steps.

    Returns
    -------
    dict
        Results including:
        - final_params: optimized parameters
        - final_loss: final function value
        - trajectory: list of (params, loss) at each step
        - n_steps: number of steps taken

    Examples
    --------
    >>> # Minimize f(x,y) = x² + y² (minimum at origin)
    >>> f = lambda p: p[0]**2 + p[1]**2
    >>> grad_f = lambda p: 2 * p
    >>> result = minimize(f, grad_f, [5.0, 3.0], Adam(lr=0.1))
    >>> np.allclose(result["final_params"], [0, 0], atol=0.01)
    True
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    trajectory = []

    for step in range(n_steps):
        loss = f(x)
        trajectory.append({
            "step": step,
            "params": x.copy(),
            "loss": float(loss),
        })

        if verbose and step % max(1, n_steps // 10) == 0:
            logger.info(f"Step {step:4d}: loss = {loss:.6f}, params = {x}")

        grads = grad_f(x)
        x = optimizer.step(x, grads)

    # Final evaluation
    final_loss = float(f(x))
    trajectory.append({
        "step": n_steps,
        "params": x.copy(),
        "loss": final_loss,
    })

    return {
        "final_params": x,
        "final_loss": final_loss,
        "trajectory": trajectory,
        "n_steps": n_steps,
        "optimizer": type(optimizer).__name__,
    }


def compare_optimizers(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    n_steps: int = 200,
) -> dict[str, dict]:
    """
    Compare all optimizers on the same function.

    This is a key educational exercise: see how different optimizers
    behave on the same problem.

    Parameters
    ----------
    f : callable
        Objective function.
    grad_f : callable
        Gradient function.
    x0 : array-like
        Initial parameters (same for all optimizers).
    n_steps : int
        Number of steps.

    Returns
    -------
    dict
        Mapping of optimizer name → minimize() results.
    """
    optimizers = {
        "GradientDescent(lr=0.01)": GradientDescent(learning_rate=0.01),
        "GradientDescent(lr=0.1)": GradientDescent(learning_rate=0.1),
        "MomentumSGD(lr=0.01)": MomentumSGD(learning_rate=0.01, momentum=0.9),
        "Adam(lr=0.01)": Adam(learning_rate=0.01),
        "Adam(lr=0.001)": Adam(learning_rate=0.001),
    }

    results = {}
    for name, opt in optimizers.items():
        result = minimize(f, grad_f, x0, opt, n_steps=n_steps)
        results[name] = result
        logger.info(
            f"{name:30s} → final_loss = {result['final_loss']:.6f}"
        )

    return results


# =============================================================================
# Verification
# =============================================================================

def verify_implementations(verbose: bool = True) -> dict[str, bool]:
    """
    Verify that all optimizers can minimize simple test functions.

    Test function: f(x,y) = x² + y²
    Known minimum: (0, 0) with f = 0

    Returns
    -------
    dict
        Mapping of optimizer name → whether it converged.
    """
    results = {}
    tol = 0.01

    # Simple quadratic: f(x,y) = x² + y², minimum at (0,0)
    f = lambda p: float(p[0] ** 2 + p[1] ** 2)
    grad_f = lambda p: 2 * np.asarray(p, dtype=np.float64)
    x0 = np.array([5.0, 3.0])

    test_cases = [
        ("GradientDescent", GradientDescent(learning_rate=0.1), 200),
        ("MomentumSGD", MomentumSGD(learning_rate=0.05, momentum=0.9), 200),
        ("Adam", Adam(learning_rate=0.1), 300),
    ]

    for name, opt, steps in test_cases:
        result = minimize(f, grad_f, x0.copy(), opt, n_steps=steps)
        converged = result["final_loss"] < tol
        results[name] = converged
        if verbose:
            logger.info(
                f"{name}: {'✓' if converged else '✗'} "
                f"(final_loss={result['final_loss']:.6f}, "
                f"params={result['final_params']})"
            )

    # --- Rosenbrock (harder test) ---
    # f(x,y) = (1-x)² + 100(y-x²)², minimum at (1,1)
    def rosenbrock(p):
        return float((1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2)

    def grad_rosenbrock(p):
        p = np.asarray(p, dtype=np.float64)
        dx = -2 * (1 - p[0]) + 200 * (p[1] - p[0] ** 2) * (-2 * p[0])
        dy = 200 * (p[1] - p[0] ** 2)
        return np.array([dx, dy])

    x0_rosen = np.array([-1.0, 1.0])
    result = minimize(
        rosenbrock, grad_rosenbrock, x0_rosen,
        Adam(learning_rate=0.005), n_steps=5000,
    )
    converged = result["final_loss"] < 0.1
    results["Adam_Rosenbrock"] = converged
    if verbose:
        logger.info(
            f"Adam on Rosenbrock: {'✓' if converged else '✗'} "
            f"(final_loss={result['final_loss']:.6f}, "
            f"target=(1,1), got={result['final_params']})"
        )

    # Summary
    total = len(results)
    passed_count = sum(results.values())
    if verbose:
        logger.info(f"\nOptimization verification: {passed_count}/{total} passed")

    return results
