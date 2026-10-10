"""
Calculus for Machine Learning — From Scratch
=============================================

Stage 1: Mathematical Foundation

WHY CALCULUS FOR ML?
    Machine learning is fundamentally an optimization problem:
    find parameters θ that minimize a loss function L(θ).

    To minimize L, we need to know which direction to adjust θ.
    The gradient ∇L tells us exactly that — it points toward the
    steepest increase, so we move in the opposite direction.

    Without calculus, you can't train a neural network.

WHAT'S IN THIS MODULE:
    - numerical_derivative: approximate df/dx using finite differences
    - numerical_partial_derivative: ∂f/∂xᵢ for multivariate functions
    - numerical_gradient: full gradient vector [∂f/∂x₁, ..., ∂f/∂xₙ]
    - chain_rule_demo: show how chain rule enables backpropagation
    - gradient_check: verify analytical gradients against numerical ones

USAGE:
    from src.math_foundations.calculus import numerical_derivative, gradient_check

    # Derivative of f(x) = x² at x=3
    df = numerical_derivative(lambda x: x**2, 3.0)  # ≈ 6.0

    # Gradient check: verify your backprop implementation
    passed = gradient_check(analytical_grad, loss_fn, params)
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Derivatives (1D)
# =============================================================================

def numerical_derivative(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-7,
    method: str = "central",
) -> float:
    """
    Compute the numerical derivative df/dx at a point.

    INTUITION:
        The derivative measures the instantaneous rate of change.
        "If I nudge x by a tiny amount, how much does f(x) change?"

        We approximate this by actually nudging x by a small h
        and measuring the change in f.

    MATH:
        Forward difference:  f'(x) ≈ (f(x+h) - f(x)) / h
        Central difference:  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

        Central difference is more accurate because errors cancel out.
        Forward difference: O(h) error
        Central difference: O(h²) error

    ML APPLICATION:
        - Gradient checking: verify backpropagation is correct
        - Understanding what a gradient IS before trusting autograd
        - Debugging custom loss functions

    Parameters
    ----------
    f : callable
        A function f: R → R.
    x : float
        Point at which to compute the derivative.
    h : float
        Step size for finite difference. Too large → inaccurate.
        Too small → floating-point errors. 1e-7 is a good default.
    method : str
        "forward" or "central".

    Returns
    -------
    float
        Approximate derivative df/dx at x.

    Examples
    --------
    >>> numerical_derivative(lambda x: x**2, 3.0)  # d/dx(x²) = 2x = 6
    5.999999...
    >>> numerical_derivative(lambda x: np.sin(x), 0.0)  # cos(0) = 1
    0.999999...
    """
    if method == "forward":
        return (f(x + h) - f(x)) / h
    elif method == "central":
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'forward' or 'central'.")


def numerical_partial_derivative(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    i: int,
    h: float = 1e-7,
) -> float:
    """
    Compute the partial derivative ∂f/∂xᵢ at a point.

    INTUITION:
        A partial derivative asks: "If I change ONLY the i-th variable
        and keep everything else fixed, how does f change?"

        For a 3D landscape, ∂f/∂x tells you the slope in the x-direction,
        and ∂f/∂y tells you the slope in the y-direction.

    MATH:
        ∂f/∂xᵢ ≈ (f(x + hεᵢ) - f(x - hεᵢ)) / (2h)

        where εᵢ is a unit vector in the i-th direction.

    Parameters
    ----------
    f : callable
        A function f: Rⁿ → R.
    x : np.ndarray
        Point at which to compute the partial derivative.
    i : int
        Index of the variable to differentiate with respect to.
    h : float
        Step size.

    Returns
    -------
    float
        Approximate partial derivative ∂f/∂xᵢ at x.
    """
    x = np.asarray(x, dtype=np.float64).copy()

    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[i] += h
    x_minus[i] -= h

    return (f(x_plus) - f(x_minus)) / (2 * h)


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    h: float = 1e-7,
) -> np.ndarray:
    """
    Compute the full gradient vector ∇f at a point.

    INTUITION:
        The gradient is a vector of ALL partial derivatives:
        ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

        It points in the direction of steepest INCREASE of f.
        To minimize f, we move in the OPPOSITE direction: -∇f.
        This is gradient descent!

    MATH:
        ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

        Each component computed via central differences.

    ML APPLICATION:
        This IS the gradient in "gradient descent."
        Every time you call loss.backward() in PyTorch,
        it computes ∇L with respect to all parameters.

    Parameters
    ----------
    f : callable
        A function f: Rⁿ → R.
    x : array-like
        Point at which to compute the gradient.
    h : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Gradient vector, same shape as x.

    Examples
    --------
    >>> # f(x,y) = x² + y²  →  ∇f = [2x, 2y]
    >>> numerical_gradient(lambda p: p[0]**2 + p[1]**2, [3.0, 4.0])
    array([6., 8.])
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        grad[i] = numerical_partial_derivative(f, x, i, h)

    return grad


# =============================================================================
# Chain Rule
# =============================================================================

def chain_rule_demo() -> dict:
    """
    Demonstrate the chain rule — the mathematical foundation of backpropagation.

    INTUITION:
        If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x).

        "The rate of change of a composition equals
         the product of the rates of change of each function."

        Neural networks are LONG chains of composed functions:
        output = f₃(f₂(f₁(x)))

        Backpropagation = chain rule applied from output back to input.

    EXAMPLE:
        Let f(x) = (3x + 2)²

        Decompose as:
            u = g(x) = 3x + 2      (inner function)
            y = f(u) = u²           (outer function)

        Chain rule:
            dy/dx = dy/du × du/dx
                  = 2u    × 3
                  = 2(3x+2) × 3
                  = 6(3x+2)

    Returns
    -------
    dict
        Demonstration results showing analytical vs numerical derivatives.
    """
    # Define the composed function: f(x) = (3x + 2)²
    def inner(x):
        return 3 * x + 2

    def outer(u):
        return u ** 2

    def composed(x):
        return outer(inner(x))

    # Analytical derivative via chain rule: 6(3x + 2)
    def analytical_derivative(x):
        return 6 * (3 * x + 2)

    # Test at x = 1.0
    x_test = 1.0
    analytical = analytical_derivative(x_test)
    numerical = numerical_derivative(composed, x_test)

    result = {
        "function": "f(x) = (3x + 2)²",
        "x": x_test,
        "f(x)": composed(x_test),
        "analytical_derivative": analytical,
        "numerical_derivative": numerical,
        "chain_rule": "dy/dx = dy/du × du/dx = 2u × 3 = 6(3x+2)",
        "match": abs(analytical - numerical) < 1e-5,
    }

    logger.info(
        f"Chain rule demo: f'({x_test}) = {analytical:.4f} "
        f"(numerical: {numerical:.6f})"
    )

    return result


# =============================================================================
# Gradient Checking
# =============================================================================

def gradient_check(
    analytical_grad_fn: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray], float],
    params: ArrayLike,
    h: float = 1e-7,
    tolerance: float = 1e-5,
) -> dict:
    """
    Verify analytical gradients against numerical gradients.

    INTUITION:
        When you implement backpropagation, you derive gradients by hand
        and code them up. But how do you know you didn't make a mistake?

        Gradient checking computes the "ground truth" gradient numerically
        (slow but reliable) and compares it against your analytical gradient
        (fast but error-prone). If they agree, your backprop is correct.

    MATH:
        For each parameter θᵢ:
            numerical:   ∂L/∂θᵢ ≈ (L(θ + hεᵢ) - L(θ - hεᵢ)) / (2h)
            analytical:  from your backprop implementation

        Relative error:
            error = ||grad_analytical - grad_numerical|| / (||grad_analytical|| + ||grad_numerical|| + ε)

        If error < 1e-5: probably correct
        If error > 1e-3: probably a bug

    ML APPLICATION:
        - Debugging custom neural network implementations
        - Verifying loss function gradients
        - Stage 3: we'll use this to verify our from-scratch backprop

    Parameters
    ----------
    analytical_grad_fn : callable
        Function that returns the analytical gradient given parameters.
    loss_fn : callable
        Loss function that takes parameters and returns a scalar.
    params : array-like
        Parameter values at which to check gradients.
    h : float
        Step size for numerical gradient.
    tolerance : float
        Maximum relative error to consider "passing."

    Returns
    -------
    dict
        Results including gradients, relative error, and pass/fail.

    Examples
    --------
    >>> # Check gradient of f(x) = x₁² + x₂²
    >>> loss_fn = lambda x: x[0]**2 + x[1]**2
    >>> grad_fn = lambda x: 2 * x  # analytical: [2x₁, 2x₂]
    >>> result = gradient_check(grad_fn, loss_fn, [3.0, 4.0])
    >>> result["passed"]
    True
    """
    params = np.asarray(params, dtype=np.float64)

    # Compute both gradients
    grad_analytical = np.asarray(analytical_grad_fn(params), dtype=np.float64)
    grad_numerical = numerical_gradient(loss_fn, params, h=h)

    # Relative error (safe against division by zero)
    diff = np.linalg.norm(grad_analytical - grad_numerical)
    denom = (
        np.linalg.norm(grad_analytical)
        + np.linalg.norm(grad_numerical)
        + 1e-15
    )
    relative_error = diff / denom

    passed = relative_error < tolerance

    result = {
        "grad_analytical": grad_analytical,
        "grad_numerical": grad_numerical,
        "relative_error": float(relative_error),
        "tolerance": tolerance,
        "passed": passed,
    }

    if passed:
        logger.info(f"Gradient check PASSED (relative error: {relative_error:.2e})")
    else:
        logger.warning(
            f"Gradient check FAILED (relative error: {relative_error:.2e}, "
            f"tolerance: {tolerance:.2e})"
        )

    return result


# =============================================================================
# Verification
# =============================================================================

def verify_implementations(verbose: bool = True) -> dict[str, bool]:
    """
    Verify calculus implementations against known analytical solutions.

    Returns
    -------
    dict
        Mapping of test name → pass/fail.
    """
    results = {}
    tol = 1e-5

    # --- d/dx(x²) = 2x ---
    f = lambda x: x ** 2
    for x_val in [0.0, 1.0, 3.0, -2.5]:
        deriv = numerical_derivative(f, x_val)
        expected = 2 * x_val
        passed = abs(deriv - expected) < tol
        results[f"d(x²)/dx at x={x_val}"] = passed
        if verbose:
            logger.info(
                f"d(x²)/dx at x={x_val}: {'✓' if passed else '✗'} "
                f"(got {deriv:.6f}, expected {expected:.6f})"
            )

    # --- d/dx(sin(x)) = cos(x) ---
    for x_val in [0.0, np.pi / 4, np.pi / 2, np.pi]:
        deriv = numerical_derivative(np.sin, x_val)
        expected = np.cos(x_val)
        passed = abs(deriv - expected) < tol
        results[f"d(sin)/dx at x={x_val:.2f}"] = passed
        if verbose:
            logger.info(
                f"d(sin)/dx at x={x_val:.2f}: {'✓' if passed else '✗'} "
                f"(got {deriv:.6f}, expected {expected:.6f})"
            )

    # --- Gradient of f(x,y) = x² + y² should be [2x, 2y] ---
    f_2d = lambda p: p[0] ** 2 + p[1] ** 2
    point = np.array([3.0, 4.0])
    grad = numerical_gradient(f_2d, point)
    expected = np.array([6.0, 8.0])
    passed = np.allclose(grad, expected, atol=tol)
    results["gradient of x²+y²"] = passed
    if verbose:
        logger.info(
            f"∇(x²+y²) at (3,4): {'✓' if passed else '✗'} "
            f"(got {grad}, expected {expected})"
        )

    # --- Chain rule ---
    cr = chain_rule_demo()
    results["chain_rule"] = cr["match"]
    if verbose:
        logger.info(f"chain rule: {'✓' if cr['match'] else '✗'}")

    # --- Gradient check ---
    gc = gradient_check(
        analytical_grad_fn=lambda x: 2 * x,
        loss_fn=lambda x: float(x[0] ** 2 + x[1] ** 2),
        params=[3.0, 4.0],
    )
    results["gradient_check"] = gc["passed"]

    # Summary
    total = len(results)
    passed_count = sum(results.values())
    if verbose:
        logger.info(f"\nCalculus verification: {passed_count}/{total} passed")

    return results
