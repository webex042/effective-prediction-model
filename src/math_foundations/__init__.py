"""
EPM Math Foundations
====================

Stage 1: Mathematical building blocks for machine learning.

Modules:
    linear_algebra  — Vectors, matrices, eigendecomposition, covariance
    calculus        — Derivatives, gradients, chain rule, gradient checking
    probability     — Distributions, Bayes' theorem, likelihood, sampling
    optimization    — GD, SGD, Momentum, Adam from scratch
"""

from src.math_foundations.linear_algebra import (
    dot_product,
    matrix_multiply,
    transpose,
    norm,
    normalize,
    covariance_matrix,
    eigendecomposition,
)

from src.math_foundations.calculus import (
    numerical_derivative,
    numerical_gradient,
    gradient_check,
)

from src.math_foundations.probability import (
    gaussian_pdf,
    gaussian_log_likelihood,
    bayes_posterior,
)

from src.math_foundations.optimization import (
    GradientDescent,
    SGD,
    MomentumSGD,
    Adam,
)
