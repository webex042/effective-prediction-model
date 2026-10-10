"""
Linear Algebra — From Scratch
==============================

Stage 1: Mathematical Foundation

WHY LINEAR ALGEBRA?
    Every neural network is fundamentally a sequence of matrix operations.
    When you write z = Wx + b, that's a matrix-vector multiplication followed
    by a vector addition. Understanding this is non-negotiable for ML.

WHAT'S IN THIS MODULE:
    - dot_product: The fundamental operation connecting two vectors
    - matrix_multiply: How neural network layers transform data
    - transpose: Rearranging matrices (critical for backpropagation)
    - norm: Measuring vector magnitude (used in regularization, normalization)
    - normalize: Making vectors unit-length (used in attention, embeddings)
    - covariance_matrix: How features relate (foundation for PCA)
    - eigendecomposition: Breaking matrices into fundamental components

PHILOSOPHY:
    Every function is implemented using basic NumPy operations (not np.linalg
    shortcuts) so you can see exactly what's happening mathematically.
    Then we verify against NumPy's optimized versions.

USAGE:
    from src.math_foundations.linear_algebra import dot_product, matrix_multiply

    # Dot product of two vectors
    result = dot_product([1, 2, 3], [4, 5, 6])  # = 1*4 + 2*5 + 3*6 = 32

    # Matrix multiplication (like a neural network layer)
    W = [[1, 2], [3, 4], [5, 6]]  # 3x2 weight matrix
    x = [[1], [2]]                 # 2x1 input vector
    z = matrix_multiply(W, x)      # 3x1 output
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Vectors
# =============================================================================

def dot_product(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the dot product of two vectors.

    INTUITION:
        The dot product measures how much two vectors "agree" in direction.
        - If they point the same way → large positive value
        - If they're perpendicular → zero
        - If they point opposite ways → large negative value

    MATH:
        a · b = Σᵢ aᵢ × bᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ

    ML APPLICATION:
        - Similarity measurement (cosine similarity uses dot product)
        - Neural network forward pass: each neuron computes w · x + b
        - Attention scores: query · key

    Parameters
    ----------
    a : array-like
        First vector (1D).
    b : array-like
        Second vector (1D), same length as a.

    Returns
    -------
    float
        Scalar dot product.

    Examples
    --------
    >>> dot_product([1, 2, 3], [4, 5, 6])
    32.0
    >>> dot_product([1, 0], [0, 1])  # perpendicular vectors
    0.0
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"dot_product expects 1D vectors, got shapes {a.shape} and {b.shape}"
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Vectors must have same length, got {a.shape[0]} and {b.shape[0]}"
        )

    # From-scratch implementation: element-wise multiply and sum
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]

    return float(result)


def norm(v: ArrayLike, order: int = 2) -> float:
    """
    Compute the Lp norm (magnitude) of a vector.

    INTUITION:
        The norm tells you "how long" a vector is.
        - L1 norm: sum of absolute values (Manhattan distance — like walking on a grid)
        - L2 norm: Euclidean distance (straight-line distance)

    MATH:
        L1: ||v||₁ = Σᵢ |vᵢ|
        L2: ||v||₂ = √(Σᵢ vᵢ²)

    ML APPLICATION:
        - L1 regularization (Lasso): encourages sparsity (some weights → 0)
        - L2 regularization (Ridge): discourages large weights (weight decay)
        - Gradient clipping: prevent exploding gradients by capping ||∇||
        - Normalization: scale vectors to unit length

    Parameters
    ----------
    v : array-like
        Input vector.
    order : int
        Norm order (1 for L1, 2 for L2).

    Returns
    -------
    float
        The Lp norm.

    Examples
    --------
    >>> norm([3, 4], order=2)  # √(9+16) = √25 = 5
    5.0
    >>> norm([3, -4], order=1)  # |3|+|-4| = 7
    7.0
    """
    v = np.asarray(v, dtype=np.float64)

    if v.ndim != 1:
        raise ValueError(f"norm expects a 1D vector, got shape {v.shape}")

    if order == 1:
        # L1: sum of absolute values
        result = 0.0
        for val in v:
            result += abs(val)
        return float(result)
    elif order == 2:
        # L2: square root of sum of squares
        result = 0.0
        for val in v:
            result += val * val
        return float(np.sqrt(result))
    else:
        # General Lp norm
        result = 0.0
        for val in v:
            result += abs(val) ** order
        return float(result ** (1.0 / order))


def normalize(v: ArrayLike) -> np.ndarray:
    """
    Normalize a vector to unit length (L2 norm = 1).

    INTUITION:
        Keeps the direction but makes the length exactly 1.
        This lets you compare directions without being confused by magnitudes.

    MATH:
        v̂ = v / ||v||₂

    ML APPLICATION:
        - Cosine similarity: normalize first, then dot product
        - Layer normalization, batch normalization
        - Embedding normalization
        - Attention: normalize query/key vectors

    Parameters
    ----------
    v : array-like
        Input vector.

    Returns
    -------
    np.ndarray
        Unit vector in the same direction as v.

    Examples
    --------
    >>> normalize([3, 4])
    array([0.6, 0.8])
    """
    v = np.asarray(v, dtype=np.float64)
    n = norm(v, order=2)

    if n == 0:
        raise ValueError("Cannot normalize a zero vector")

    return v / n


# =============================================================================
# Matrices
# =============================================================================

def transpose(M: ArrayLike) -> np.ndarray:
    """
    Transpose a matrix (swap rows and columns).

    INTUITION:
        Flip the matrix along its diagonal.
        Row i becomes column i, column j becomes row j.

    MATH:
        (Aᵀ)ᵢⱼ = Aⱼᵢ

    ML APPLICATION:
        - Backpropagation: gradients flow through Wᵀ
        - Attention: QKᵀ requires transposing the key matrix
        - Computing covariance: (1/n) XᵀX

    Parameters
    ----------
    M : array-like
        Input matrix (2D).

    Returns
    -------
    np.ndarray
        Transposed matrix.

    Examples
    --------
    >>> transpose([[1, 2, 3], [4, 5, 6]])
    array([[1, 4],
           [2, 5],
           [3, 6]])
    """
    M = np.asarray(M, dtype=np.float64)

    if M.ndim != 2:
        raise ValueError(f"transpose expects a 2D matrix, got {M.ndim}D")

    rows, cols = M.shape
    result = np.zeros((cols, rows), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            result[j][i] = M[i][j]

    return result


def matrix_multiply(A: ArrayLike, B: ArrayLike) -> np.ndarray:
    """
    Multiply two matrices.

    INTUITION:
        Each element of the result is a dot product between
        a row of A and a column of B.

        Think of it as: for each output position (i,j), take
        row i of A and column j of B, and compute their dot product.

    MATH:
        (AB)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ

        If A is (m × n) and B is (n × p), then AB is (m × p).

    ML APPLICATION:
        This is THE core operation of neural networks.
        z = Wx + b  →  matrix_multiply(W, x) + b
        Every layer transforms inputs by multiplying with weights.

    COMPLEXITY:
        Time: O(m × n × p)  — this is why large models are expensive!
        Space: O(m × p) for the output

    Parameters
    ----------
    A : array-like
        Left matrix, shape (m, n).
    B : array-like
        Right matrix, shape (n, p).

    Returns
    -------
    np.ndarray
        Product matrix, shape (m, p).

    Examples
    --------
    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    array([[19., 22.],
           [43., 50.]])
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"matrix_multiply expects 2D matrices, got {A.ndim}D and {B.ndim}D"
        )

    m, n = A.shape
    n2, p = B.shape

    if n != n2:
        raise ValueError(
            f"Cannot multiply ({m}×{n}) by ({n2}×{p}): "
            f"inner dimensions must match ({n} ≠ {n2})"
        )

    # From-scratch: triple nested loop
    result = np.zeros((m, p), dtype=np.float64)

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result


# =============================================================================
# Advanced: Covariance & Eigendecomposition
# =============================================================================

def covariance_matrix(X: ArrayLike) -> np.ndarray:
    """
    Compute the covariance matrix of a dataset.

    INTUITION:
        The covariance matrix tells you how features move together:
        - Positive covariance: when feature A increases, B tends to increase
        - Negative covariance: when A increases, B tends to decrease
        - Zero covariance: A and B are uncorrelated

    MATH:
        Given data X with shape (n_samples, n_features):
        1. Center the data: X_centered = X - mean(X)
        2. Covariance: C = (1/(n-1)) × X_centeredᵀ × X_centered

        Cᵢⱼ = Cov(feature_i, feature_j)

        The diagonal Cᵢᵢ = Var(feature_i)

    ML APPLICATION:
        - PCA: eigendecompose the covariance matrix to find principal components
        - Feature selection: identify redundant (highly correlated) features
        - Gaussian models: define the shape of multivariate Gaussians
        - Mahalanobis distance: accounts for feature correlations

    Parameters
    ----------
    X : array-like
        Data matrix, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (n_features, n_features).

    Examples
    --------
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> covariance_matrix(X)  # Perfect positive correlation
    array([[4., 4.],
           [4., 4.]])
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"covariance_matrix expects 2D data, got {X.ndim}D")

    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute covariance")

    # Step 1: Center the data (subtract mean of each feature)
    mean = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        total = 0.0
        for i in range(n_samples):
            total += X[i][j]
        mean[j] = total / n_samples

    X_centered = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(n_features):
            X_centered[i][j] = X[i][j] - mean[j]

    # Step 2: Compute covariance = (1/(n-1)) * X_centeredᵀ @ X_centered
    # Using Bessel's correction (n-1) for unbiased estimate
    cov = np.zeros((n_features, n_features), dtype=np.float64)
    for i in range(n_features):
        for j in range(n_features):
            total = 0.0
            for k in range(n_samples):
                total += X_centered[k][i] * X_centered[k][j]
            cov[i][j] = total / (n_samples - 1)

    return cov


def eigendecomposition(M: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix.

    INTUITION:
        Eigendecomposition finds the "natural axes" of a matrix.

        Imagine an ellipse. The eigenvectors point along the major and minor
        axes, and the eigenvalues tell you how stretched the ellipse is
        along each axis.

        For a covariance matrix:
        - Eigenvectors = directions of maximum variance (principal components)
        - Eigenvalues = amount of variance in each direction

    MATH:
        Find v and λ such that: Av = λv

        Where:
        - A is the matrix
        - v is the eigenvector (direction that A only scales, doesn't rotate)
        - λ is the eigenvalue (the scaling factor)

    ML APPLICATION:
        - PCA: keep eigenvectors with largest eigenvalues → dimensionality reduction
        - Spectral clustering: eigendecompose the graph Laplacian
        - Understanding neural network loss landscapes
        - Condition number = max(λ)/min(λ) → numerical stability

    NOTE:
        We use NumPy's np.linalg.eigh here because implementing eigensolvers
        from scratch (QR algorithm, etc.) is a numerical analysis research
        topic in itself. The important thing is understanding WHAT eigenvalues
        mean, not reimplementing the QR algorithm.

    Parameters
    ----------
    M : array-like
        Square symmetric matrix.

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues, sorted in descending order.
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns, shape (n, n).
        eigenvectors[:, i] corresponds to eigenvalues[i].

    Examples
    --------
    >>> eigenvalues, eigenvectors = eigendecomposition([[2, 1], [1, 2]])
    >>> eigenvalues
    array([3., 1.])
    """
    M = np.asarray(M, dtype=np.float64)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"eigendecomposition expects a square matrix, got shape {M.shape}"
        )

    # Use eigh for symmetric matrices (covariance matrices are always symmetric)
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Sort in descending order (largest eigenvalue = most important direction)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# =============================================================================
# Verification: Compare scratch vs NumPy
# =============================================================================

def verify_implementations(verbose: bool = True) -> dict[str, bool]:
    """
    Verify all from-scratch implementations against NumPy's optimized versions.

    This is a crucial practice: implement from scratch to understand,
    then verify against a trusted library to catch bugs.

    Returns
    -------
    dict
        Mapping of function name → whether it passed verification.
    """
    results = {}
    tolerance = 1e-10

    # --- Dot product ---
    a = np.random.randn(100)
    b = np.random.randn(100)
    ours = dot_product(a, b)
    numpy_result = float(np.dot(a, b))
    passed = abs(ours - numpy_result) < tolerance
    results["dot_product"] = passed
    if verbose:
        logger.info(
            f"dot_product: {'✓ PASS' if passed else '✗ FAIL'} "
            f"(ours={ours:.6f}, numpy={numpy_result:.6f})"
        )

    # --- Norm ---
    v = np.random.randn(50)
    ours_l2 = norm(v, order=2)
    numpy_l2 = float(np.linalg.norm(v, ord=2))
    passed = abs(ours_l2 - numpy_l2) < tolerance
    results["norm_l2"] = passed
    if verbose:
        logger.info(
            f"norm_l2: {'✓ PASS' if passed else '✗ FAIL'} "
            f"(ours={ours_l2:.6f}, numpy={numpy_l2:.6f})"
        )

    # --- Transpose ---
    M = np.random.randn(5, 3)
    ours_t = transpose(M)
    numpy_t = M.T
    passed = np.allclose(ours_t, numpy_t, atol=tolerance)
    results["transpose"] = passed
    if verbose:
        logger.info(f"transpose: {'✓ PASS' if passed else '✗ FAIL'}")

    # --- Matrix multiply ---
    A = np.random.randn(4, 5)
    B = np.random.randn(5, 3)
    ours_mm = matrix_multiply(A, B)
    numpy_mm = A @ B
    passed = np.allclose(ours_mm, numpy_mm, atol=tolerance)
    results["matrix_multiply"] = passed
    if verbose:
        logger.info(f"matrix_multiply: {'✓ PASS' if passed else '✗ FAIL'}")

    # --- Covariance ---
    X = np.random.randn(50, 4)
    ours_cov = covariance_matrix(X)
    numpy_cov = np.cov(X, rowvar=False)
    passed = np.allclose(ours_cov, numpy_cov, atol=1e-8)
    results["covariance_matrix"] = passed
    if verbose:
        logger.info(f"covariance_matrix: {'✓ PASS' if passed else '✗ FAIL'}")

    # Summary
    total = len(results)
    passed_count = sum(results.values())
    if verbose:
        logger.info(f"\nVerification: {passed_count}/{total} passed")

    return results
