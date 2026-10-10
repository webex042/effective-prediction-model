"""
Tests for Stage 1 — Mathematical Foundation
=============================================

Tests for all modules in src/math_foundations/:
    - linear_algebra
    - calculus
    - probability
    - optimization

Every test verifies our from-scratch implementations against
known mathematical results or NumPy's optimized versions.
"""

import numpy as np
import pytest

from src.math_foundations.linear_algebra import (
    dot_product,
    matrix_multiply,
    transpose,
    norm,
    normalize,
    covariance_matrix,
    eigendecomposition,
    verify_implementations as verify_linalg,
)
from src.math_foundations.calculus import (
    numerical_derivative,
    numerical_partial_derivative,
    numerical_gradient,
    chain_rule_demo,
    gradient_check,
    verify_implementations as verify_calc,
)
from src.math_foundations.probability import (
    gaussian_pdf,
    gaussian_log_pdf,
    gaussian_log_likelihood,
    bayes_posterior,
    sample_gaussian,
    kl_divergence_gaussians,
    entropy_discrete,
    mle_gaussian,
    verify_implementations as verify_prob,
)
from src.math_foundations.optimization import (
    GradientDescent,
    SGD,
    MomentumSGD,
    Adam,
    minimize,
    verify_implementations as verify_opt,
)


# =============================================================================
# Linear Algebra Tests
# =============================================================================

class TestDotProduct:
    """Tests for dot_product()."""

    def test_basic(self):
        assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0

    def test_perpendicular_vectors(self):
        """Perpendicular vectors should have dot product = 0."""
        assert dot_product([1, 0], [0, 1]) == 0.0

    def test_parallel_vectors(self):
        """Parallel vectors: dot = product of norms."""
        assert dot_product([2, 0], [3, 0]) == 6.0

    def test_negative(self):
        """Opposite direction → negative dot product."""
        assert dot_product([1, 0], [-1, 0]) == -1.0

    def test_matches_numpy(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        assert abs(dot_product(a, b) - float(np.dot(a, b))) < 1e-10

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            dot_product([1, 2], [1, 2, 3])

    def test_non_1d(self):
        with pytest.raises(ValueError, match="1D"):
            dot_product([[1, 2]], [1, 2])


class TestNorm:
    """Tests for norm()."""

    def test_l2_basic(self):
        assert abs(norm([3, 4], order=2) - 5.0) < 1e-10

    def test_l1_basic(self):
        assert abs(norm([3, -4], order=1) - 7.0) < 1e-10

    def test_unit_vector(self):
        assert abs(norm([1, 0, 0], order=2) - 1.0) < 1e-10

    def test_zero_vector(self):
        assert norm([0, 0, 0], order=2) == 0.0

    def test_matches_numpy(self):
        v = np.random.randn(50)
        assert abs(norm(v, 2) - np.linalg.norm(v)) < 1e-10


class TestNormalize:
    """Tests for normalize()."""

    def test_basic(self):
        result = normalize([3, 4])
        assert abs(norm(result) - 1.0) < 1e-10

    def test_preserves_direction(self):
        v = [3.0, 4.0]
        result = normalize(v)
        np.testing.assert_allclose(result, [0.6, 0.8], atol=1e-10)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            normalize([0, 0, 0])


class TestTranspose:
    """Tests for transpose()."""

    def test_basic(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = transpose(M)
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_square(self):
        M = [[1, 2], [3, 4]]
        result = transpose(M)
        expected = np.array([[1, 3], [2, 4]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_double_transpose(self):
        """Transposing twice should give back the original."""
        M = np.random.randn(3, 5)
        np.testing.assert_allclose(transpose(transpose(M)), M, atol=1e-10)

    def test_matches_numpy(self):
        M = np.random.randn(4, 7)
        np.testing.assert_allclose(transpose(M), M.T, atol=1e-10)


class TestMatrixMultiply:
    """Tests for matrix_multiply()."""

    def test_basic(self):
        result = matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_non_square(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 2)
        result = matrix_multiply(A, B)
        np.testing.assert_allclose(result, A @ B, atol=1e-10)

    def test_identity(self):
        """A × I = A"""
        A = np.random.randn(3, 3)
        I = np.eye(3)
        np.testing.assert_allclose(matrix_multiply(A, I), A, atol=1e-10)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="inner dimensions"):
            matrix_multiply([[1, 2]], [[1, 2]])  # (1,2) × (1,2) invalid


class TestCovarianceMatrix:
    """Tests for covariance_matrix()."""

    def test_perfect_correlation(self):
        """Perfectly correlated features: y = 2x."""
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=float)
        cov = covariance_matrix(X)
        # Off-diagonal should equal product of stds
        assert cov[0, 1] > 0  # Positive correlation

    def test_matches_numpy(self):
        X = np.random.randn(100, 5)
        ours = covariance_matrix(X)
        numpy_cov = np.cov(X, rowvar=False)
        np.testing.assert_allclose(ours, numpy_cov, atol=1e-8)

    def test_symmetric(self):
        X = np.random.randn(50, 4)
        cov = covariance_matrix(X)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_diagonal_is_variance(self):
        X = np.random.randn(100, 3)
        cov = covariance_matrix(X)
        for j in range(3):
            var = np.var(X[:, j], ddof=1)
            assert abs(cov[j, j] - var) < 1e-8


class TestEigendecomposition:
    """Tests for eigendecomposition()."""

    def test_known_eigenvalues(self):
        """2x2 identity matrix: eigenvalues should be [1, 1]."""
        vals, vecs = eigendecomposition(np.eye(2))
        np.testing.assert_allclose(vals, [1.0, 1.0], atol=1e-10)

    def test_av_equals_lambda_v(self):
        """Verify Av = λv for each eigenpair."""
        M = np.array([[2, 1], [1, 2]], dtype=float)
        vals, vecs = eigendecomposition(M)
        for i in range(len(vals)):
            Av = M @ vecs[:, i]
            lambda_v = vals[i] * vecs[:, i]
            np.testing.assert_allclose(Av, lambda_v, atol=1e-10)

    def test_sorted_descending(self):
        M = np.array([[5, 1, 0], [1, 3, 1], [0, 1, 1]], dtype=float)
        vals, _ = eigendecomposition(M)
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


class TestLinalgVerification:
    """Run the full verification suite."""

    def test_all_pass(self):
        results = verify_linalg(verbose=False)
        for name, passed in results.items():
            assert passed, f"Linear algebra verification failed: {name}"


# =============================================================================
# Calculus Tests
# =============================================================================

class TestNumericalDerivative:
    """Tests for numerical_derivative()."""

    def test_x_squared(self):
        """d/dx(x²) = 2x."""
        assert abs(numerical_derivative(lambda x: x ** 2, 3.0) - 6.0) < 1e-5

    def test_sin(self):
        """d/dx(sin(x)) = cos(x)."""
        for x in [0.0, np.pi / 4, np.pi]:
            expected = np.cos(x)
            assert abs(numerical_derivative(np.sin, x) - expected) < 1e-5

    def test_exp(self):
        """d/dx(eˣ) = eˣ."""
        assert abs(numerical_derivative(np.exp, 1.0) - np.e) < 1e-5

    def test_constant(self):
        """d/dx(5) = 0."""
        assert abs(numerical_derivative(lambda x: 5.0, 2.0)) < 1e-5

    def test_forward_vs_central(self):
        """Central should be more accurate than forward difference."""
        f = lambda x: x ** 3
        x = 2.0
        exact = 12.0  # 3x² at x=2
        error_forward = abs(numerical_derivative(f, x, method="forward") - exact)
        error_central = abs(numerical_derivative(f, x, method="central") - exact)
        assert error_central < error_forward


class TestNumericalGradient:
    """Tests for numerical_gradient()."""

    def test_quadratic(self):
        """∇(x² + y²) = [2x, 2y]."""
        f = lambda p: p[0] ** 2 + p[1] ** 2
        grad = numerical_gradient(f, [3.0, 4.0])
        np.testing.assert_allclose(grad, [6.0, 8.0], atol=1e-5)

    def test_linear(self):
        """∇(3x + 2y + 1) = [3, 2]."""
        f = lambda p: 3 * p[0] + 2 * p[1] + 1
        grad = numerical_gradient(f, [0.0, 0.0])
        np.testing.assert_allclose(grad, [3.0, 2.0], atol=1e-5)


class TestChainRule:
    """Tests for chain_rule_demo()."""

    def test_demo_passes(self):
        result = chain_rule_demo()
        assert result["match"]


class TestGradientCheck:
    """Tests for gradient_check()."""

    def test_correct_gradient(self):
        """Correct gradient should pass."""
        result = gradient_check(
            analytical_grad_fn=lambda x: 2 * x,
            loss_fn=lambda x: float(np.sum(x ** 2)),
            params=[3.0, 4.0],
        )
        assert result["passed"]

    def test_wrong_gradient_fails(self):
        """Incorrect gradient should fail."""
        result = gradient_check(
            analytical_grad_fn=lambda x: 3 * x,  # Wrong! Should be 2x
            loss_fn=lambda x: float(np.sum(x ** 2)),
            params=[3.0, 4.0],
        )
        assert not result["passed"]


class TestCalculusVerification:
    """Run the full verification suite."""

    def test_all_pass(self):
        results = verify_calc(verbose=False)
        for name, passed in results.items():
            assert passed, f"Calculus verification failed: {name}"


# =============================================================================
# Probability Tests
# =============================================================================

class TestGaussianPDF:
    """Tests for gaussian_pdf()."""

    def test_peak_of_standard_normal(self):
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(gaussian_pdf(0.0) - expected) < 1e-10

    def test_symmetry(self):
        assert abs(gaussian_pdf(-1.0) - gaussian_pdf(1.0)) < 1e-10

    def test_wider_distribution(self):
        """Larger std → lower peak."""
        assert gaussian_pdf(0.0, std=2.0) < gaussian_pdf(0.0, std=1.0)

    def test_shifted_mean(self):
        """Peak should be at the mean."""
        assert gaussian_pdf(5.0, mean=5.0) > gaussian_pdf(4.0, mean=5.0)

    def test_array_input(self):
        result = gaussian_pdf([-1.0, 0.0, 1.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_negative_std_raises(self):
        with pytest.raises(ValueError, match="positive"):
            gaussian_pdf(0.0, std=-1.0)


class TestGaussianLogPDF:
    """Tests for gaussian_log_pdf()."""

    def test_consistency_with_pdf(self):
        """log(pdf(x)) should equal log_pdf(x)."""
        x = 1.5
        log_p = gaussian_log_pdf(x)
        p = gaussian_pdf(x)
        assert abs(log_p - np.log(p)) < 1e-10


class TestGaussianLogLikelihood:
    """Tests for gaussian_log_likelihood()."""

    def test_better_model_higher_likelihood(self):
        """Data near mean should have higher likelihood."""
        data = [0.1, -0.1, 0.05, -0.05]
        ll_good = gaussian_log_likelihood(data, mean=0.0, std=0.5)
        ll_bad = gaussian_log_likelihood(data, mean=5.0, std=0.5)
        assert ll_good > ll_bad


class TestBayesPosterior:
    """Tests for bayes_posterior()."""

    def test_medical_test(self):
        """Classic medical test example."""
        post = bayes_posterior(prior=0.01, likelihood=0.95, evidence=0.05)
        assert abs(post - 0.19) < 1e-10

    def test_high_prior(self):
        """High prior → higher posterior."""
        post_low = bayes_posterior(prior=0.01, likelihood=0.9, evidence=0.1)
        post_high = bayes_posterior(prior=0.5, likelihood=0.9, evidence=0.5)
        assert post_high > post_low

    def test_invalid_prior(self):
        with pytest.raises(ValueError, match="Prior"):
            bayes_posterior(prior=1.5, likelihood=0.5, evidence=0.5)


class TestKLDivergence:
    """Tests for kl_divergence_gaussians()."""

    def test_same_distribution(self):
        assert abs(kl_divergence_gaussians(0, 1, 0, 1)) < 1e-10

    def test_non_negative(self):
        """KL divergence is always ≥ 0."""
        for _ in range(20):
            mu1, s1 = np.random.randn(), abs(np.random.randn()) + 0.1
            mu2, s2 = np.random.randn(), abs(np.random.randn()) + 0.1
            assert kl_divergence_gaussians(mu1, s1, mu2, s2) >= -1e-10

    def test_known_value(self):
        """KL(N(1,1) || N(0,1)) = 0.5."""
        assert abs(kl_divergence_gaussians(1.0, 1.0, 0.0, 1.0) - 0.5) < 1e-10


class TestEntropy:
    """Tests for entropy_discrete()."""

    def test_fair_coin(self):
        assert abs(entropy_discrete([0.5, 0.5]) - 1.0) < 1e-10

    def test_certain_outcome(self):
        assert abs(entropy_discrete([1.0, 0.0])) < 1e-10

    def test_uniform_maximum(self):
        """Uniform distribution has maximum entropy."""
        n = 4
        uniform = [1.0 / n] * n
        biased = [0.7, 0.1, 0.1, 0.1]
        assert entropy_discrete(uniform) > entropy_discrete(biased)

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="sum to 1"):
            entropy_discrete([0.3, 0.3])


class TestMLEGaussian:
    """Tests for mle_gaussian()."""

    def test_recovers_parameters(self):
        np.random.seed(42)
        data = np.random.normal(5.0, 2.0, size=10000)
        mle = mle_gaussian(data)
        assert abs(mle["mean"] - 5.0) < 0.1
        assert abs(mle["std_mle"] - 2.0) < 0.1

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            mle_gaussian([1.0])


class TestSampleGaussian:
    """Tests for sample_gaussian()."""

    def test_mean_and_std(self):
        """Large sample should approximate the true parameters."""
        samples = sample_gaussian(mean=3.0, std=2.0, size=50000, seed=42)
        assert abs(np.mean(samples) - 3.0) < 0.1
        assert abs(np.std(samples) - 2.0) < 0.1

    def test_reproducibility(self):
        s1 = sample_gaussian(size=10, seed=42)
        s2 = sample_gaussian(size=10, seed=42)
        np.testing.assert_array_equal(s1, s2)


class TestProbVerification:
    """Run the full verification suite."""

    def test_all_pass(self):
        results = verify_prob(verbose=False)
        for name, passed in results.items():
            assert passed, f"Probability verification failed: {name}"


# =============================================================================
# Optimization Tests
# =============================================================================

class TestGradientDescentOptimizer:
    """Tests for GradientDescent."""

    def test_single_step(self):
        gd = GradientDescent(learning_rate=0.1)
        params = np.array([5.0, 3.0])
        grads = np.array([10.0, 6.0])
        result = gd.step(params, grads)
        np.testing.assert_allclose(result, [4.0, 2.4])

    def test_converges_quadratic(self):
        """Should minimize x² + y² to near zero."""
        gd = GradientDescent(learning_rate=0.1)
        f = lambda p: float(p[0] ** 2 + p[1] ** 2)
        grad_f = lambda p: 2 * np.asarray(p, dtype=np.float64)
        result = minimize(f, grad_f, [5.0, 3.0], gd, n_steps=200)
        assert result["final_loss"] < 0.01

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="positive"):
            GradientDescent(learning_rate=-0.1)


class TestMomentumSGDOptimizer:
    """Tests for MomentumSGD."""

    def test_builds_velocity(self):
        """Repeated same gradient should accelerate."""
        opt = MomentumSGD(learning_rate=0.1, momentum=0.9)
        params = np.array([10.0])
        grads = np.array([1.0])

        # First step: velocity = 1.0, update = -0.1
        p1 = opt.step(params, grads)
        # Second step: velocity = 0.9 + 1.0 = 1.9, update = -0.19
        p2 = opt.step(p1, grads)

        step1 = params[0] - p1[0]
        step2 = p1[0] - p2[0]
        assert step2 > step1  # Momentum causes larger second step

    def test_converges(self):
        opt = MomentumSGD(learning_rate=0.05, momentum=0.9)
        f = lambda p: float(p[0] ** 2 + p[1] ** 2)
        grad_f = lambda p: 2 * np.asarray(p, dtype=np.float64)
        result = minimize(f, grad_f, [5.0, 3.0], opt, n_steps=200)
        assert result["final_loss"] < 0.01


class TestAdamOptimizer:
    """Tests for Adam."""

    def test_converges_quadratic(self):
        adam = Adam(learning_rate=0.1)
        f = lambda p: float(p[0] ** 2 + p[1] ** 2)
        grad_f = lambda p: 2 * np.asarray(p, dtype=np.float64)
        result = minimize(f, grad_f, [5.0, 3.0], adam, n_steps=300)
        assert result["final_loss"] < 0.01

    def test_converges_rosenbrock(self):
        """Adam should make progress on the Rosenbrock function."""
        def rosenbrock(p):
            return float((1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2)

        def grad_rosenbrock(p):
            p = np.asarray(p, dtype=np.float64)
            dx = -2 * (1 - p[0]) + 200 * (p[1] - p[0] ** 2) * (-2 * p[0])
            dy = 200 * (p[1] - p[0] ** 2)
            return np.array([dx, dy])

        adam = Adam(learning_rate=0.005)
        result = minimize(
            rosenbrock, grad_rosenbrock, [-1.0, 1.0], adam, n_steps=5000
        )
        assert result["final_loss"] < 0.1

    def test_reset(self):
        adam = Adam()
        p = np.array([1.0, 2.0])
        g = np.array([0.1, 0.2])
        adam.step(p, g)
        assert adam.m is not None
        adam.reset()
        assert adam.m is None
        assert adam.step_count == 0


class TestOptVerification:
    """Run the full verification suite."""

    def test_all_pass(self):
        results = verify_opt(verbose=False)
        for name, passed in results.items():
            assert passed, f"Optimization verification failed: {name}"
