"""
Probability & Statistics — From Scratch
========================================

Stage 1: Mathematical Foundation

WHY PROBABILITY FOR ML?
    Machine learning is about making predictions under UNCERTAINTY.

    - Classification: P(class = cat | image)
    - Regression: P(y | x) — what output distribution given this input?
    - Generation: P(x) — what does the data distribution look like?
    - Bayesian ML: P(θ | data) — what parameters are plausible given data?

    Diffusion models (Stage 8-9) are deeply rooted in probability:
    they learn to reverse a noise process by modeling p(x_{t-1} | x_t).

WHAT'S IN THIS MODULE:
    - gaussian_pdf: The normal distribution — most important distribution in ML
    - gaussian_log_likelihood: Log-probability of data under a Gaussian
    - bayes_posterior: Bayes' theorem — updating beliefs with evidence
    - sample_gaussian: Generate samples from a Gaussian distribution
    - kl_divergence_gaussians: Measure difference between distributions (used in VAEs)
    - entropy: Measure uncertainty in a distribution

USAGE:
    from src.math_foundations.probability import gaussian_pdf, bayes_posterior

    # Probability density at x=0 for standard normal N(0,1)
    p = gaussian_pdf(0.0, mean=0.0, std=1.0)  # ≈ 0.3989

    # Bayes' theorem
    posterior = bayes_posterior(prior=0.01, likelihood=0.95, evidence=0.05)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Gaussian (Normal) Distribution
# =============================================================================

def gaussian_pdf(
    x: Union[float, ArrayLike],
    mean: float = 0.0,
    std: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Compute the probability density function (PDF) of a Gaussian distribution.

    INTUITION:
        The Gaussian (bell curve) is the most common distribution in nature
        and ML. The Central Limit Theorem tells us that the average of many
        independent random variables tends to be Gaussian, regardless of their
        individual distributions.

        The PDF tells you: "How relatively likely is this particular value?"
        (Note: for continuous distributions, the PDF can exceed 1.0.
         It's a density, not a probability.)

    MATH:
        p(x | μ, σ) = (1 / (σ√(2π))) × exp(-(x - μ)² / (2σ²))

        Where:
        - μ (mean): center of the bell curve
        - σ (std dev): width of the bell curve
        - σ²: variance = spread of the distribution

    ML APPLICATION:
        - Loss functions: MSE is equivalent to maximizing Gaussian likelihood
        - VAEs: latent space uses Gaussian prior/posterior
        - Diffusion models: noise is Gaussian
        - Weight initialization: often Gaussian (e.g., Xavier, He)
        - Batch normalization: normalizes to approximately Gaussian

    Parameters
    ----------
    x : float or array-like
        Point(s) at which to evaluate the PDF.
    mean : float
        Mean (μ) of the distribution.
    std : float
        Standard deviation (σ). Must be positive.

    Returns
    -------
    float or np.ndarray
        PDF value(s) at x.

    Examples
    --------
    >>> gaussian_pdf(0.0, mean=0.0, std=1.0)  # Peak of standard normal
    0.3989422804014327
    >>> gaussian_pdf(1.0, mean=0.0, std=1.0)  # 1 std from mean
    0.24197072451914337
    """
    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    x = np.asarray(x, dtype=np.float64)

    # Implement the formula step by step
    coefficient = 1.0 / (std * np.sqrt(2.0 * np.pi))
    exponent = -((x - mean) ** 2) / (2.0 * std ** 2)
    pdf = coefficient * np.exp(exponent)

    # Return scalar if input was scalar
    if pdf.ndim == 0:
        return float(pdf)
    return pdf


def gaussian_log_pdf(
    x: Union[float, ArrayLike],
    mean: float = 0.0,
    std: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Compute the log of the Gaussian PDF.

    WHY LOG?
        In practice we almost always work with LOG probabilities because:
        1. Products become sums: log(p₁ × p₂) = log(p₁) + log(p₂)
        2. Avoids numerical underflow: p = 1e-300 is dangerous, log(p) = -690 is fine
        3. Loss functions use negative log-likelihood: -log p(data | θ)

    MATH:
        log p(x | μ, σ) = -½ log(2πσ²) - (x - μ)² / (2σ²)

    Parameters
    ----------
    x : float or array-like
        Point(s) at which to evaluate.
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation.

    Returns
    -------
    float or np.ndarray
        Log-PDF value(s).
    """
    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    x = np.asarray(x, dtype=np.float64)
    variance = std ** 2

    log_pdf = -0.5 * np.log(2.0 * np.pi * variance) - ((x - mean) ** 2) / (2.0 * variance)

    if log_pdf.ndim == 0:
        return float(log_pdf)
    return log_pdf


def gaussian_log_likelihood(
    data: ArrayLike,
    mean: float = 0.0,
    std: float = 1.0,
) -> float:
    """
    Compute the total log-likelihood of data under a Gaussian model.

    INTUITION:
        "How likely is my entire dataset, assuming it came from N(μ, σ²)?"

        Higher log-likelihood = the model explains the data better.
        Maximum Likelihood Estimation (MLE) finds the μ and σ that maximize this.

    MATH:
        log L(μ, σ | data) = Σᵢ log p(xᵢ | μ, σ)

        For Gaussian, the MLE solutions are:
        μ_MLE = mean(data)
        σ²_MLE = mean((data - μ)²)

    ML APPLICATION:
        - Training = maximizing log-likelihood (or equivalently, minimizing -log L)
        - MSE loss is proportional to negative Gaussian log-likelihood
        - This connects statistics to neural network training!

    Parameters
    ----------
    data : array-like
        Observed data points.
    mean : float
        Assumed mean of the Gaussian.
    std : float
        Assumed standard deviation.

    Returns
    -------
    float
        Total log-likelihood (sum of log-probabilities).

    Examples
    --------
    >>> data = [0.1, -0.2, 0.3, 0.0, -0.1]
    >>> gaussian_log_likelihood(data, mean=0.0, std=1.0)
    -4.637...
    """
    data = np.asarray(data, dtype=np.float64)
    log_probs = gaussian_log_pdf(data, mean=mean, std=std)
    return float(np.sum(log_probs))


# =============================================================================
# Bayes' Theorem
# =============================================================================

def bayes_posterior(
    prior: float,
    likelihood: float,
    evidence: float,
) -> float:
    """
    Compute the posterior probability using Bayes' theorem.

    INTUITION:
        Bayes' theorem tells you how to update your beliefs
        when you see new evidence.

        Example: A medical test
        - Prior: P(disease) = 1% (rare disease)
        - Likelihood: P(positive test | disease) = 95% (good test)
        - P(positive test) = 5% (overall positive rate)
        - Posterior: P(disease | positive test) = ?

        Result: 19% — even with a positive test, there's only a 19% chance
        of actually having the disease! (Because the disease is rare.)

    MATH:
        P(A | B) = P(B | A) × P(A) / P(B)

        posterior = likelihood × prior / evidence

    ML APPLICATION:
        - Bayesian neural networks: P(weights | data)
        - Naive Bayes classifier: P(class | features)
        - Posterior predictive: P(y_new | data)
        - Diffusion models: relate to Bayesian inference over denoising steps

    Parameters
    ----------
    prior : float
        P(A) — prior probability of the hypothesis.
    likelihood : float
        P(B|A) — probability of evidence given the hypothesis.
    evidence : float
        P(B) — total probability of the evidence.

    Returns
    -------
    float
        P(A|B) — posterior probability.

    Examples
    --------
    >>> bayes_posterior(prior=0.01, likelihood=0.95, evidence=0.05)
    0.19
    """
    if evidence <= 0:
        raise ValueError(f"Evidence P(B) must be positive, got {evidence}")
    if not (0 <= prior <= 1):
        raise ValueError(f"Prior must be in [0, 1], got {prior}")
    if not (0 <= likelihood <= 1):
        raise ValueError(f"Likelihood must be in [0, 1], got {likelihood}")

    posterior = (likelihood * prior) / evidence

    # Posterior should be a valid probability
    if posterior > 1.0:
        logger.warning(
            f"Posterior {posterior:.4f} > 1.0 — check your inputs. "
            "Evidence P(B) should satisfy P(B) ≥ P(B|A)×P(A)."
        )

    return float(posterior)


# =============================================================================
# Sampling
# =============================================================================

def sample_gaussian(
    mean: float = 0.0,
    std: float = 1.0,
    size: Union[int, tuple] = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from a Gaussian distribution.

    INTUITION:
        Generating random numbers from a known distribution.
        This is fundamental to:
        - Data augmentation
        - Monte Carlo methods
        - Diffusion models (adding noise)
        - VAE reparameterization trick
        - Stochastic gradient descent (random mini-batches)

    MATH:
        x ~ N(μ, σ²)

        Reparameterization trick (important for VAEs!):
        x = μ + σ × ε,  where ε ~ N(0, 1)

        This separates the randomness (ε) from the parameters (μ, σ),
        allowing gradients to flow through μ and σ.

    Parameters
    ----------
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation.
    size : int or tuple
        Number/shape of samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Samples from N(mean, std²).

    Examples
    --------
    >>> samples = sample_gaussian(mean=0, std=1, size=1000, seed=42)
    >>> abs(np.mean(samples)) < 0.1  # Should be close to 0
    True
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Reparameterization trick: x = mean + std * epsilon
    epsilon = rng.standard_normal(size)
    samples = mean + std * epsilon

    return samples


# =============================================================================
# KL Divergence (for VAEs later)
# =============================================================================

def kl_divergence_gaussians(
    mu1: float, sigma1: float,
    mu2: float = 0.0, sigma2: float = 1.0,
) -> float:
    """
    KL divergence between two univariate Gaussians.

    INTUITION:
        KL divergence measures "how different" two probability distributions are.
        It's NOT symmetric: KL(P||Q) ≠ KL(Q||P).

        Think of it as: "How many extra bits do I need to encode samples from P
        using an encoding optimized for Q?"

    MATH:
        KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) =
            log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

    ML APPLICATION:
        - VAE loss: KL(q(z|x) || p(z)) forces encoder toward the prior
        - Diffusion models: forward process has known KL divergence
        - Regularization: penalize complex posteriors
        - Policy optimization (PPO): constrain policy updates

    Parameters
    ----------
    mu1, sigma1 : float
        Parameters of distribution P (e.g., encoder output).
    mu2, sigma2 : float
        Parameters of distribution Q (e.g., prior). Default: N(0,1).

    Returns
    -------
    float
        KL(P || Q). Always ≥ 0. Equal to 0 iff P = Q.

    Examples
    --------
    >>> kl_divergence_gaussians(0.0, 1.0, 0.0, 1.0)  # Same distribution
    0.0
    >>> kl_divergence_gaussians(1.0, 1.0, 0.0, 1.0)  # Shifted mean
    0.5
    """
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("Standard deviations must be positive")

    kl = (
        np.log(sigma2 / sigma1)
        + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        - 0.5
    )
    return float(kl)


# =============================================================================
# Entropy
# =============================================================================

def entropy_discrete(probs: ArrayLike) -> float:
    """
    Compute the entropy of a discrete probability distribution.

    INTUITION:
        Entropy measures UNCERTAINTY or INFORMATION CONTENT.
        - High entropy → lots of uncertainty (uniform distribution)
        - Low entropy → very predictable (one outcome dominates)

        A fair coin (p=0.5, 0.5) has entropy = 1 bit.
        A biased coin (p=0.99, 0.01) has entropy ≈ 0.08 bits.

    MATH:
        H(P) = -Σᵢ pᵢ × log₂(pᵢ)

    ML APPLICATION:
        - Cross-entropy loss: -Σ y_true × log(y_pred) — the standard classification loss
        - Information gain in decision trees
        - Exploration in reinforcement learning
        - Evaluating model confidence

    Parameters
    ----------
    probs : array-like
        Probability distribution (must sum to 1, all values ≥ 0).

    Returns
    -------
    float
        Entropy in bits (using log₂).

    Examples
    --------
    >>> entropy_discrete([0.5, 0.5])  # Fair coin: 1 bit
    1.0
    >>> entropy_discrete([1.0, 0.0])  # Certain outcome: 0 bits
    0.0
    """
    probs = np.asarray(probs, dtype=np.float64)

    if np.any(probs < 0):
        raise ValueError("Probabilities must be non-negative")
    if abs(np.sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1, got {np.sum(probs)}")

    # Avoid log(0) by only computing for nonzero probabilities
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * np.log2(p)

    return float(h)


# =============================================================================
# Maximum Likelihood Estimation
# =============================================================================

def mle_gaussian(data: ArrayLike) -> dict[str, float]:
    """
    Compute Maximum Likelihood Estimates for Gaussian parameters.

    INTUITION:
        "What μ and σ make the observed data most probable?"

        MLE finds the parameters that maximize the likelihood function
        (or equivalently, minimize the negative log-likelihood).

    MATH:
        μ_MLE = (1/n) Σᵢ xᵢ                    (sample mean)
        σ²_MLE = (1/n) Σᵢ (xᵢ - μ_MLE)²        (sample variance, biased)

        Note: MLE gives the biased variance (divides by n, not n-1).
        The unbiased estimator divides by (n-1), which is what np.var(ddof=1) uses.

    ML APPLICATION:
        - Training a model = finding MLE parameters
        - Linear regression MLE = ordinary least squares
        - MSE loss minimization = Gaussian MLE

    Parameters
    ----------
    data : array-like
        Observed data points.

    Returns
    -------
    dict
        MLE estimates with keys: mean, variance_mle, std_mle,
        variance_unbiased, std_unbiased, log_likelihood.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n < 2:
        raise ValueError("Need at least 2 data points for MLE")

    # MLE estimates (from scratch)
    mean_mle = 0.0
    for x in data:
        mean_mle += x
    mean_mle /= n

    var_mle = 0.0
    for x in data:
        var_mle += (x - mean_mle) ** 2
    var_mle /= n

    # Unbiased variance (Bessel's correction)
    var_unbiased = var_mle * n / (n - 1)

    std_mle = np.sqrt(var_mle)
    std_unbiased = np.sqrt(var_unbiased)

    # Log-likelihood at MLE parameters
    ll = gaussian_log_likelihood(data, mean=mean_mle, std=std_mle if std_mle > 0 else 1e-10)

    return {
        "mean": float(mean_mle),
        "variance_mle": float(var_mle),
        "std_mle": float(std_mle),
        "variance_unbiased": float(var_unbiased),
        "std_unbiased": float(std_unbiased),
        "log_likelihood": float(ll),
        "n_samples": n,
    }


# =============================================================================
# Verification
# =============================================================================

def verify_implementations(verbose: bool = True) -> dict[str, bool]:
    """
    Verify probability implementations against known values and scipy.

    Returns
    -------
    dict
        Mapping of test name → pass/fail.
    """
    results = {}
    tol = 1e-6

    # --- Gaussian PDF at peak ---
    peak = gaussian_pdf(0.0, mean=0.0, std=1.0)
    expected_peak = 1.0 / np.sqrt(2 * np.pi)  # ≈ 0.3989
    passed = abs(peak - expected_peak) < tol
    results["gaussian_pdf_peak"] = passed
    if verbose:
        logger.info(
            f"Gaussian PDF at mean: {'✓' if passed else '✗'} "
            f"(got {peak:.6f}, expected {expected_peak:.6f})"
        )

    # --- Gaussian PDF symmetry ---
    left = gaussian_pdf(-1.0, mean=0.0, std=1.0)
    right = gaussian_pdf(1.0, mean=0.0, std=1.0)
    passed = abs(left - right) < tol
    results["gaussian_symmetry"] = passed
    if verbose:
        logger.info(f"Gaussian symmetry: {'✓' if passed else '✗'}")

    # --- Log PDF consistency ---
    x_test = 1.5
    log_p = gaussian_log_pdf(x_test, mean=0.0, std=1.0)
    p = gaussian_pdf(x_test, mean=0.0, std=1.0)
    passed = abs(log_p - np.log(p)) < tol
    results["log_pdf_consistency"] = passed
    if verbose:
        logger.info(f"Log PDF = log(PDF): {'✓' if passed else '✗'}")

    # --- Bayes' theorem ---
    post = bayes_posterior(prior=0.01, likelihood=0.95, evidence=0.05)
    expected_post = 0.95 * 0.01 / 0.05  # = 0.19
    passed = abs(post - expected_post) < tol
    results["bayes_theorem"] = passed
    if verbose:
        logger.info(
            f"Bayes' theorem: {'✓' if passed else '✗'} "
            f"(got {post:.4f}, expected {expected_post:.4f})"
        )

    # --- KL divergence: same distributions → 0 ---
    kl_same = kl_divergence_gaussians(0.0, 1.0, 0.0, 1.0)
    passed = abs(kl_same) < tol
    results["kl_same_distributions"] = passed
    if verbose:
        logger.info(f"KL(P||P) = 0: {'✓' if passed else '✗'} (got {kl_same:.6f})")

    # --- KL divergence: known value ---
    kl_shifted = kl_divergence_gaussians(1.0, 1.0, 0.0, 1.0)
    expected_kl = 0.5  # KL(N(1,1)||N(0,1)) = 0.5
    passed = abs(kl_shifted - expected_kl) < tol
    results["kl_known_value"] = passed
    if verbose:
        logger.info(
            f"KL(N(1,1)||N(0,1)) = 0.5: {'✓' if passed else '✗'} "
            f"(got {kl_shifted:.6f})"
        )

    # --- Entropy: fair coin ---
    h_fair = entropy_discrete([0.5, 0.5])
    passed = abs(h_fair - 1.0) < tol
    results["entropy_fair_coin"] = passed
    if verbose:
        logger.info(
            f"Entropy of fair coin = 1 bit: {'✓' if passed else '✗'} "
            f"(got {h_fair:.6f})"
        )

    # --- MLE: should recover true parameters ---
    np.random.seed(42)
    true_mean, true_std = 5.0, 2.0
    data = np.random.normal(true_mean, true_std, size=10000)
    mle = mle_gaussian(data)
    passed_mean = abs(mle["mean"] - true_mean) < 0.1
    passed_std = abs(mle["std_mle"] - true_std) < 0.1
    results["mle_mean"] = passed_mean
    results["mle_std"] = passed_std
    if verbose:
        logger.info(
            f"MLE mean: {'✓' if passed_mean else '✗'} "
            f"(got {mle['mean']:.4f}, true {true_mean})"
        )
        logger.info(
            f"MLE std: {'✓' if passed_std else '✗'} "
            f"(got {mle['std_mle']:.4f}, true {true_std})"
        )

    # Summary
    total = len(results)
    passed_count = sum(results.values())
    if verbose:
        logger.info(f"\nProbability verification: {passed_count}/{total} passed")

    return results
