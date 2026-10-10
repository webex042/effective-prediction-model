"""
EPM Reproducibility Utilities
==============================

Ensures experiments produce identical results when re-run.

WHY REPRODUCIBILITY MATTERS:
    In research, every result must be reproducible. If you can't reproduce
    your own results, you can't trust them, and no one else can verify them.

    Sources of randomness in ML:
    1. Random number generators (NumPy, Python, PyTorch)
    2. Data shuffling / sampling
    3. Weight initialization
    4. Dropout masks
    5. GPU non-determinism (floating-point order of operations)

    We control these by:
    - Setting seeds for all random number generators
    - Optionally enabling deterministic algorithms
    - Logging the full environment (Python version, package versions, hardware)

USAGE:
    from src.reproducibility import set_seed, get_environment_info

    # Set all seeds
    set_seed(42)

    # Capture environment for experiment logs
    env = get_environment_info()
"""

from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Seed Management
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    This function seeds:
    1. Python's built-in random module
    2. NumPy's random number generator
    3. PyTorch (if installed) — both CPU and CUDA

    Parameters
    ----------
    seed : int
        The seed value. Same seed → same results.
    deterministic : bool
        If True and PyTorch is available, enables deterministic algorithms.
        This may slightly reduce performance but ensures reproducibility.

    Notes
    -----
    WHY MULTIPLE SEEDS?
        Each library maintains its own random number generator (RNG).
        NumPy's RNG is independent from Python's, which is independent
        from PyTorch's. We must seed ALL of them.

    Examples
    --------
    >>> set_seed(42)
    >>> np.random.rand(3)
    array([0.37454012, 0.95071431, 0.73199394])
    >>> set_seed(42)
    >>> np.random.rand(3)  # Same output!
    array([0.37454012, 0.95071431, 0.73199394])
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (optional — not installed in Stage 0)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seed set to {seed} (deterministic={deterministic})")
    except ImportError:
        logger.debug("PyTorch not installed — skipping torch seed")

    logger.info(f"Random seed set to {seed}")


def get_seed_state() -> dict[str, Any]:
    """
    Capture the current state of all random number generators.

    This allows you to save and restore exact random states
    for debugging or resuming interrupted experiments.

    Returns
    -------
    dict
        States of Python, NumPy, and optionally PyTorch RNGs.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }

    try:
        import torch
        state["torch_cpu"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    return state


# =============================================================================
# Environment Capture
# =============================================================================

def get_environment_info() -> dict[str, Any]:
    """
    Capture the full computational environment.

    WHY?
        Results can differ across Python versions, library versions,
        and hardware. Logging this information helps diagnose
        irreproducibility issues.

    Returns
    -------
    dict
        Environment information including:
        - Python version
        - OS information
        - Key package versions
        - Hardware info
        - Timestamp

    Examples
    --------
    >>> env = get_environment_info()
    >>> env["python_version"]
    '3.11.5'
    >>> env["packages"]["numpy"]
    '1.26.0'
    """
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # Package versions
    packages = {}
    for pkg_name in ["numpy", "pandas", "matplotlib", "yaml", "pytest"]:
        try:
            pkg = __import__(pkg_name)
            packages[pkg_name] = getattr(pkg, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = "not installed"

    # Optional ML packages
    for pkg_name in ["sklearn", "torch", "xgboost"]:
        try:
            pkg = __import__(pkg_name)
            packages[pkg_name] = getattr(pkg, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = "not installed"

    info["packages"] = packages

    # GPU info (if PyTorch is available)
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = (
                torch.cuda.get_device_properties(0).total_mem / 1024**2
            )
    except ImportError:
        info["cuda_available"] = False

    return info


def log_environment() -> None:
    """Log the full environment info at INFO level."""
    env = get_environment_info()
    logger.info("=" * 60)
    logger.info("ENVIRONMENT")
    logger.info("=" * 60)
    logger.info(f"Python:   {env['python_version'].split()[0]}")
    logger.info(f"OS:       {env['os']}")
    logger.info(f"Machine:  {env['machine']}")
    for pkg, ver in env.get("packages", {}).items():
        logger.info(f"  {pkg}: {ver}")
    if env.get("cuda_available"):
        logger.info(f"GPU:      {env.get('gpu_name', 'unknown')}")
    else:
        logger.info("GPU:      Not available")
    logger.info("=" * 60)
