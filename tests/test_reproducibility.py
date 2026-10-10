"""
Tests for the EPM Reproducibility System
=========================================

These tests verify that:
1. set_seed() produces deterministic random numbers
2. Seed state can be captured and compared
3. Environment info returns expected fields
"""

import numpy as np

from src.reproducibility import set_seed, get_seed_state, get_environment_info


class TestSetSeed:
    """Tests for seed management."""

    def test_numpy_determinism(self):
        """Same seed should produce identical NumPy random sequences."""
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds should produce different sequences."""
        set_seed(42)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)

    def test_python_random_determinism(self):
        """Python's random module should also be deterministic."""
        import random

        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b


class TestSeedState:
    """Tests for seed state capture."""

    def test_state_has_expected_keys(self):
        """Seed state should contain python and numpy states."""
        set_seed(42)
        state = get_seed_state()
        assert "python" in state
        assert "numpy" in state


class TestEnvironmentInfo:
    """Tests for environment capture."""

    def test_basic_fields_present(self):
        """Environment info should contain essential fields."""
        env = get_environment_info()
        assert "python_version" in env
        assert "os" in env
        assert "packages" in env
        assert "timestamp" in env

    def test_packages_include_numpy(self):
        """NumPy should always be listed (it's a core dependency)."""
        env = get_environment_info()
        assert "numpy" in env["packages"]
        assert env["packages"]["numpy"] != "not installed"

    def test_timestamp_is_iso_format(self):
        """Timestamp should be in ISO 8601 format."""
        env = get_environment_info()
        # Basic check: contains 'T' separator and date components
        assert "T" in env["timestamp"]
