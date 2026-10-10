"""
Tests for the EPM Configuration System
=======================================

These tests verify that:
1. Config loads YAML files correctly
2. Dot-notation access works
3. Deep merge correctly combines configs
4. Frozen configs prevent mutation
5. Missing keys raise clear errors
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config, _deep_merge, load_yaml, save_config


# =============================================================================
# Config Class Tests
# =============================================================================

class TestConfig:
    """Tests for the Config wrapper class."""

    def test_basic_access(self):
        """Config should support dot-notation access."""
        cfg = Config({"learning_rate": 0.001, "epochs": 100})
        assert cfg.learning_rate == 0.001
        assert cfg.epochs == 100

    def test_nested_access(self):
        """Nested dicts should become nested Config objects."""
        cfg = Config({
            "training": {
                "lr": 0.001,
                "optimizer": {
                    "name": "adam",
                    "weight_decay": 0.01,
                },
            }
        })
        assert cfg.training.lr == 0.001
        assert cfg.training.optimizer.name == "adam"
        assert cfg.training.optimizer.weight_decay == 0.01

    def test_missing_key_raises_error(self):
        """Accessing a non-existent key should raise AttributeError."""
        cfg = Config({"a": 1})
        with pytest.raises(AttributeError, match="has no key 'b'"):
            _ = cfg.b

    def test_contains(self):
        """'in' operator should work."""
        cfg = Config({"a": 1, "b": 2})
        assert "a" in cfg
        assert "c" not in cfg

    def test_to_dict(self):
        """to_dict() should round-trip back to the original dict."""
        original = {"x": 1, "nested": {"y": 2, "z": 3}}
        cfg = Config(original)
        result = cfg.to_dict()
        assert result == original

    def test_freeze_prevents_mutation(self):
        """Frozen config should raise error on assignment."""
        cfg = Config({"a": 1}).freeze()
        with pytest.raises(AttributeError, match="frozen"):
            cfg.a = 2

    def test_freeze_is_recursive(self):
        """Freezing should propagate to nested configs."""
        cfg = Config({"outer": {"inner": 1}}).freeze()
        with pytest.raises(AttributeError, match="frozen"):
            cfg.outer.inner = 2

    def test_get_with_default(self):
        """get() should return a default for missing keys."""
        cfg = Config({"a": 1})
        assert cfg.get("a") == 1
        assert cfg.get("missing", 42) == 42
        assert cfg.get("missing") is None

    def test_keys(self):
        """keys() should return all top-level keys."""
        cfg = Config({"a": 1, "b": 2, "c": 3})
        assert sorted(cfg.keys()) == ["a", "b", "c"]

    def test_repr(self):
        """repr should show the dict representation."""
        cfg = Config({"x": 1})
        assert "x" in repr(cfg)


# =============================================================================
# Deep Merge Tests
# =============================================================================

class TestDeepMerge:
    """Tests for the _deep_merge function."""

    def test_simple_override(self):
        """Override values should replace base values."""
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        """Nested dicts should be merged recursively."""
        base = {"training": {"lr": 0.001, "epochs": 100}}
        override = {"training": {"lr": 0.0001}}
        result = _deep_merge(base, override)
        assert result == {"training": {"lr": 0.0001, "epochs": 100}}

    def test_new_keys_added(self):
        """Keys in override but not in base should be added."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        """Deep merge should not modify the original base dict."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}  # Unchanged


# =============================================================================
# YAML Loading Tests
# =============================================================================

class TestLoadYaml:
    """Tests for YAML file loading."""

    def test_load_valid_yaml(self, tmp_path):
        """Should parse a valid YAML file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnumber: 42\n")
        result = load_yaml(yaml_file)
        assert result == {"key": "value", "number": 42}

    def test_load_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_yaml("nonexistent_file.yaml")

    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML file should return empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        result = load_yaml(yaml_file)
        assert result == {}


# =============================================================================
# Save Config Tests
# =============================================================================

class TestSaveConfig:
    """Tests for saving config to YAML."""

    def test_save_and_reload(self, tmp_path):
        """Saved config should be loadable and match original."""
        original = Config({"lr": 0.001, "nested": {"a": 1, "b": "hello"}})
        save_path = tmp_path / "saved.yaml"
        save_config(original, save_path)

        # Reload and verify
        loaded = load_yaml(save_path)
        assert loaded == original.to_dict()
