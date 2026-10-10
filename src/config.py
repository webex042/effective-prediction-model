"""
EPM Configuration System
========================

Provides a YAML-based configuration system with Python dataclass parsing.

WHY THIS EXISTS:
    Hardcoding hyperparameters in code makes experiments unreproducible and
    hard to track. A configuration system lets us:
    1. Version-control our settings alongside our code
    2. Override defaults per-experiment without changing source code
    3. Log the exact config used for every experiment

HOW IT WORKS:
    1. Load a YAML file into a nested dictionary
    2. Access values with dot-notation via a Config wrapper
    3. Merge multiple configs (default + experiment overrides)
    4. Freeze config after loading to prevent accidental mutation

USAGE:
    from src.config import load_config

    # Load default config
    config = load_config()

    # Load with overrides
    config = load_config("configs/experiment_01.yaml")

    # Access values
    lr = config.training.learning_rate
    seed = config.reproducibility.seed
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml


# =============================================================================
# Config Class
# =============================================================================

class Config:
    """
    A lightweight configuration wrapper that supports dot-notation access.

    Converts nested dictionaries into attribute-accessible objects.
    This makes configs feel natural in Python code:
        config.training.learning_rate  instead of  config['training']['learning_rate']

    Parameters
    ----------
    data : dict
        The raw configuration dictionary.
    name : str
        A human-readable name for this config section (for error messages).

    Examples
    --------
    >>> cfg = Config({"training": {"lr": 0.001, "epochs": 100}})
    >>> cfg.training.lr
    0.001
    >>> cfg.training.epochs
    100
    """

    def __init__(self, data: dict[str, Any], name: str = "root") -> None:
        # Store the raw dict (use object.__setattr__ to bypass our __setattr__)
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_frozen", False)

        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = Config(value, name=f"{name}.{key}")
            else:
                self._data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Access config values as attributes: config.training.lr"""
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(
                f"Config section '{self._name}' has no key '{key}'. "
                f"Available keys: {list(self._data.keys())}"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        """Prevent accidental modification after freezing."""
        if self._frozen:
            raise AttributeError(
                f"Config is frozen. Cannot set '{key}'. "
                "Create a new config or use merge() for overrides."
            )
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: 'learning_rate' in config.training"""
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def freeze(self) -> Config:
        """
        Freeze this config to prevent further modification.
        Returns self for chaining.
        """
        object.__setattr__(self, "_frozen", True)
        for value in self._data.values():
            if isinstance(value, Config):
                value.freeze()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain nested dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Safe access with a default value."""
        return self._data.get(key, default)

    def keys(self) -> list[str]:
        """Return all top-level keys."""
        return list(self._data.keys())


# =============================================================================
# Config Loading & Merging
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep-merge two dictionaries. Values in 'override' take precedence.

    WHY deep merge?
        If you have a default config with 50 settings and want to change
        only 2 for an experiment, you should only need to specify those 2
        in your override file. Deep merge handles nested structures correctly.

    Parameters
    ----------
    base : dict
        The base (default) configuration.
    override : dict
        The override configuration (experiment-specific).

    Returns
    -------
    dict
        Merged configuration with override values taking precedence.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the YAML file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return data


def load_config(
    override_path: Optional[str | Path] = None,
    default_path: str | Path = "configs/default.yaml",
) -> Config:
    """
    Load project configuration.

    Loads the default config, then optionally merges experiment-specific
    overrides on top.

    Parameters
    ----------
    override_path : str or Path, optional
        Path to an override YAML file. Only specified keys are overridden;
        all other settings keep their default values.
    default_path : str or Path
        Path to the default YAML config. Usually 'configs/default.yaml'.

    Returns
    -------
    Config
        A frozen Config object with dot-notation access.

    Examples
    --------
    >>> config = load_config()
    >>> config.reproducibility.seed
    42

    >>> config = load_config("configs/experiment_01.yaml")
    >>> config.training.learning_rate  # Overridden value
    0.0001
    """
    # Find project root (walk up from this file until we find configs/)
    project_root = _find_project_root()

    default_full = project_root / default_path
    base_data = load_yaml(default_full) if default_full.exists() else {}

    if override_path is not None:
        override_full = Path(override_path)
        if not override_full.is_absolute():
            override_full = project_root / override_full
        override_data = load_yaml(override_full)
        merged = _deep_merge(base_data, override_data)
    else:
        merged = base_data

    return Config(merged).freeze()


def _find_project_root() -> Path:
    """
    Find the project root by looking for pyproject.toml or configs/ directory.
    Walks up from the current working directory.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "configs").is_dir():
            return parent
    # Fallback to cwd
    return current


def save_config(config: Config, path: str | Path) -> None:
    """
    Save a Config object to a YAML file.

    Useful for logging the exact configuration used in an experiment.

    Parameters
    ----------
    config : Config
        The configuration to save.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
