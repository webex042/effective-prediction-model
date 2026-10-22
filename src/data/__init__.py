"""EPM Data Module — dataset registry, verified loaders, caching and validation."""

from src.data.registry import (
    TASKS,
    DatasetSpec,
    DATASETS,
    get_dataset_spec,
    list_datasets,
    validate_registry,
)
from src.data.loaders import (
    LoadedDataset,
    DatasetLoadError,
    load_dataset,
    download_dataset,
    dataset_status,
    clear_memory_cache,
)

__all__ = [
    "TASKS", "DatasetSpec", "DATASETS", "get_dataset_spec", "list_datasets", "validate_registry",
    "LoadedDataset", "DatasetLoadError", "load_dataset", "download_dataset", "dataset_status",
    "clear_memory_cache",
]
