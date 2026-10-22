"""EPM Benchmarking — algorithm catalogue, dataset compatibility and the fair benchmark runner."""

from src.benchmarking.catalog import (
    AlgorithmSpec,
    ALGORITHMS,
    get_algorithm,
    list_algorithms,
    check_compatibility,
    compatible_algorithms,
    recommend_algorithms,
    recommend_datasets,
)
from src.benchmarking.runner import BenchmarkConfig, BenchmarkError, run_benchmark, load_history, load_run

__all__ = [
    "AlgorithmSpec", "ALGORITHMS", "get_algorithm", "list_algorithms", "check_compatibility",
    "compatible_algorithms", "recommend_algorithms", "recommend_datasets",
    "BenchmarkConfig", "BenchmarkError", "run_benchmark", "load_history", "load_run",
]
