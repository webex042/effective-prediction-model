"""
EPM Model Compression Module — Stage 12
=========================================
Knowledge Distillation, Quantization, and Pruning.
"""

from .distillation import (
    KnowledgeDistillationLoss,
    DistillationTrainer,
    FeatureHintProjector,
)
from .quantization import (
    calculate_scale_and_zero_point,
    quantize_tensor,
    dequantize_tensor,
    quantize_dequantize,
    compute_sqnr,
    get_model_size_kb,
    quantize_model_dynamic_int8,
    profile_layer_quantization_errors,
)
from .pruning import (
    calculate_sparsity,
    prune_unstructured_magnitude,
    StructuredMLPCompressor,
    iterative_prune_and_finetune,
)
from .benchmarks import (
    BenchmarkClassifier,
    generate_synthetic_industrial_data,
    measure_inference_latency,
    evaluate_accuracy,
    count_parameters,
    run_stage12_benchmarks,
)

__all__ = [
    "KnowledgeDistillationLoss",
    "DistillationTrainer",
    "FeatureHintProjector",
    "calculate_scale_and_zero_point",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_dequantize",
    "compute_sqnr",
    "get_model_size_kb",
    "quantize_model_dynamic_int8",
    "profile_layer_quantization_errors",
    "calculate_sparsity",
    "prune_unstructured_magnitude",
    "StructuredMLPCompressor",
    "iterative_prune_and_finetune",
    "BenchmarkClassifier",
    "generate_synthetic_industrial_data",
    "measure_inference_latency",
    "evaluate_accuracy",
    "count_parameters",
    "run_stage12_benchmarks",
]
