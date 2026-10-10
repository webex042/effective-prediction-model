"""
EPM Multimodal Module — Multimodal Representation Learning & Fusion Architectures
==================================================================================

Provides datasets, modality encoders, fusion mechanisms, and ablation benchmarking:
  - Dataset: MultimodalEquipmentDataset, generate_paired_multimodal_dataset
  - Encoders: SensorEncoder, TextEncoder
  - Fusion: ConcatenationFusion, GatedMultimodalFusion, CrossAttentionFusion
  - Models: MultimodalClassifier, SensorOnlyClassifier, TextOnlyClassifier
  - Ablation: run_multimodal_ablation_study, evaluate_model, compute_multiclass_metrics
"""

from src.multimodal.dataset import (
    MultimodalEquipmentDataset,
    generate_paired_multimodal_dataset,
)
from src.multimodal.models import (
    SensorEncoder,
    TextEncoder,
    ConcatenationFusion,
    GatedMultimodalFusion,
    CrossAttentionFusion,
    MultimodalClassifier,
    SensorOnlyClassifier,
    TextOnlyClassifier,
)
from src.multimodal.ablation import (
    compute_multiclass_metrics,
    train_model,
    evaluate_model,
    run_multimodal_ablation_study,
)

__all__ = [
    "MultimodalEquipmentDataset",
    "generate_paired_multimodal_dataset",
    "SensorEncoder",
    "TextEncoder",
    "ConcatenationFusion",
    "GatedMultimodalFusion",
    "CrossAttentionFusion",
    "MultimodalClassifier",
    "SensorOnlyClassifier",
    "TextOnlyClassifier",
    "compute_multiclass_metrics",
    "train_model",
    "evaluate_model",
    "run_multimodal_ablation_study",
]
