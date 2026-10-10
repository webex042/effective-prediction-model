"""
EPM Neural Networks Module
==========================

From-scratch NumPy and PyTorch implementations:
  - Layers: Linear, ReLU, Sigmoid, Tanh
  - Losses: SoftmaxCrossEntropyLoss, MSELoss
  - Scratch Model: NeuralNetworkScratch (with analytical backprop and Adam)
  - PyTorch Model: PyTorchMLP (nn.Module equivalent with weight synchronization)
  - Benchmarking: benchmark_scratch_vs_pytorch
"""

from src.neural_networks.layers import (
    Layer,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    SoftmaxCrossEntropyLoss,
    MSELoss,
)
from src.neural_networks.models_scratch import NeuralNetworkScratch
from src.neural_networks.models_pytorch import (
    PyTorchMLP,
    train_pytorch_model,
    benchmark_scratch_vs_pytorch,
    HAS_TORCH,
)

__all__ = [
    "Layer",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "SoftmaxCrossEntropyLoss",
    "MSELoss",
    "NeuralNetworkScratch",
    "PyTorchMLP",
    "train_pytorch_model",
    "benchmark_scratch_vs_pytorch",
    "HAS_TORCH",
]
