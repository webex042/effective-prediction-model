"""
EPM Generative Module — Autoencoders, Variational Autoencoders & Latent Generative Models
========================================================================================

Provides NumPy scratch implementations, PyTorch production architectures, and benchmarks:
  - From Scratch: VAEScratch, check_vae_numerical_gradients
  - Models: StandardAutoencoder, VariationalAutoencoder, ConditionalVAE, train_generative_model
  - Benchmarks: evaluate_generative_model, run_generative_benchmark
"""

from src.generative.vae_scratch import (
    VAEScratch,
    check_vae_numerical_gradients,
)
from src.generative.models import (
    StandardAutoencoder,
    VariationalAutoencoder,
    ConditionalVAE,
    train_generative_model,
)
from src.generative.benchmarks import (
    evaluate_generative_model,
    run_generative_benchmark,
)

__all__ = [
    "VAEScratch",
    "check_vae_numerical_gradients",
    "StandardAutoencoder",
    "VariationalAutoencoder",
    "ConditionalVAE",
    "train_generative_model",
    "evaluate_generative_model",
    "run_generative_benchmark",
]
