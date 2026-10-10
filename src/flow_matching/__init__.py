"""
EPM Flow Matching Module — Continuous Normalizing Flows & Optimal Transport Vector Fields
========================================================================================

Exported Symbols:
    - ContinuousTimeEmbedding, VelocityFieldNet, ConditionalVelocityFieldNet
    - ProbabilityPath, OptimalTransportCFMPath, GaussianCFMPath, compute_minibatch_ot
    - ODESolver, euler_step, midpoint_step, rk4_step, compute_path_straightness
    - ContinuousNormalizingFlow, compute_exact_divergence, compute_hutchinson_divergence
    - compute_sliced_wasserstein, generate_two_moons_data, generate_sensor_manifold_data
    - train_flow_matching, run_flow_matching_benchmark
"""

from .models import (
    ContinuousTimeEmbedding,
    VelocityFieldNet,
    ConditionalVelocityFieldNet,
)
from .probability_paths import (
    ProbabilityPath,
    OptimalTransportCFMPath,
    GaussianCFMPath,
    compute_minibatch_ot,
)
from .ode_solvers import (
    ODESolver,
    euler_step,
    midpoint_step,
    rk4_step,
    compute_path_straightness,
)
from .continuous_normalizing_flow import (
    ContinuousNormalizingFlow,
    compute_exact_divergence,
    compute_hutchinson_divergence,
)
from .benchmarks import (
    compute_sliced_wasserstein,
    generate_two_moons_data,
    generate_sensor_manifold_data,
    train_flow_matching,
    run_flow_matching_benchmark,
)

__all__ = [
    "ContinuousTimeEmbedding",
    "VelocityFieldNet",
    "ConditionalVelocityFieldNet",
    "ProbabilityPath",
    "OptimalTransportCFMPath",
    "GaussianCFMPath",
    "compute_minibatch_ot",
    "ODESolver",
    "euler_step",
    "midpoint_step",
    "rk4_step",
    "compute_path_straightness",
    "ContinuousNormalizingFlow",
    "compute_exact_divergence",
    "compute_hutchinson_divergence",
    "compute_sliced_wasserstein",
    "generate_two_moons_data",
    "generate_sensor_manifold_data",
    "train_flow_matching",
    "run_flow_matching_benchmark",
]
