"""
Unit Tests for Stage 10 — Flow Matching & Continuous Normalizing Flows
=====================================================================
"""

import math
import pytest
import numpy as np
import torch

from src.flow_matching.models import (
    ContinuousTimeEmbedding,
    VelocityFieldNet,
    ConditionalVelocityFieldNet,
)
from src.flow_matching.probability_paths import (
    OptimalTransportCFMPath,
    GaussianCFMPath,
    compute_minibatch_ot,
)
from src.flow_matching.ode_solvers import (
    ODESolver,
    euler_step,
    rk4_step,
    compute_path_straightness,
)
from src.flow_matching.continuous_normalizing_flow import (
    ContinuousNormalizingFlow,
    compute_exact_divergence,
    compute_hutchinson_divergence,
)
from src.flow_matching.benchmarks import compute_sliced_wasserstein


class TestVelocityFieldModels:
    """Verifies vector field neural network forward passes, conditioning, and gradients."""

    def test_continuous_time_embedding_shapes(self):
        batch_size = 16
        embed_dim = 64
        model = ContinuousTimeEmbedding(embed_dim=embed_dim)
        t = torch.rand(batch_size, 1)

        emb = model(t)
        assert emb.shape == (batch_size, embed_dim)
        assert not torch.isnan(emb).any()

    def test_velocity_field_net_forward_and_backward(self):
        batch_size = 8
        input_dim = 4
        vnet = VelocityFieldNet(input_dim=input_dim, hidden_dim=64, num_layers=2)

        x = torch.randn(batch_size, input_dim, requires_grad=True)
        t = torch.rand(batch_size, 1)

        v = vnet(x, t)
        assert v.shape == (batch_size, input_dim)

        loss = v.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for p in vnet.parameters():
            assert p.grad is not None

    def test_conditional_velocity_field_net(self):
        batch_size = 12
        input_dim = 5
        num_classes = 3
        vnet = ConditionalVelocityFieldNet(input_dim=input_dim, num_classes=num_classes, hidden_dim=48)

        x = torch.randn(batch_size, input_dim)
        t = torch.rand(batch_size, 1)
        labels = torch.randint(0, num_classes, (batch_size,))

        v = vnet(x, t, condition=labels)
        assert v.shape == (batch_size, input_dim)


class TestProbabilityPaths:
    """Verifies analytical interpolation paths and Optimal Transport assignment."""

    def test_optimal_transport_cfm_path(self):
        path = OptimalTransportCFMPath(use_ot=False)
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[10.0, 20.0]])

        # At t=0, x_t = x0
        t0 = torch.tensor([[0.0]])
        assert torch.allclose(path.interpolate(x0, x1, t0), x0)

        # At t=1, x_t = x1
        t1 = torch.tensor([[1.0]])
        assert torch.allclose(path.interpolate(x0, x1, t1), x1)

        # At t=0.5, midpoint
        t_mid = torch.tensor([[0.5]])
        assert torch.allclose(path.interpolate(x0, x1, t_mid), torch.tensor([[5.0, 10.0]]))

        # Target velocity is constant x1 - x0
        u_t = path.target_velocity(x0, x1, t_mid)
        assert torch.allclose(u_t, torch.tensor([[10.0, 20.0]]))

    def test_gaussian_cfm_path(self):
        sigma_min = 1e-4
        path = GaussianCFMPath(sigma_min=sigma_min)
        x0 = torch.tensor([[2.0, 4.0]])
        x1 = torch.tensor([[10.0, 20.0]])

        # At t=0, x_t = x0
        t0 = torch.tensor([[0.0]])
        assert torch.allclose(path.interpolate(x0, x1, t0), x0, atol=1e-5)

        # At t=1, x_t = sigma_min * x0 + x1
        t1 = torch.tensor([[1.0]])
        expected_t1 = sigma_min * x0 + x1
        assert torch.allclose(path.interpolate(x0, x1, t1), expected_t1, atol=1e-5)

    def test_minibatch_optimal_transport_cost_reduction(self):
        # Create non-aligned batches
        x0 = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        x1 = torch.tensor([[10.0, 10.0], [0.0, 0.0]])

        # Unpaired cost: ||(0,0) - (10,10)||^2 + ||(10,10) - (0,0)||^2 = 200 + 200 = 400
        unpaired_cost = torch.sum((x0 - x1) ** 2).item()

        _, x1_matched = compute_minibatch_ot(x0, x1)
        # Paired cost: ||(0,0) - (0,0)||^2 + ||(10,10) - (10,10)||^2 = 0
        paired_cost = torch.sum((x0 - x1_matched) ** 2).item()

        assert paired_cost < unpaired_cost
        assert np.isclose(paired_cost, 0.0)


class TestODESolvers:
    """Verifies ODE integration accuracy and path curvature metrics."""

    def test_analytical_linear_decay_convergence(self):
        # Solve dx/dt = -x, x(0) = 1.0 => x(1) = exp(-1) = 0.367879...
        def linear_decay_v(x, t):
            return -x

        x0 = torch.tensor([[1.0]], dtype=torch.float64)
        analytical_x1 = math.exp(-1.0)

        # Euler solver
        euler_solver = ODESolver(method="euler")
        x_euler, _, nfe_euler = euler_solver.integrate(
            v_fn=linear_decay_v, x0=x0, t_start=0.0, t_end=1.0, num_steps=20
        )
        assert nfe_euler == 20

        # RK4 solver
        rk4_solver = ODESolver(method="rk4")
        x_rk4, _, nfe_rk4 = rk4_solver.integrate(
            v_fn=linear_decay_v, x0=x0, t_start=0.0, t_end=1.0, num_steps=20
        )
        assert nfe_rk4 == 80

        err_euler = abs(x_euler.item() - analytical_x1)
        err_rk4 = abs(x_rk4.item() - analytical_x1)

        # RK4 error should be orders of magnitude smaller than Euler
        assert err_rk4 < 1e-6
        assert err_rk4 < err_euler

    def test_straightness_metric_properties(self):
        # 1. Perfectly straight trajectory: x(t) = t * [1, 1]
        steps = 10
        t_vals = torch.linspace(0.0, 1.0, steps + 1).unsqueeze(-1).unsqueeze(-1)  # (11, 1, 1)
        straight_traj = torch.cat([t_vals * 5.0, t_vals * 10.0], dim=-1)  # (11, 1, 2)

        mean_s, std_s = compute_path_straightness(straight_traj)
        assert np.isclose(mean_s, 1.0, atol=1e-4)

        # 2. Zig-zag / curved trajectory
        curved_points = [torch.tensor([[0.0, 0.0]])]
        for step in range(1, steps + 1):
            offset_y = 5.0 if step % 2 == 1 else -5.0
            curved_points.append(torch.tensor([[float(step), offset_y]]))
        curved_traj = torch.stack(curved_points, dim=0)

        mean_curved, _ = compute_path_straightness(curved_traj)
        assert mean_curved < 0.6  # Significantly less than 1.0


class TestDivergenceAndCNF:
    """Verifies exact divergence, Hutchinson estimator, and CNF pipeline."""

    def test_exact_divergence_linear_field(self):
        # For v(x) = A x, div(v) = Tr(A)
        # Let A = [[2.0, 0.5], [1.0, 3.0]] => Tr(A) = 2.0 + 3.0 = 5.0
        A = torch.tensor([[2.0, 0.5], [1.0, 3.0]])

        def linear_v(x, t):
            return x @ A.T

        x = torch.randn(8, 2)
        t = torch.zeros(8, 1)

        div_exact = compute_exact_divergence(linear_v, x, t)
        assert div_exact.shape == (8, 1)
        assert torch.allclose(div_exact, torch.full((8, 1), 5.0), atol=1e-5)

    def test_hutchinson_divergence_expectation(self):
        A = torch.tensor([[1.5, 0.0], [0.0, 2.5]])  # Tr(A) = 4.0

        def linear_v(x, t):
            return x @ A.T

        x = torch.randn(100, 2)
        t = torch.zeros(100, 1)

        div_hutch = compute_hutchinson_divergence(linear_v, x, t, num_samples=50, noise_dist="rademacher")
        mean_div = div_hutch.mean().item()
        assert np.isclose(mean_div, 4.0, atol=0.2)

    def test_cnf_loss_and_sampling(self):
        dim = 2
        vnet = VelocityFieldNet(input_dim=dim, hidden_dim=32, num_layers=2)
        cnf = ContinuousNormalizingFlow(vnet, data_dim=dim, default_solver="euler")
        path = OptimalTransportCFMPath(use_ot=True)

        # 1. Compute loss
        x_target = torch.randn(16, dim)
        loss = cnf.compute_cfm_loss(x_target, path)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

        # 2. Generate samples
        samples, traj, nfe = cnf.sample(batch_size=10, num_steps=5, return_trajectory=True, seed=42)
        assert samples.shape == (10, dim)
        assert traj.shape == (6, 10, dim)
        assert nfe == 5

        # 3. Compute exact NLL
        log_p, nll = cnf.compute_log_prob(x_target[:4], num_steps=5, divergence_method="exact")
        assert log_p.shape == (4, 1)
        assert nll.shape == (4, 1)
        assert not torch.isnan(nll).any()


class TestMetrics:
    """Verifies Wasserstein metric on known distributions."""

    def test_sliced_wasserstein_identical_distributions(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200, 3)
        dist = compute_sliced_wasserstein(data, data)
        assert np.isclose(dist, 0.0, atol=1e-5)

    def test_sliced_wasserstein_shifted_distributions(self):
        rng = np.random.RandomState(42)
        data_a = rng.randn(200, 2)
        data_b = data_a + 5.0
        dist = compute_sliced_wasserstein(data_a, data_b)
        assert dist > 4.0
