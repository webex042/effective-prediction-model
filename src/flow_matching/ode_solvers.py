"""
Numerical ODE Solvers for Continuous Probability Flows
======================================================
Solves initial value problems (IVP):
    dx/dt = v_theta(x, t),  x(t_0) = x_0

Provides explicit fixed-step numerical integration schemes:
    - Euler Method (1st order, 1 NFE / step)
    - Midpoint / RK2 Method (2nd order, 2 NFE / step)
    - Runge-Kutta 4 (RK4, 4th order, 4 NFE / step)

Includes trajectory recording and trajectory straightness (curvature) metrics.
"""

from typing import Callable, Tuple, List, Optional
import torch


def euler_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Explicit Euler single step."""
    v = v_fn(x, t)
    return x + dt * v


def midpoint_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Explicit Midpoint (Runge-Kutta 2nd Order) step."""
    half_dt = 0.5 * dt
    k1 = v_fn(x, t)
    t_mid = t + half_dt
    x_mid = x + half_dt * k1
    k2 = v_fn(x_mid, t_mid)
    return x + dt * k2


def rk4_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Classical Runge-Kutta 4th Order (RK4) step."""
    half_dt = 0.5 * dt
    k1 = v_fn(x, t)
    k2 = v_fn(x + half_dt * k1, t + half_dt)
    k3 = v_fn(x + half_dt * k2, t + half_dt)
    k4 = v_fn(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ODESolver:
    """
    ODE Integrator supporting Euler, Midpoint, and RK4 algorithms.

    Can integrate forward in time (t_start=0.0 -> t_end=1.0 for generation)
    or backward in time (t_start=1.0 -> t_end=0.0 for inversion/likelihood).
    """

    SUPPORTED_METHODS = ("euler", "midpoint", "rk4")

    def __init__(self, method: str = "euler"):
        method_lower = method.lower()
        if method_lower not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown solver method '{method}'. Choose from {self.SUPPORTED_METHODS}"
            )
        self.method = method_lower

    def integrate(
        self,
        v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        num_steps: int = 20,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Integrate the ODE dx/dt = v_fn(x, t) from t_start to t_end.

        Args:
            v_fn: Velocity function callable (x, t) -> v.
            x0: Initial state of shape (B, D).
            t_start: Starting integration time.
            t_end: Ending integration time.
            num_steps: Discretization step count.
            return_trajectory: Whether to store intermediate points.

        Returns:
            Tuple of:
                - x_final: Terminal state of shape (B, D).
                - trajectory: Tensor of shape (num_steps + 1, B, D) if return_trajectory, else None.
                - total_nfe: Total number of function evaluations performed.
        """
        dt = (t_end - t_start) / num_steps
        current_x = x0.clone()
        device = x0.device
        batch_size = x0.size(0)

        nfe_per_step = {
            "euler": 1,
            "midpoint": 2,
            "rk4": 4,
        }[self.method]
        total_nfe = nfe_per_step * num_steps

        trajectory_points = [current_x.clone()] if return_trajectory else None

        for step in range(num_steps):
            current_t_val = t_start + step * dt
            t_tensor = torch.full((batch_size, 1), current_t_val, device=device, dtype=x0.dtype)

            if self.method == "euler":
                current_x = euler_step(v_fn, current_x, t_tensor, dt)
            elif self.method == "midpoint":
                current_x = midpoint_step(v_fn, current_x, t_tensor, dt)
            elif self.method == "rk4":
                current_x = rk4_step(v_fn, current_x, t_tensor, dt)

            if return_trajectory:
                trajectory_points.append(current_x.clone())

        trajectory_tensor = torch.stack(trajectory_points, dim=0) if return_trajectory else None
        return current_x, trajectory_tensor, total_nfe


def compute_path_straightness(trajectory: torch.Tensor) -> Tuple[float, float]:
    """
    Compute straightness ratio S = Displacement / Arc_Length across an ensemble.

    For straight paths (Optimal Transport limit): S = 1.0.
    For curved or erratic stochastic paths: S < 1.0.

    Args:
        trajectory: Tensor of shape (K + 1, B, D) containing trajectory points over time.

    Returns:
        Tuple of (mean_straightness, std_straightness) across the batch.
    """
    if trajectory.dim() != 3:
        raise ValueError(f"Expected trajectory of shape (steps, batch, dim), got {trajectory.shape}")

    # Total Euclidean displacement between endpoints: ||x_K - x_0||_2
    endpoints_disp = torch.norm(trajectory[-1] - trajectory[0], p=2, dim=-1)  # (B,)

    # Total arc length: sum_{k=0}^{K-1} ||x_{k+1} - x_k||_2
    step_diffs = trajectory[1:] - trajectory[:-1]  # (K, B, D)
    step_lengths = torch.norm(step_diffs, p=2, dim=-1)  # (K, B)
    total_arc_length = torch.sum(step_lengths, dim=0)  # (B,)

    # Straightness ratio S in (0, 1]
    straightness = endpoints_disp / (total_arc_length + 1e-8)
    # Clamp to [0, 1] for numerical float edge cases
    straightness = torch.clamp(straightness, 0.0, 1.0)

    mean_s = float(straightness.mean().item())
    std_s = float(straightness.std().item()) if straightness.numel() > 1 else 0.0
    return mean_s, std_s
