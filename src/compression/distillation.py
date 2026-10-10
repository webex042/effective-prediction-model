"""
EPM Stage 12 — Knowledge Distillation Module
=============================================

Implements the Teacher-Student distillation framework for both classification
and continuous regression / forecasting tasks.

MATHEMATICAL FOUNDATION
-----------------------
1. Soft Target Distillation (Hinton et al., 2015):
   Given student logits z_S and teacher logits z_T, temperatures T > 0:
     p_S(i; T) = exp(z_S_i / T) / sum_j exp(z_S_j / T)
     p_T(i; T) = exp(z_T_i / T) / sum_j exp(z_T_j / T)

   The composite loss is:
     L_KD = alpha * L_hard(y, y_hat_S) + (1 - alpha) * T^2 * KL(p_S(T) || p_T(T))

   The factor T^2 compensates for the gradients scaling as 1/T^2 when T is large.
   In the limit T -> infinity, the gradient matches MSE on zero-mean logits!

2. Continuous Regression / Representation Distillation:
   For regression (time series / sensor forecasting), we align continuous predictions:
     L_KD_reg = alpha * MSE(y, y_hat_S) + (1 - alpha) * MSE(y_hat_S, y_hat_T)

3. Feature / Hint Distillation (FitNets, Romero et al., 2014):
   Aligns intermediate hidden feature maps with a projection layer W_hint:
     L_hint = 0.5 * || W_hint * h_S - h_T ||_2^2
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined Hard + Soft distillation loss with temperature scaling.

    Supports:
      - 'classification': CrossEntropy hard loss + KL-Divergence soft loss
      - 'regression': MSE hard loss + MSE soft target alignment

    Parameters
    ----------
    temperature : float, default=4.0
        Temperature scaling factor T for softening probability distributions.
    alpha : float, default=0.3
        Weight for the hard task loss (1 - alpha is assigned to the teacher distillation loss).
    task : str, default='classification'
        'classification' or 'regression'.
    feature_weight : float, default=0.0
        Weight for optional intermediate feature hint loss.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        task: str = "classification",
        feature_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.task = task.lower()
        self.feature_weight = float(feature_weight)

        if self.task not in ("classification", "regression"):
            raise ValueError(f"Unsupported task: {task}. Must be 'classification' or 'regression'")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        feature_projector: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite distillation loss.

        Returns
        -------
        dict with keys: 'total_loss', 'hard_loss', 'soft_loss', 'hint_loss'
        """
        if self.task == "classification":
            # Hard cross-entropy loss against ground truth labels
            hard_loss = F.cross_entropy(student_logits, targets)

            # Soft probability distillation
            # log_softmax on student at temperature T
            log_p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            # softmax on teacher at temperature T (detached to avoid backprop through teacher)
            p_teacher = F.softmax(teacher_logits.detach() / self.temperature, dim=-1)

            # KL divergence: KL(p_teacher || p_student) = sum p_T * (log p_T - log p_S)
            # PyTorch kl_div(input=log_p, target=p) calculates target * (log target - input)
            kl_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
            soft_loss = kl_loss * (self.temperature ** 2)

        else:  # regression
            hard_loss = F.mse_loss(student_logits, targets)
            soft_loss = F.mse_loss(student_logits, teacher_logits.detach())

        # Intermediate feature hint loss (optional)
        hint_loss = torch.tensor(0.0, device=student_logits.device)
        if (
            self.feature_weight > 0.0
            and student_features is not None
            and teacher_features is not None
        ):
            s_feat = student_features
            if feature_projector is not None:
                s_feat = feature_projector(s_feat)
            hint_loss = F.mse_loss(s_feat, teacher_features.detach())

        total_loss = (
            self.alpha * hard_loss
            + (1.0 - self.alpha) * soft_loss
            + self.feature_weight * hint_loss
        )

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "hint_loss": hint_loss,
        }


class FeatureHintProjector(nn.Module):
    """Linear projection to match student hidden dimension with teacher hidden dimension."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationTrainer:
    """
    Standardized trainer for distilling a pre-trained Teacher model into a lightweight Student.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: KnowledgeDistillationLoss,
        feature_projector: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.feature_projector = feature_projector.to(device) if feature_projector else None
        self.device = device

        # Teacher is always frozen
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.student.train()
        total_loss, total_hard, total_soft = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch["input"].to(self.device), batch["target"].to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = self.teacher(inputs)
                if isinstance(teacher_out, tuple):
                    teacher_logits, teacher_feats = teacher_out[0], teacher_out[1]
                else:
                    teacher_logits, teacher_feats = teacher_out, None

            student_out = self.student(inputs)
            if isinstance(student_out, tuple):
                student_logits, student_feats = student_out[0], student_out[1]
            else:
                student_logits, student_feats = student_out, None

            loss_dict = self.criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                student_features=student_feats,
                teacher_features=teacher_feats,
                feature_projector=self.feature_projector,
            )

            loss_dict["total_loss"].backward()
            self.optimizer.step()

            total_loss += loss_dict["total_loss"].item()
            total_hard += loss_dict["hard_loss"].item()
            total_soft += loss_dict["soft_loss"].item()
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            "loss": total_loss / denom,
            "hard_loss": total_hard / denom,
            "soft_loss": total_soft / denom,
        }

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        task: str = "classification",
    ) -> Dict[str, float]:
        self.student.eval()
        total_correct = 0
        total_samples = 0
        total_mse = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs, targets = batch["input"].to(self.device), batch["target"].to(self.device)

                student_out = self.student(inputs)
                logits = student_out[0] if isinstance(student_out, tuple) else student_out

                if task == "classification":
                    preds = torch.argmax(logits, dim=-1)
                    total_correct += (preds == targets).sum().item()
                    total_samples += targets.size(0)
                else:
                    total_mse += F.mse_loss(logits, targets, reduction="sum").item()
                    total_samples += targets.numel()

        if task == "classification":
            acc = total_correct / max(total_samples, 1)
            return {"accuracy": acc}
        else:
            mse = total_mse / max(total_samples, 1)
            return {"rmse": math.sqrt(mse), "mse": mse}
