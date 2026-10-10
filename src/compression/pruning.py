"""
EPM Stage 12 — Model Pruning Module
====================================

Implements magnitude-based unstructured pruning, structured neuron/channel pruning
with physical tensor compaction, and iterative pruning schedules.

MATHEMATICAL FOUNDATION
-----------------------
1. Unstructured Magnitude Pruning (Hagiwara, 1993; Han et al., 2015):
   Zero out the smallest individual weight parameters:
     M_ij = 1 if |W_ij| >= tau else 0
     W_pruned = W (elementwise) * M

   Sparsity is defined as:
     S = sum(1 - M) / total_parameters in [0, 1]

   Caveat: Unstructured sparsity creates random zero patterns. Without specialized
   sparse GEMM (SpMM) acceleration, wall-clock inference latency may not improve.

2. Structured Pruning (Li et al., 2016):
   Prunes entire rows (neurons / output channels) of weight matrices based on L1-norm:
     Score_i = sum_j |W_ij|
   Physically removes the lowest-scoring rows in layer l, and the corresponding columns
   in layer l+1. This physically shrinks tensor dimensions from (M, N) -> (M', N'),
   guaranteeing true wall-clock FLOP reduction and memory savings on standard hardware!

3. Iterative Pruning with Fine-Tuning (Lottery Ticket / Frankle & Carbin, 2018):
   Rather than pruning 70% in one shot (which can collapse capacity), iteratively:
     1. Prune delta_p %
     2. Fine-tune for E epochs
     3. Repeat until target sparsity is achieved.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def calculate_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Computes exact percentage of zero weights across model parameters.
    """
    total_params = 0
    zero_params = 0
    layer_sparsity = {}

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            numel = param.numel()
            zeros = (param == 0).sum().item()
            total_params += numel
            zero_params += zeros
            layer_sparsity[name] = float(zeros / max(numel, 1))

    global_sparsity = float(zero_params / max(total_params, 1))
    return {
        "global_sparsity": global_sparsity,
        "total_weight_elements": total_params,
        "zero_weight_elements": zero_params,
        "layer_sparsity": layer_sparsity,
    }


def prune_unstructured_magnitude(
    model: nn.Module,
    sparsity: float = 0.5,
    layer_types: Tuple = (nn.Linear,),
) -> nn.Module:
    """
    Applies magnitude-based unstructured pruning to all layers of specified types.
    Sets the bottom `sparsity` fraction of weights to exactly 0.0.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"Sparsity must be in [0, 1), got {sparsity}")

    if sparsity == 0.0:
        return model

    # Collect all candidate weights for global thresholding
    all_weights = []
    target_params = []
    for module in model.modules():
        if isinstance(module, layer_types) and hasattr(module, "weight"):
            w = module.weight.data
            all_weights.append(w.abs().view(-1))
            target_params.append(module.weight)

    if not all_weights:
        return model

    cat_weights = torch.cat(all_weights)
    k = int(sparsity * cat_weights.numel())
    if k == 0:
        return model

    threshold = torch.kthvalue(cat_weights, k).values.item()

    # Apply binary zeroing and collect masks
    masks = {}
    with torch.no_grad():
        for param in target_params:
            mask = (param.data.abs() > threshold).float()
            param.data.mul_(mask)
            masks[param] = mask

    return masks


class StructuredMLPCompressor:
    """
    Physically prunes hidden layers of sequential Multi-Layer Perceptrons (MLPs).
    Shrinks weight dimensions and reconstructs a compact model.
    """

    @staticmethod
    def prune_mlp_hidden_layer(
        fc1: nn.Linear,
        fc2: nn.Linear,
        keep_ratio: float = 0.7,
    ) -> Tuple[nn.Linear, nn.Linear]:
        """
        Prunes neurons between fc1 and fc2 based on L1-norm of fc1 output weights.

        fc1: (H, in_dim)
        fc2: (out_dim, H)

        Returns
        -------
        compact_fc1: (H_new, in_dim)
        compact_fc2: (out_dim, H_new)
        """
        W1 = fc1.weight.data  # shape (H, in_dim)
        b1 = fc1.bias.data if fc1.bias is not None else None
        W2 = fc2.weight.data  # shape (out_dim, H)
        b2 = fc2.bias.data if fc2.bias is not None else None

        H = W1.shape[0]
        H_new = max(int(round(H * keep_ratio)), 1)
        if H_new >= H:
            return fc1, fc2

        # L1-norm ranking across output rows
        neuron_importance = W1.abs().sum(dim=1)  # shape (H,)
        top_indices = torch.topk(neuron_importance, k=H_new, largest=True).indices
        top_indices, _ = torch.sort(top_indices)

        # Compact fc1
        new_fc1 = nn.Linear(fc1.in_features, H_new, bias=b1 is not None)
        new_fc1.weight.data = W1[top_indices, :].clone()
        if b1 is not None:
            new_fc1.bias.data = b1[top_indices].clone()

        # Compact fc2 (slice corresponding input columns)
        new_fc2 = nn.Linear(H_new, fc2.out_features, bias=b2 is not None)
        new_fc2.weight.data = W2[:, top_indices].clone()
        if b2 is not None:
            new_fc2.bias.data = b2.clone()

        return new_fc1, new_fc2


def iterative_prune_and_finetune(
    model: nn.Module,
    train_fn,
    eval_fn,
    target_sparsity: float = 0.6,
    num_steps: int = 3,
    finetune_epochs: int = 2,
) -> Dict[str, Any]:
    """
    Performs iterative magnitude pruning with intermittent fine-tuning.

    Returns history of sparsity and evaluation metrics at each step.
    """
    history = []
    step_delta = target_sparsity / num_steps

    # Initial baseline evaluation
    init_metric = eval_fn(model)
    history.append({"step": 0, "sparsity": 0.0, "eval": init_metric})

    for step in range(1, num_steps + 1):
        target_step_sparsity = min(step * step_delta, target_sparsity)
        # Prune to target step sparsity
        masks = prune_unstructured_magnitude(model, sparsity=target_step_sparsity)

        # Fine-tune model to recover loss
        if finetune_epochs > 0:
            train_fn(model, epochs=finetune_epochs)
            # Reapply masks to preserve zeros after optimizer step
            if masks:
                with torch.no_grad():
                    for param, mask in masks.items():
                        param.data.mul_(mask)

        current_sp = calculate_sparsity(model)["global_sparsity"]
        metric = eval_fn(model)
        history.append({
            "step": step,
            "sparsity": current_sp,
            "eval": metric,
        })

    return {
        "final_sparsity": history[-1]["sparsity"],
        "history": history,
    }
