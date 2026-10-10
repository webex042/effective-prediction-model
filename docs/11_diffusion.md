# Diffusion Models

> **Stage 8–9 — Diffusion**

## Overview

Diffusion models are a class of generative models that learn to reverse a gradual noising process. They have achieved state-of-the-art results in image generation and are being adapted for other domains.

## Topics Covered

### 1. Forward Process
- Gradually add Gaussian noise: x₀ → x₁ → ... → xₜ
- q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)·xₜ₋₁, βₜ·I)
- Noise schedule: β₁, β₂, ..., βₜ

### 2. Reverse Process
- Learn to denoise: p_θ(xₜ₋₁ | xₜ)
- Neural network predicts noise (or signal)
- Training objective: simplified denoising score matching

### 3. DDPM vs. DDIM
- DDPM: stochastic sampling (many steps)
- DDIM: deterministic sampling (fewer steps)
- Trade-off: quality vs. speed

### 4. Diffusion for Prediction
- NOT the same as "faster prediction"
- Generative framework for modeling distributions over outputs
- Uncertainty quantification through multiple samples

## Implementation

*To be completed in Stage 8–9.*

## References

*To be added.*
