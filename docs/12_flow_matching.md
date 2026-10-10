# Flow Matching

> **Stage 10 — Flow Matching**

## Overview

Flow matching is an alternative to diffusion that learns continuous-time probability flows. It constructs a vector field that transforms a simple distribution (e.g., Gaussian) into the data distribution.

## Topics Covered

### 1. Continuous Normalizing Flows
- Neural ODEs: dx/dt = v_θ(x, t)
- Transforming distributions continuously
- Instantaneous change of variables

### 2. Flow Matching Objective
- Conditional flow matching
- Simpler training than score-based diffusion
- Connection to optimal transport

### 3. Comparison with Diffusion
- DDPM vs. DDIM vs. Flow Matching
- Number of function evaluations
- Quality vs. computational cost

## Implementation

*To be completed in Stage 10.*

## References

*To be added.*
