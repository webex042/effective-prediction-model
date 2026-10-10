# Model Compression

> **Stage 12 — Model Compression**

## Overview

Model compression techniques reduce model size, memory usage, and inference latency while attempting to preserve predictive performance.

## Topics Covered

### 1. Knowledge Distillation
- Teacher-student framework
- Soft labels and temperature scaling
- Loss: α·L_hard + (1-α)·T²·KL(soft_student || soft_teacher)

### 2. Quantization
- FP32 → FP16 → INT8
- Post-training quantization vs. quantization-aware training
- Measuring actual speedup (not just theoretical)

### 3. Pruning
- Unstructured pruning (individual weights)
- Structured pruning (entire channels/heads)
- Magnitude-based vs. learned pruning
- Caveat: sparsity ≠ automatic speedup without hardware support

## Implementation

*To be completed in Stage 12.*

## References

*To be added.*
