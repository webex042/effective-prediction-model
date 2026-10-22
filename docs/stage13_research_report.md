# Stage 13 Research Report — h1_reduced_step_forecasting

**Fingerprint:** `552a2214`  
**Hypothesis (H1):** A reduced-step generative model can maintain competitive forecasting performance while reducing inference latency and computational cost compared with conventional (many-step) diffusion sampling.

## 1. Research Question
Can a reduced-step generative (diffusion) forecaster retain competitive forecasting accuracy while reducing inference latency and the number of function evaluations (NFE) relative to many-step diffusion sampling?

## 2. Experimental Setup
- **Dataset:** synthetic_industrial_telemetry (v1), 1600 timesteps @ 10 min
- **Window / horizon:** 24 → 12 (target channel index 0)
- **Split (time-aware, no shuffle):** train=1095, val=235, test=235 windows
- **Seeds:** [0, 1] (metrics are mean ± std with bootstrap CI)
- **Diffusion:** cosine schedule, T=100, hidden=96, MC samples=16, eta=0.3
- **Environment:** Python 3.13.7, torch 2.14.1+cpu, device=cpu

## 3. Baselines (all genuinely trained on identical splits)
| Model | Params | Size (KB) | MAE (°, ↓) | RMSE (°, ↓) | Latency ms/win (↓) |
|---|---|---|---|---|---|
| linear | 1,452 | 5.67 | 3.543 ± 0.384 (95% CI [3.159, 3.927]) | 4.531 ± 0.496 (95% CI [4.035, 5.028]) | 0.000 ± 0.000 (95% CI [0.000, 0.000]) |
| lstm | 8,076 | 34.88 | 6.243 ± 0.704 (95% CI [5.539, 6.948]) | 7.498 ± 1.173 (95% CI [6.325, 8.671]) | 0.035 ± 0.001 (95% CI [0.034, 0.037]) |
| transformer | 17,676 | 143.55 | 7.226 ± 1.639 (95% CI [5.587, 8.865]) | 8.487 ± 1.385 (95% CI [7.102, 9.872]) | 0.105 ± 0.002 (95% CI [0.104, 0.107]) |

## 4. Diffusion NFE Sweep (same weights, varying DDIM steps)
| Steps (NFE) | RMSE (↓) | CRPS (↓) | PICP@90% (→0.90) | MPIW (↓) | Latency ms/win (↓) |
|---|---|---|---|---|---|
| 100 | 2.823 ± 0.329 (95% CI [2.494, 3.153]) | 1.752 ± 0.277 (95% CI [1.475, 2.029]) | 0.807 ± 0.076 (95% CI [0.731, 0.882]) | 9.960 ± 0.421 (95% CI [9.540, 10.381]) | 192.101 ± 8.461 (95% CI [183.641, 200.562]) |
| 50 | 2.753 ± 0.397 (95% CI [2.355, 3.150]) | 1.683 ± 0.239 (95% CI [1.445, 1.922]) | 0.878 ± 0.048 (95% CI [0.830, 0.926]) | 11.098 ± 0.707 (95% CI [10.391, 11.805]) | 103.349 ± 8.292 (95% CI [95.057, 111.640]) |
| 20 | 2.879 ± 0.466 (95% CI [2.413, 3.344]) | 1.794 ± 0.228 (95% CI [1.566, 2.022]) | 0.897 ± 0.052 (95% CI [0.845, 0.950]) | 13.855 ± 0.841 (95% CI [13.015, 14.696]) | 44.956 ± 3.250 (95% CI [41.705, 48.206]) |
| 10 | 2.998 ± 0.604 (95% CI [2.394, 3.603]) | 1.911 ± 0.204 (95% CI [1.706, 2.115]) | 0.932 ± 0.017 (95% CI [0.915, 0.948]) | 16.515 ± 1.703 (95% CI [14.813, 18.218]) | 23.140 ± 1.044 (95% CI [22.096, 24.184]) |
| 5 | 5.198 ± 0.204 (95% CI [4.994, 5.402]) | 3.367 ± 0.026 (95% CI [3.341, 3.393]) | 0.915 ± 0.012 (95% CI [0.903, 0.927]) | 26.009 ± 1.875 (95% CI [24.135, 27.884]) | 13.817 ± 0.564 (95% CI [13.253, 14.381]) |

## 5. H1 Efficiency Analysis
Reference (full) sampler: **100 steps**, RMSE=2.8235, latency=192.101 ms/window.

| Reduced steps | NFE reduction | Speedup | ΔRMSE vs full | Within full-step CI? |
|---|---|---|---|---|
| 50 | 2.0× | 1.86× | -2.51% | yes |
| 20 | 5.0× | 4.27× | +1.96% | yes |
| 10 | 10.0× | 8.3× | +6.18% | yes |
| 5 | 20.0× | 13.9× | +84.10% | no |

## 6. Interpretation
At least one reduced-step setting stays within the full-step RMSE confidence interval. The most aggressive such setting uses 10 steps (10.0× fewer NFE, 8.3× faster) with an RMSE change of +6.18% — consistent with H1 on this dataset. Settings that fall outside the CI mark where accuracy degrades.

## 7. Limitations
- Synthetic industrial telemetry (not a real sensor archive); conclusions are about the modeling trade-off, not a specific machine.
- Timing measured on CPU; GPU latency ratios will differ.
- Single architecture family; no hyperparameter search on the test set.

## 8. Reproduction
```bash
python demo_stage13.py            # full experiment (several minutes, CPU)
python -m pytest tests/test_research.py -q
```
Config fingerprint `552a2214` pins dataset, splits, seeds, and model/sampling settings. Registry records: 20261022_070943_s13_linear_552a2214, 20261022_070943_s13_lstm_552a2214, 20261022_070943_s13_transformer_552a2214, 20261022_070943_s13_ddim100_552a2214, 20261022_070943_s13_ddim50_552a2214, 20261022_070943_s13_ddim20_552a2214, 20261022_070943_s13_ddim10_552a2214, 20261022_070943_s13_ddim5_552a2214.
