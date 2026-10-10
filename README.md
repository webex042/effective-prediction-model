# EPM — Efficient Predictive Multimodal Generative Model

> **Investigating whether multimodal representation learning combined with efficient generative modeling can produce accurate predictions while reducing inference time, memory usage, computational complexity, and model size.**

---

## About

**EPM** is a research-oriented machine learning project designed as a progressive learning journey — from mathematical foundations through classical ML, deep learning, NLP, transformers, generative modeling, and efficiency optimization.

The project is structured as a series of clearly separated stages, each building on the previous one. Every algorithm is first understood mathematically, implemented from scratch, tested, then optimized with production-grade libraries.

### Why "EPM"?

- **Efficient** — We investigate model compression, efficient attention, quantization, pruning, and reduced sampling steps to minimize computational cost.
- **Predictive** — The core task is prediction (forecasting, classification, anomaly detection) rather than open-ended generation.
- **Multimodal** — We combine numerical, time-series, and textual data into unified representations.
- **Generative Model** — We explore generative frameworks (VAE, Diffusion, Flow Matching) not just for data generation, but as a principled approach to predictive modeling with uncertainty quantification.

---

## Core Research Question

> Can efficient generative and multimodal models achieve comparable or better predictive performance than conventional ML/DL models while significantly reducing inference time, memory consumption, FLOPs, and parameter count?

### Secondary Questions

1. How does model complexity affect prediction quality?
2. Can generative models model uncertainty better than deterministic predictors?
3. Can diffusion-based or flow-matching approaches reduce prediction uncertainty?
4. Can we reduce diffusion sampling steps without significantly degrading performance?
5. Can efficient attention reduce computational complexity?
6. Can knowledge distillation produce a smaller model with similar performance?
7. Can quantization and pruning reduce memory and latency?
8. Can one representation model work across numerical, time-series, and textual data?
9. How robust is the model under distribution shift?

---

## Development Stages

| Stage | Topic | Description | Status |
|-------|-------|-------------|--------|
| 0 | Environment | Project structure, config, logging, reproducibility | Completed |
| 1 | Mathematics | Linear algebra, calculus, probability, optimization | Completed |
| 2 | Classical ML | Linear/logistic regression, trees, SVM, boosting | Completed |
| 3 | Neural Networks | Forward/backward propagation from scratch | Completed |
| 4 | NLP | Tokenization, embeddings, attention mechanism | Completed |
| 5 | Time Series | LSTM, GRU, Transformer for sensor forecasting | Completed |
| 6 | Multimodal | Combining sensor + text modalities | Completed |
| 7 | Generative Models | AE, VAE, GAN concepts, diffusion intro | Completed |
| 8 | Diffusion | DDPM & DDIM accelerated sampling | Completed |
| 9 | Diffusion for Time Series | Probabilistic trajectory forecasting | Completed |
| 10 | Flow Matching | Continuous normalizing flows, comparison with diffusion | Completed |
| 11 | Efficiency | Efficient attention, complexity analysis | Completed |
| 12 | Compression | Distillation, quantization, pruning | Completed |
| 13 | Research Experiment | Reproducible H1 study: reduced-step generative forecasting vs real baselines | Completed |

**Progression rule:** Each stage is completed (learned, implemented, tested, benchmarked, documented) before advancing.

---

## Stage 13 — Reproducible Research Experiment (H1)

Stage 13 turns the earlier building blocks into a controlled, reproducible study of the project's primary hypothesis:

> **H1:** a reduced-step generative (diffusion) forecaster can retain competitive forecasting accuracy while cutting inference cost versus many-step diffusion sampling.

**How integrity is preserved**
- **Real models only.** Baselines (least-squares linear AR, LSTM, Transformer) are genuinely trained on the same data — never "ground truth + noise".
- **No leakage.** Chronological split (no shuffling) and the standardizer is fit on **training windows only**.
- **Clean H1 test.** The *same* trained diffusion weights are sampled at decreasing DDIM step counts (NFE = 100 → 50 → 20 → 10 → 5), so only inference cost changes.
- **Variability.** Multiple seeds → mean ± std with 95% bootstrap confidence intervals.
- Everything is persisted to an **experiment registry** (`experiments/registry/`) and an automated Markdown report (`docs/stage13_research_report.md`).

**Run it**

```bash
python demo_stage13.py           # full experiment (trains real models on CPU; a few minutes)
python demo_stage13.py --quick   # fast smoke configuration
pytest tests/test_research.py -v  # 21 Stage 13 unit tests
```

**Recorded finding (seeds 0–1, config fingerprint `552a2214`; see the report for CIs):** full 100-step diffusion reaches RMSE ≈ 2.82 (better than every deterministic baseline here, best being linear at ≈ 4.53). Reducing to 50/20/10 steps stays **within the full-step RMSE confidence interval** while running **1.9× / 4.6× / 8.4× faster**; at 5 steps accuracy collapses (+84 % RMSE). So **H1 holds down to ~10 steps on this dataset and breaks at 5** — a nuanced, non-fabricated result. Numbers come from real runs and will vary with hardware/seeds.

---

---

## Project Structure

```
EPM/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
├── .gitignore
├── configs/
│   └── default.yaml          # Default configuration
├── data/
│   ├── raw/                  # Unprocessed datasets
│   ├── processed/            # Cleaned/transformed data
│   └── README.md             # Data documentation
├── notebooks/                # Educational Jupyter notebooks
├── src/                      # Source code
│   ├── config.py             # Configuration system
│   ├── logger.py             # Structured logging
│   ├── reproducibility.py    # Seed management & environment
│   ├── experiment.py         # Experiment tracking
│   ├── data/                 # Data loading utilities
│   ├── preprocessing/        # Data preprocessing
│   ├── classical_ml/         # Stage 2: Classical ML models
│   ├── neural_networks/      # Stage 3: Neural networks
│   ├── nlp/                  # Stage 4: NLP pipeline
│   ├── transformers_/        # Stage 4-6: Transformer models
│   ├── timeseries/           # Stage 5: Time-series models
│   ├── diffusion/            # Stage 8-9: Diffusion models
│   ├── flow_matching/        # Stage 10: Flow matching
│   ├── efficiency/           # Stage 11: Computational efficiency
│   ├── multimodal/           # Stage 6: Multimodal fusion
│   ├── compression/          # Stage 12: Model compression
│   ├── research/             # Stage 13: Research experiment pipeline (config, registry, metrics, report)
│   └── evaluation/           # Metrics & benchmarking
├── models/                   # Saved model checkpoints
├── experiments/              # Experiment logs & results (registry/ holds Stage 12/13 records)
├── benchmarks/               # Benchmark results (incl. test_status.json, stage13_research.json)
├── tests/                    # Unit tests
├── docs/                     # Educational documentation + stage13_research_report.md
├── index.html, app.js, styles.css   # Static monochrome retro dashboard (Vercel + run.py --serve)
└── api/                      # Zero-dependency serverless API serving real artifact data
```

---

## Development Philosophy

For every important algorithm:

1. **Explain** — Intuition, then mathematics
2. **Implement from scratch** — NumPy/Python, no hiding behind libraries
3. **Test** — Verify correctness (numerical gradient checking, etc.)
4. **Optimize** — PyTorch implementation
5. **Compare** — Speed, memory, correctness
6. **Benchmark** — Systematic evaluation

---

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (281 tests)
pytest tests/ -v

# Interactive Runner (menu for all stages & demos)
python run.py

# Launch Vercel-ready Interactive Web Dashboard locally
python run.py --serve
```

---

## Vercel Deployment

The project is fully configured for deployment on [Vercel](https://vercel.com):

```bash
# Deploy with Vercel CLI
vercel
```

- **Configuration:** [vercel.json](file:///c:/Users/prans/OneDrive/Desktop/Project-1/vercel.json)
- **Zero-Dependency Serverless Functions:** [api/index.py](file:///c:/Users/prans/OneDrive/Desktop/Project-1/api/index.py)
- **Interactive Web Dashboard:** [index.html](file:///c:/Users/prans/OneDrive/Desktop/Project-1/index.html) with Vanilla JS client engines and publication figure gallery.


---

## Metrics Tracked

### Predictive
Accuracy, Precision, Recall, F1, AUROC, AUPRC, MAE, RMSE

### Generative
Likelihood, sample quality, calibration, uncertainty quality

### Computational
Training time, inference time, peak RAM, GPU memory, parameter count, FLOPs, sampling steps, model size (MB)

---

## Scope & Limitations

- This project uses **controlled, small-to-medium datasets** — not internet-scale training.
- We do **not** claim to build a system that "predicts anything."
- Prediction tasks are **clearly defined**: classification, regression, forecasting, anomaly detection.
- All experimental claims are **backed by measured results**, not theoretical assumptions.
- The project is designed for **learning and research**, not production deployment.

---

## License

This project is for educational and research purposes.

---

## Author

Research project developed as a progressive ML/DL learning journey.
