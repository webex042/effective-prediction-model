# Changelog

All notable changes to EPM are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-11-06

First tagged release. Stages 0–13 are complete: each stage was learned,
implemented, tested, benchmarked and documented before the next one started.

### Added

- **Stage 0 — Environment.** Project structure, configuration, logging and
  reproducibility controls.
- **Stage 1 — Mathematics.** Linear algebra, calculus, probability and
  optimisation, implemented from scratch with plots.
- **Stage 2 — Classical ML.** Linear and logistic regression, decision trees,
  SVM and boosting.
- **Stage 3 — Neural networks.** Forward and backward propagation written from
  scratch.
- **Stage 4 — NLP.** Tokenisation, embeddings and the attention mechanism.
- **Stage 5 — Time series.** LSTM, GRU and Transformer forecasters for sensor
  data.
- **Stage 6 — Multimodal.** Fusion of sensor and text modalities.
- **Stage 7 — Generative models.** Autoencoders, VAEs, GAN concepts and an
  introduction to diffusion.
- **Stage 8 — Diffusion.** DDPM training and DDIM accelerated sampling.
- **Stage 9 — Diffusion for time series.** Probabilistic trajectory forecasting.
- **Stage 10 — Flow matching.** Continuous normalising flows, compared against
  diffusion.
- **Stage 11 — Efficiency.** Efficient attention and complexity analysis.
- **Stage 12 — Compression.** Distillation, quantisation and pruning.
- **Stage 13 — Research experiment.** A reproducible H1 study with an experiment
  registry and an automated Markdown report.
- **Dataset registry** with 20 datasets across scikit-learn, UCI, OpenML, NAB
  and EPM's own synthetic generators, each documenting source and licence.
- **Fair benchmark runner** covering 44 benchmarkable algorithms on a shared
  split and seed, with CV / rolling-origin validation, latency, model size,
  task-specific metrics and a Pareto summary.
- **EPM Chat**, a no-LLM, no-network natural-language front end to the benchmark
  pipeline.
- **`epm_chatbot/`**, a standalone embeddable prediction service with origin
  allowlisting.
- **Interactive web dashboard** with a publication figure gallery, deployable to
  Vercel.
- **502 tests** covering preprocessing, metrics, reproducibility and every stage.
- 16 documentation chapters in `docs/`.

### Findings

- **H1 holds down to roughly 10 DDIM steps and breaks at 5.** Sampling the same
  trained diffusion weights at decreasing step counts (NFE 100 → 50 → 20 → 10 → 5)
  stays within the full-step RMSE confidence interval down to 10 steps while
  running up to 8.4× faster; at 5 steps RMSE degrades by about 84%.
- Full 100-step diffusion reached RMSE ≈ 2.82, ahead of every deterministic
  baseline tried here (best: linear AR at ≈ 4.53).
- Numbers come from real runs on seeds 0–1, config fingerprint `552a2214`, and
  will vary with hardware and seeds.

### Known limitations

- Controlled, small-to-medium datasets only; not internet-scale training.
- Designed for learning and research, not production deployment.
- The dashboard and local experiment API assume localhost and have no
  authentication.
