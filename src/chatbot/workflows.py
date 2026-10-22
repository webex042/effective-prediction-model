"""
Phase 3 chatbot workflows: generation, generative-model comparison and model
compression on registry datasets.

Every workflow is *composed* from EPM's existing models and trainers:

* generation        → ``src.generative.models.VariationalAutoencoder`` /
                      ``StandardAutoencoder`` + ``train_generative_model``
* generative compare → ``src.flow_matching`` (VelocityFieldNet, ContinuousNormalizingFlow,
                      OptimalTransportCFMPath, train_flow_matching) and
                      ``src.diffusion`` (DiffusionSchedule, DenoisingMLP, GaussianDiffusion)
* compression       → ``src.compression`` (BenchmarkClassifier, DistillationTrainer,
                      KnowledgeDistillationLoss, quantize_model_dynamic_int8,
                      prune_unstructured_magnitude, StructuredMLPCompressor)

Integrity rules: all fidelity / accuracy numbers are computed on a held-out
split the models never saw; scaling statistics come from the training split;
every table includes reference baselines (train-vs-test distance, an independent
Gaussian) so a number can be judged; nothing is estimated or padded.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loaders import load_dataset
from src.data.registry import get_dataset_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ProgressFn = Callable[[float, str], None]


CLIP_X0 = 5.0  # data are standardised, so |x_0| <= 5 sd covers essentially all real rows


class WorkflowError(ValueError):
    """User-facing problem with a workflow request."""


def _noop(f: float, m: str) -> None:
    return None


def _run_id(kind: str, dataset_id: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{kind}_{dataset_id}"


def workflows_dir() -> Path:
    return PROJECT_ROOT / "experiments" / "workflows"


def _save(result: Dict[str, Any]) -> str:
    d = workflows_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{result['run_id']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return _rel(p)


def _rel(p: Path) -> str:
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _numeric_matrix(dataset_id: str, seed: int, max_rows: int, test_frac: float = 0.25):
    """Numeric feature matrix, train/test split, train-only standardisation."""
    spec = get_dataset_spec(dataset_id)
    ds = load_dataset(dataset_id, seed=seed, allow_large=False)
    num = [c for c in ds.numeric_features if c in ds.features.columns]
    if len(num) < 2:
        raise WorkflowError(f"'{spec.name}' has fewer than 2 numeric features; generative models need at least 2.")
    X = ds.features[num].astype(float)
    dropped = [c for c in ds.features.columns if c not in num]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    if len(idx) > max_rows:
        idx = idx[:max_rows]
    n_test = max(20, int(len(idx) * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    mu, sd = Xtr.mean().to_numpy(), Xtr.std(ddof=0).to_numpy()
    sd = np.where(sd < 1e-8, 1.0, sd)
    Ztr = ((Xtr.to_numpy() - mu) / sd).astype(np.float32)
    Zte = ((Xte.to_numpy() - mu) / sd).astype(np.float32)
    return spec, num, dropped, Ztr, Zte, mu, sd


def _sliced_w(a: np.ndarray, b: np.ndarray, seed: int) -> float:
    from src.flow_matching.benchmarks import compute_sliced_wasserstein

    return float(compute_sliced_wasserstein(a, b, num_projections=64, random_state=seed))


def _mmd(a: np.ndarray, b: np.ndarray) -> float:
    from src.diffusion.benchmarks import compute_multivariate_mmd

    gamma = 1.0 / max(a.shape[1], 1)
    return float(compute_multivariate_mmd(a, b, gamma=gamma))


def _reference_rows(Ztr: np.ndarray, Zte: np.ndarray, seed: int) -> List[Dict[str, Any]]:
    """Context for distances: real-vs-real floor and an independent-Gaussian baseline."""
    rng = np.random.default_rng(seed)
    n = len(Zte)
    real = Ztr[rng.choice(len(Ztr), size=min(n, len(Ztr)), replace=False)]
    gauss = rng.normal(Ztr.mean(0), Ztr.std(0) + 1e-8, size=(n, Ztr.shape[1])).astype(np.float32)
    return [
        {"model": "Reference: real training rows", "sliced_w": round(_sliced_w(real, Zte, seed), 4),
         "mmd": round(_mmd(real, Zte), 4), "note": "best achievable (sampling noise floor)"},
        {"model": "Baseline: independent Gaussian", "sliced_w": round(_sliced_w(gauss, Zte, seed), 4),
         "mmd": round(_mmd(gauss, Zte), 4), "note": "matches per-feature mean/std, ignores correlations"},
    ]


def _n_params(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# =============================================================================
# 1. Generation with a VAE (optionally compared with a plain AE)
# =============================================================================

def run_generation(
    dataset_id: str,
    num_samples: int = 100,
    latent_dim: Optional[int] = None,
    epochs: int = 80,
    seed: int = 42,
    compare_ae: bool = True,
    max_rows: int = 20000,
    beta: Optional[float] = None,
    progress: ProgressFn = _noop,
) -> Dict[str, Any]:
    import torch
    from scipy.stats import ks_2samp
    from torch.utils.data import DataLoader, TensorDataset

    from src.generative.models import StandardAutoencoder, VariationalAutoencoder, train_generative_model
    from src.reproducibility import set_seed

    if not 1 <= num_samples <= 10000:
        raise WorkflowError("num_samples must be between 1 and 10,000.")
    if not 1 <= epochs <= 500:
        raise WorkflowError("epochs must be between 1 and 500.")
    progress(0.05, "Loading and standardising data")
    spec, cols, dropped, Ztr, Zte, mu, sd = _numeric_matrix(dataset_id, seed, max_rows)
    d = Ztr.shape[1]
    latent = latent_dim or max(2, min(8, d // 2))
    if not 1 <= latent < d:
        raise WorkflowError(f"latent_dim must be between 1 and {d - 1} for {d} features.")
    hidden = [max(16, 4 * d), max(8, 2 * d)]
    # The Stage 7 VAE averages reconstruction MSE over features but sums KL over latent dims.
    # beta = 2*sigma^2/d makes that equal (up to a constant) to the standard ELBO with a Gaussian
    # decoder of std sigma; sigma = 0.5 (standardised units) -> beta = 0.5/d. beta = 1 collapses
    # the posterior on low-dimensional tabular data (KL -> 0, samples ~ constant).
    beta = beta if beta is not None else 0.5 / d
    if not 0 < beta <= 10:
        raise WorkflowError("beta must be in (0, 10].")
    set_seed(seed, deterministic=False)
    loader = DataLoader(TensorDataset(torch.from_numpy(Ztr)), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    progress(0.15, f"Training VAE ({d}-dim input → {latent}-dim latent)")
    vae = VariationalAutoencoder(input_dim=d, hidden_dims=hidden, latent_dim=latent, beta=beta, dropout=0.0)
    t0 = time.perf_counter()
    vae, hist = train_generative_model(vae, loader, epochs=epochs, lr=2e-3)
    train_s = time.perf_counter() - t0
    vae.eval()
    xt = torch.from_numpy(Zte)
    with torch.no_grad():
        mu_z, logvar = vae.encode(xt)
        recon = vae.decode(mu_z)  # posterior-mean reconstruction
        losses = vae.compute_loss(xt, recon, mu_z, logvar)
        torch.manual_seed(seed)
        t1 = time.perf_counter()
        gen = vae.sample_prior(num_samples=max(num_samples, len(Zte))).numpy()
        sample_ms = (time.perf_counter() - t1) * 1000 / max(num_samples, len(Zte))
    gen_eval = gen[: len(Zte)]
    progress(0.7, "Scoring synthetic samples against held-out rows")

    rows = [{"model": "VAE (Stage 7)", "sliced_w": round(_sliced_w(gen_eval, Zte, seed), 4),
             "mmd": round(_mmd(gen_eval, Zte), 4), "note": f"{num_samples} samples from p(z)=N(0,I)"}]
    rows += _reference_rows(Ztr, Zte, seed)

    # per-feature fidelity in original units
    real_te = Zte * sd + mu
    syn = gen[:num_samples] * sd + mu
    feats = []
    for j, c in enumerate(cols):
        p = ks_2samp(real_te[:, j], (gen_eval * sd + mu)[:, j]).pvalue
        feats.append({"feature": c, "real_mean": round(float(real_te[:, j].mean()), 4),
                      "synth_mean": round(float(syn[:, j].mean()), 4),
                      "real_std": round(float(real_te[:, j].std()), 4),
                      "synth_std": round(float(syn[:, j].std()), 4), "ks_p": round(float(p), 4)})
    corr_err = float(np.abs(np.corrcoef(Zte, rowvar=False) - np.corrcoef(gen_eval, rowvar=False)).mean()) if d > 1 else None

    ae_info = None
    if compare_ae:
        progress(0.8, "Training plain autoencoder for comparison")
        ae = StandardAutoencoder(input_dim=d, hidden_dims=hidden, latent_dim=latent, dropout=0.0)
        ae, _ = train_generative_model(ae, loader, epochs=epochs, lr=2e-3)
        ae.eval()
        with torch.no_grad():
            ae_recon, _ = ae(xt)
        ae_info = {"recon_mse": round(float(((ae_recon - xt) ** 2).mean()), 5), "params": _n_params(ae),
                   "note": "A plain AE has no prior over its latent space, so it cannot sample new data; "
                           "it is shown for reconstruction quality only."}

    run_id = _run_id("generate", dataset_id)
    out_csv = workflows_dir() / f"{run_id}_samples.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(syn, columns=cols).to_csv(out_csv, index=False)
    result = {
        "kind": "generate", "run_id": run_id, "dataset_id": dataset_id, "dataset_name": spec.name,
        "model": "vae", "latent_dim": latent, "input_dim": d, "epochs": epochs, "seed": seed, "beta": round(beta, 5),
        "n_train": int(len(Ztr)), "n_test": int(len(Zte)), "num_samples": num_samples,
        "dropped_columns": dropped,
        "vae": {"recon_mse_test": round(float(losses["recon_loss"]), 5), "kl_test": round(float(losses["kl_loss"]), 4),
                "loss_test": round(float(losses["loss"]), 5), "final_train_loss": round(float(hist[-1]), 5),
                "params": _n_params(vae), "train_time_s": round(train_s, 3), "sample_ms": round(sample_ms, 4)},
        "distribution": rows, "features": feats,
        "mean_abs_corr_error": None if corr_err is None else round(corr_err, 4),
        "ae": ae_info, "samples_csv": _rel(out_csv),
        "samples_preview": [[round(float(v), 4) for v in r] for r in syn[:5]], "columns": cols,
        "notes": [
            "Distances are computed in standardised units against held-out rows the models never saw.",
            f"VAE loss = per-feature reconstruction MSE + β·KL with β={beta:.4g} (= 0.5/d, i.e. a Gaussian decoder with "
            "σ=0.5 in standardised units); β=1 collapses the posterior on low-dimensional tables. "
            "The loss is a surrogate, not an exact log-likelihood.",
            "KS p-values compare each marginal separately; small p-values flag features the VAE reproduces poorly.",
        ],
    }
    result["saved_to"] = _save(result)
    progress(1.0, "Done")
    return result


# =============================================================================
# 2. Diffusion vs flow matching (held-out evaluation)
# =============================================================================

def run_generative_comparison(
    dataset_id: str = "epm_sensor_telemetry",
    epochs: int = 40,
    seed: int = 42,
    max_rows: int = 3000,
    n_eval: int = 500,
    progress: ProgressFn = _noop,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.diffusion.gaussian_diffusion import GaussianDiffusion
    from src.diffusion.models import DenoisingMLP
    from src.diffusion.schedules import DiffusionSchedule
    from src.flow_matching.continuous_normalizing_flow import ContinuousNormalizingFlow
    from src.flow_matching.benchmarks import train_flow_matching
    from src.flow_matching.models import VelocityFieldNet
    from src.flow_matching.ode_solvers import compute_path_straightness
    from src.flow_matching.probability_paths import OptimalTransportCFMPath
    from src.reproducibility import set_seed

    if not 1 <= epochs <= 300:
        raise WorkflowError("epochs must be between 1 and 300.")
    progress(0.03, "Preparing held-out split")
    spec, cols, dropped, Ztr, Zte, mu, sd = _numeric_matrix(dataset_id, seed, max_rows)
    Zte = Zte[:n_eval]
    d = Ztr.shape[1]
    n = len(Zte)
    set_seed(seed, deterministic=False)
    loader = DataLoader(TensorDataset(torch.from_numpy(Ztr)), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    rows: List[Dict[str, Any]] = []

    def record(name, family, nfe, samples, ms, params, traj=None):
        s = samples.detach().numpy() if hasattr(samples, "detach") else samples
        straight = None
        if traj is not None:
            straight = round(float(compute_path_straightness(traj)[0]), 3)
        rows.append({"model": name, "family": family, "nfe": nfe, "sliced_w": round(_sliced_w(s, Zte, seed), 4),
                     "mmd": round(_mmd(s, Zte), 4), "straightness": straight,
                     "latency_ms_per_sample": round(ms, 4), "params": params})

    progress(0.08, "Training OT-CFM flow matching")
    vnet = VelocityFieldNet(input_dim=d, hidden_dim=96, num_layers=3)
    cnf = ContinuousNormalizingFlow(vnet, data_dim=d, default_solver="euler")
    t0 = time.perf_counter()
    train_flow_matching(cnf, OptimalTransportCFMPath(use_ot=True), loader, epochs=epochs, lr=2e-3)
    flow_train_s = time.perf_counter() - t0
    for steps in (20, 10):
        t0 = time.perf_counter()
        smp, traj, nfe = cnf.sample(batch_size=n, num_steps=steps, solver_method="euler", return_trajectory=True, seed=seed)
        record(f"OT-CFM flow matching (Euler, {steps} steps)", "flow matching", int(nfe), smp,
               (time.perf_counter() - t0) * 1000 / n, _n_params(cnf), traj)

    progress(0.5, "Training DDPM/DDIM diffusion")
    sched = DiffusionSchedule(timesteps=100, schedule_type="cosine")
    net = DenoisingMLP(input_dim=d, hidden_dim=96, num_blocks=3)
    diff = GaussianDiffusion(schedule=sched)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        for (bx,) in loader:
            opt.zero_grad()
            loss = diff.compute_training_loss(net, bx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if ep % 10 == 0:
            progress(0.5 + 0.3 * ep / epochs, f"Diffusion epoch {ep + 1}/{epochs}")
    diff_train_s = time.perf_counter() - t0
    net.eval()
    progress(0.82, "Sampling DDPM (100 steps) and DDIM (20 / 10 steps)")
    torch.manual_seed(seed)
    with torch.no_grad():
        t0 = time.perf_counter()
        smp, traj = diff.sample_ddpm(net, (n, d), return_trajectory=True, clip_x0=CLIP_X0)
        record("DDPM diffusion (100 steps)", "diffusion", 100, smp, (time.perf_counter() - t0) * 1000 / n,
               _n_params(net), torch.stack(traj, 0))
        for steps in (20, 10):
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            smp, traj = diff.sample_ddim(net, (n, d), num_steps=steps, eta=0.0, return_trajectory=True,
                                             clip_x0=CLIP_X0)
            record(f"DDIM diffusion ({steps} steps)", "diffusion", steps, smp, (time.perf_counter() - t0) * 1000 / n,
                   _n_params(net), torch.stack(traj, 0))

    refs = _reference_rows(Ztr, Zte, seed)
    rows.sort(key=lambda r: r["sliced_w"])
    result = {
        "kind": "compare_generative", "run_id": _run_id("generative", dataset_id), "dataset_id": dataset_id,
        "dataset_name": spec.name, "input_dim": d, "columns": cols, "epochs": epochs, "seed": seed,
        "n_train": int(len(Ztr)), "n_eval": n, "models": rows, "references": refs,
        "train_time_s": {"flow_matching": round(flow_train_s, 2), "diffusion": round(diff_train_s, 2)},
        "notes": [
            "Both families use the same data, epochs, batch size and comparable networks (hidden 96, 3 blocks).",
            "Sliced-Wasserstein and MMD are measured against held-out rows; compare them with the reference rows.",
            "Straightness = endpoint displacement / path length of the sampling trajectory (1.0 = straight line).",
            "Rows of a time series are treated as independent vectors here (no temporal conditioning).",
            "Diffusion samplers clip the predicted x_0 to ±5 standard deviations (reference DDPM 'clip_denoised'); "
            "without it the cosine schedule's final alpha_bar (~2e-7) amplifies noise-prediction error ~2000x.",
            "No likelihood (NLL) is reported: the Stage 10 benchmark's NLL values for the 5-step and Gaussian-CFM "
            "variants are offsets of the OT-CFM value, not measurements, so they are not reused.",
        ],
    }
    result["saved_to"] = _save(result)
    progress(1.0, "Done")
    return result


# =============================================================================
# 3. Compression of a classifier trained on a registry dataset
# =============================================================================

def run_compression(
    dataset_id: str,
    epochs: int = 30,
    seed: int = 42,
    max_rows: int = 30000,
    progress: ProgressFn = _noop,
) -> Dict[str, Any]:
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset

    from src.compression.benchmarks import BenchmarkClassifier, count_parameters, measure_inference_latency
    from src.compression.distillation import DistillationTrainer, KnowledgeDistillationLoss
    from src.compression.pruning import StructuredMLPCompressor, calculate_sparsity, prune_unstructured_magnitude
    from src.compression.quantization import get_model_size_kb, quantize_model_dynamic_int8
    from src.evaluation.task_metrics import classification_metrics
    from src.preprocessing.pipelines import build_tabular_preprocessor, split_tabular
    from src.reproducibility import set_seed

    spec = get_dataset_spec(dataset_id)
    if not spec.supports("classification"):
        raise WorkflowError(f"Compression is demonstrated on classifiers; '{spec.name}' is not a classification dataset. "
                            "Try breast_cancer, wine, digits, iris or adult.")
    if not 1 <= epochs <= 300:
        raise WorkflowError("epochs must be between 1 and 300.")
    progress(0.05, "Loading data and fitting preprocessing on the training split")
    ds = load_dataset(dataset_id, seed=seed)
    y = LabelEncoder().fit_transform(ds.target)
    split = split_tabular(len(ds.features), pd.Series(y), "stratified", 0.25, 0.0, seed, max_rows)
    pre = build_tabular_preprocessor(ds.numeric_features, ds.categorical_features, True)
    Xtr = pre.fit_transform(ds.features.iloc[split.train_idx]).astype(np.float32)
    Xte = pre.transform(ds.features.iloc[split.test_idx]).astype(np.float32)
    ytr, yte = y[split.train_idx], y[split.test_idx]
    n_cls = int(len(np.unique(y)))
    in_dim = Xtr.shape[1]
    set_seed(seed, deterministic=False)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))),
                              batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    xte = torch.from_numpy(Xte)
    sample = xte[:1]

    class Single(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = copy.deepcopy(m)

        def forward(self, x):
            out = self.model(x)
            return out[0] if isinstance(out, tuple) else out

    def fit_plain(model, label):
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                logits, _ = model(xb)
                F.cross_entropy(logits, yb).backward()
                opt.step()
        return model

    rows: List[Dict[str, Any]] = []

    def evaluate(name, technique, model, params, sparsity=0.0):
        model.eval()
        with torch.no_grad():
            out = model(xte)
            logits = out[0] if isinstance(out, tuple) else out
            proba = torch.softmax(logits, dim=1).numpy()
        pred = proba.argmax(1)
        m, _ = classification_metrics(yte, pred, proba, classes=list(range(n_cls)))
        lat = measure_inference_latency(model, sample, warmup=10, reps=100)["latency_us_per_sample"]
        rows.append({"variant": name, "technique": technique, "accuracy": m["accuracy"], "f1": m["f1"],
                     "size_kb": round(get_model_size_kb(model), 2), "latency_us": round(lat, 2),
                     "params": params, "sparsity": round(sparsity, 3)})

    progress(0.15, "Training teacher [128,128,64]")
    teacher = fit_plain(BenchmarkClassifier(in_dim, [128, 128, 64], n_cls), "teacher")
    evaluate("Teacher (FP32)", "full-size MLP", teacher, count_parameters(teacher))
    progress(0.35, "Training student [32,16] from scratch")
    student = fit_plain(BenchmarkClassifier(in_dim, [32, 16], n_cls), "student")
    evaluate("Student from scratch (FP32)", "small MLP, no teacher", student, count_parameters(student))
    progress(0.5, "Distilling teacher → student (T=4, α=0.3)")
    distilled = BenchmarkClassifier(in_dim, [32, 16], n_cls)
    trainer = DistillationTrainer(teacher=teacher, student=distilled,
                                  optimizer=torch.optim.Adam(distilled.parameters(), lr=0.003),
                                  criterion=KnowledgeDistillationLoss(temperature=4.0, alpha=0.3))
    for _ in range(epochs):
        trainer.train_epoch(train_loader)
    evaluate("Distilled student (FP32)", "knowledge distillation", distilled, count_parameters(distilled))
    progress(0.7, "INT8 dynamic quantisation and pruning")
    q = quantize_model_dynamic_int8(Single(distilled))
    evaluate("Distilled + INT8", "dynamic post-training quantisation", q, count_parameters(distilled))
    pruned = copy.deepcopy(distilled)
    prune_unstructured_magnitude(pruned, sparsity=0.5)
    evaluate("Distilled + 50% unstructured pruning", "global magnitude pruning (dense storage)", pruned,
             count_parameters(pruned), calculate_sparsity(pruned)["global_sparsity"])
    struct = copy.deepcopy(distilled)
    fc0, fc1 = struct.backbone[0], struct.backbone[2]
    struct.backbone[0], struct.backbone[2] = StructuredMLPCompressor.prune_mlp_hidden_layer(fc0, fc1, keep_ratio=0.625)
    evaluate("Distilled + structured pruning (32→20)", "L1 neuron pruning", struct, count_parameters(struct))

    base = rows[0]
    for r in rows:
        r["size_reduction"] = round(base["size_kb"] / r["size_kb"], 2) if r["size_kb"] else None
        r["accuracy_retention"] = round(r["accuracy"] / base["accuracy"], 4) if base["accuracy"] else None
    result = {
        "kind": "compress", "run_id": _run_id("compress", dataset_id), "dataset_id": dataset_id,
        "dataset_name": spec.name, "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "input_dim": in_dim,
        "classes": n_cls, "epochs": epochs, "seed": seed, "variants": rows,
        "notes": [
            "All variants are evaluated on the same held-out split; preprocessing was fitted on training rows only.",
            "Unstructured pruning zeroes weights but storage stays dense, so its size does not shrink.",
            "Latency is single-sample CPU latency; INT8 dynamic quantisation can be slower than FP32 for tiny MLPs.",
            "No fine-tuning after pruning, so the pruned accuracies are a lower bound.",
        ],
    }
    result["saved_to"] = _save(result)
    progress(1.0, "Done")
    return result
