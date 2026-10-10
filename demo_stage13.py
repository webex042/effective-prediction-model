"""
EPM Stage 13 — Research Experiment Demo
=======================================

Runs the primary research experiment (H1: reduced-step generative forecasting)
end-to-end with REAL trained models, persists registry records + a benchmark
summary, writes an automated Markdown report, and renders monochrome figures.

Run:
    python demo_stage13.py
    python demo_stage13.py --quick     # fast smoke configuration

Nothing here is fabricated: baselines are genuinely trained, and the diffusion
forecaster is trained once then sampled at several DDIM step counts (NFE).
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.logger import setup_logging
from src.research import ResearchConfig, run_experiment, generate_report
from src.research.registry import ExperimentRegistry

# Monochrome, retro-plot styling (black / grayscale only)
plt.rcParams.update({
    "font.family": "monospace",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#000000",
    "axes.linewidth": 1.2,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.color": "#cccccc",
    "grid.linestyle": ":",
})
GRAYS = ["#000000", "#303030", "#777777", "#aaaaaa", "#cccccc"]

PLOTS_DIR = Path("docs/plots/stage13")
WEB_PLOTS_DIR = Path("plots/stage13")
BENCH_DIR = Path("benchmarks")
DOCS_DIR = Path("docs")
for d in (PLOTS_DIR, WEB_PLOTS_DIR, BENCH_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _mean(block, metric):
    m = block.get(metric)
    return m["mean"] if m else float("nan")


def _std(block, metric):
    m = block.get(metric)
    return m["std"] if m else 0.0


def make_figures(summary: dict) -> list:
    diff = summary["diffusion"]
    steps = sorted(diff.keys(), key=int)
    rmse = [_mean(diff[s]["metrics"], "rmse") for s in steps]
    rmse_err = [_std(diff[s]["metrics"], "rmse") for s in steps]
    lat = [_mean(diff[s]["metrics"], "latency_ms") for s in steps]
    crps = [_mean(diff[s]["metrics"], "crps") for s in steps]
    saved = []

    # Figure 1: NFE sweep — RMSE and latency vs sampling steps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax1.errorbar(steps, rmse, yerr=rmse_err, marker="o", color="#000000",
                 linewidth=1.5, markersize=6, capsize=4, label="RMSE (±std)")
    ax1.set_title("Forecast RMSE vs Sampling Steps (NFE)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("DDIM sampling steps (fewer = cheaper)")
    ax1.set_ylabel("RMSE — physical units (lower better)")
    ax1.grid(True); ax1.legend(fontsize=8)
    ax2.plot(steps, lat, marker="s", color="#000000", linewidth=1.5, markersize=6)
    ax2.set_title("Inference Latency vs Sampling Steps", fontsize=10, fontweight="bold")
    ax2.set_xlabel("DDIM sampling steps")
    ax2.set_ylabel("Latency (ms / window, lower better)")
    ax2.grid(True)
    plt.tight_layout()
    p1 = PLOTS_DIR / "h1_nfe_sweep_rmse_latency.png"
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig); saved.append(p1)

    # Figure 2: Pareto — latency vs RMSE (diffusion steps + baselines)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.plot(lat, rmse, "--", color="#777777", linewidth=1.0, zorder=1)
    for x, y, s in zip(lat, rmse, steps):
        ax.scatter(x, y, color="#000000", s=70, zorder=3)
        ax.annotate(f"DDIM-{s}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    markers = ["^", "v", "D", "P"]
    for i, (name, b) in enumerate(summary["baselines"].items()):
        bx = _mean(b["metrics"], "latency_ms"); by = _mean(b["metrics"], "rmse")
        ax.scatter(bx, by, color=GRAYS[min(i + 1, len(GRAYS) - 1)],
                   marker=markers[i % len(markers)], s=90, edgecolor="#000000",
                   zorder=3, label=f"{name} (baseline)")
        ax.annotate(name, (bx, by), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_title("Accuracy–Latency Pareto Trade-off", fontsize=11, fontweight="bold")
    ax.set_xlabel("Latency (ms / window, log scale, lower better)")
    ax.set_ylabel("RMSE (lower better)")
    ax.set_xscale("log"); ax.grid(True); ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    p2 = PLOTS_DIR / "h1_accuracy_latency_pareto.png"
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig); saved.append(p2)

    # Figure 3: CRPS vs steps (probabilistic quality)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(steps, crps, marker="D", color="#000000", linewidth=1.5, markersize=6)
    ax.set_title("Probabilistic Quality (CRPS) vs Sampling Steps", fontsize=10, fontweight="bold")
    ax.set_xlabel("DDIM sampling steps"); ax.set_ylabel("CRPS (lower better)")
    ax.grid(True)
    plt.tight_layout()
    p3 = PLOTS_DIR / "h1_crps_vs_steps.png"
    fig.savefig(p3, bbox_inches="tight"); plt.close(fig); saved.append(p3)

    import shutil
    WEB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in saved:
        shutil.copy(p, WEB_PLOTS_DIR / p.name)
    return saved


def main():
    parser = argparse.ArgumentParser(description="EPM Stage 13 research experiment")
    parser.add_argument("--quick", action="store_true", help="fast smoke configuration")
    args = parser.parse_args()

    setup_logging(level="WARNING", file=False)
    print("=" * 78)
    print("  EPM — Stage 13: Research Experiment (H1 reduced-step generative forecasting)")
    print("=" * 78)

    if args.quick:
        cfg = ResearchConfig.quick()
    else:
        cfg = ResearchConfig.h1_reduced_step_forecasting(
            name="h1_reduced_step_forecasting",
            n_timesteps=1600, window_size=24, horizon=12,
            hidden_dim=96, context_dim=48, diffusion_timesteps=100,
            baselines=["linear", "lstm", "transformer"],
            epochs=18, seeds=[0, 1], sampling_steps=[100, 50, 20, 10, 5],
            n_mc_samples=16, latency_reps=30,
        )

    reg = ExperimentRegistry(root="experiments/registry")
    print(f"\n[Run] fingerprint={cfg.fingerprint()} seeds={cfg.seeds} "
          f"steps={cfg.sampling_steps}\n")
    summary = run_experiment(cfg, registry=reg, verbose=True)

    # Persist benchmark summary
    bench_path = BENCH_DIR / "stage13_research.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[Saved] benchmark summary -> {bench_path}")

    # Automated report
    report = generate_report(summary)
    report_path = DOCS_DIR / "stage13_research_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Saved] research report   -> {report_path}")

    figs = make_figures(summary)
    for p in figs:
        print(f"[Saved] figure            -> {p}")

    print("\n" + "=" * 78)
    print("  STAGE 13 COMPLETE — see docs/stage13_research_report.md")
    print("  H1 analysis:")
    for c in summary["h1_analysis"].get("comparisons", []):
        verdict = "within full-step CI" if c["rmse_within_full_ci"] else "accuracy drop"
        print(f"    DDIM-{c['sampling_steps']:>3}: {c['speedup_vs_full']}x faster, "
              f"RMSE {c['rmse_change_pct']:+.1f}% ({verdict})")
    print("=" * 78)


if __name__ == "__main__":
    main()
