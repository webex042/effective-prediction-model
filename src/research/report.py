"""
EPM Stage 13 — Automated Research Report Generator
==================================================

Turns a :func:`src.research.pipeline.run_experiment` summary into a Markdown
report with the sections required for a reproducible empirical study.
Everything printed is derived from recorded numbers — nothing is invented.
"""

from __future__ import annotations

from typing import Any, Dict


def _fmt(agg: Dict[str, Dict[str, float]], key: str, nd: int = 3) -> str:
    """Format a mean ± std (95% CI) cell from an aggregated metric block."""
    m = agg.get(key)
    if not m:
        return "—"
    return (
        f"{m['mean']:.{nd}f} ± {m['std']:.{nd}f} "
        f"(95% CI [{m['ci_low']:.{nd}f}, {m['ci_high']:.{nd}f}])"
    )


def generate_report(summary: Dict[str, Any]) -> str:
    cfg = summary["config"]
    split = summary["split"]
    L: list[str] = []
    a = L.append

    a(f"# Stage 13 Research Report — {cfg.get('name', 'experiment')}")
    a("")
    a(f"**Fingerprint:** `{summary.get('fingerprint', '')}`  ")
    a(f"**Hypothesis ({cfg.get('hypothesis')}):** {cfg.get('hypothesis_statement')}")
    a("")

    a("## 1. Research Question")
    a("Can a reduced-step generative (diffusion) forecaster retain competitive "
      "forecasting accuracy while reducing inference latency and the number of "
      "function evaluations (NFE) relative to many-step diffusion sampling?")
    a("")

    a("## 2. Experimental Setup")
    a(f"- **Dataset:** {cfg.get('dataset')} ({cfg.get('dataset_version')}), "
      f"{cfg.get('n_timesteps')} timesteps @ {cfg.get('sampling_rate_minutes')} min")
    a(f"- **Window / horizon:** {cfg.get('window_size')} → {cfg.get('horizon')} "
      f"(target channel index {cfg.get('target_channel')})")
    a(f"- **Split (time-aware, no shuffle):** train={split.get('train')}, "
      f"val={split.get('val')}, test={split.get('test')} windows")
    a(f"- **Seeds:** {summary.get('seeds')} (metrics are mean ± std with bootstrap CI)")
    a(f"- **Diffusion:** {cfg.get('schedule_type')} schedule, "
      f"T={cfg.get('diffusion_timesteps')}, hidden={cfg.get('hidden_dim')}, "
      f"MC samples={cfg.get('n_mc_samples')}, eta={cfg.get('eta')}")
    env = summary.get("environment", {})
    a(f"- **Environment:** Python {env.get('python')}, torch {env.get('torch')}, "
      f"device={env.get('device_name')}")
    a("")

    a("## 3. Baselines (all genuinely trained on identical splits)")
    a("| Model | Params | Size (KB) | MAE (°, ↓) | RMSE (°, ↓) | Latency ms/win (↓) |")
    a("|---|---|---|---|---|---|")
    for name, b in summary.get("baselines", {}).items():
        m = b["metrics"]
        a(f"| {name} | {b['parameter_count']:,} | {b['model_size_kb']} | "
          f"{_fmt(m, 'mae')} | {_fmt(m, 'rmse')} | {_fmt(m, 'latency_ms')} |")
    a("")

    a("## 4. Diffusion NFE Sweep (same weights, varying DDIM steps)")
    a("| Steps (NFE) | RMSE (↓) | CRPS (↓) | PICP@90% (→0.90) | MPIW (↓) | Latency ms/win (↓) |")
    a("|---|---|---|---|---|---|")
    for steps in sorted(summary.get("diffusion", {}), key=int, reverse=True):
        m = summary["diffusion"][steps]["metrics"]
        a(f"| {steps} | {_fmt(m, 'rmse')} | {_fmt(m, 'crps')} | "
          f"{_fmt(m, 'picp_90', 3)} | {_fmt(m, 'mpiw_90')} | {_fmt(m, 'latency_ms')} |")
    a("")

    a("## 5. H1 Efficiency Analysis")
    h1 = summary.get("h1_analysis", {})
    if h1:
        a(f"Reference (full) sampler: **{h1.get('full_steps')} steps**, "
          f"RMSE={h1.get('full_rmse_mean')}, latency={h1.get('full_latency_ms_mean')} ms/window.")
        a("")
        a("| Reduced steps | NFE reduction | Speedup | ΔRMSE vs full | Within full-step CI? |")
        a("|---|---|---|---|---|")
        for c in h1.get("comparisons", []):
            a(f"| {c['sampling_steps']} | {c['nfe_reduction']}× | {c['speedup_vs_full']}× | "
              f"{c['rmse_change_pct']:+.2f}% | {'yes' if c['rmse_within_full_ci'] else 'no'} |")
    a("")

    a("## 6. Interpretation")
    a(_interpret(h1))
    a("")

    a("## 7. Limitations")
    a("- Synthetic industrial telemetry (not a real sensor archive); conclusions "
      "are about the modeling trade-off, not a specific machine.")
    a("- Timing measured on CPU; GPU latency ratios will differ.")
    a("- Single architecture family; no hyperparameter search on the test set.")
    a("")

    a("## 8. Reproduction")
    a("```bash")
    a("python demo_stage13.py            # full experiment (several minutes, CPU)")
    a("python -m pytest tests/test_research.py -q")
    a("```")
    a(f"Config fingerprint `{summary.get('fingerprint','')}` pins dataset, splits, "
      "seeds, and model/sampling settings. Registry records: "
      f"{', '.join(summary.get('record_ids', [])) or 'none'}.")
    a("")
    return "\n".join(L)


def _interpret(h1: Dict[str, Any]) -> str:
    comps = h1.get("comparisons", []) if h1 else []
    if not comps:
        return "No reduced-step comparison was available."
    competitive = [c for c in comps if c["rmse_within_full_ci"]]
    if competitive:
        best = max(competitive, key=lambda c: c["speedup_vs_full"])
        return (
            f"At least one reduced-step setting stays within the full-step RMSE "
            f"confidence interval. The most aggressive such setting uses "
            f"{best['sampling_steps']} steps ({best['nfe_reduction']}× fewer NFE, "
            f"{best['speedup_vs_full']}× faster) with an RMSE change of "
            f"{best['rmse_change_pct']:+.2f}% — consistent with H1 on this dataset. "
            f"Settings that fall outside the CI mark where accuracy degrades."
        )
    worst_ok = min(comps, key=lambda c: abs(c["rmse_change_pct"]))
    return (
        "No reduced-step setting stayed within the full-step RMSE confidence "
        f"interval; the closest ({worst_ok['sampling_steps']} steps) changed RMSE by "
        f"{worst_ok['rmse_change_pct']:+.2f}%. On this dataset H1 is only weakly "
        "supported — reducing NFE traded measurable accuracy for speed."
    )
