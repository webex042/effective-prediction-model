"""
Chat responses and result formatting (shared by the web panel and the terminal).

``ChatResponse.text`` uses a tiny markdown subset (**bold**, `code`, line breaks,
"- " bullets) that the web client renders safely; ``tables`` carry structured
results that the web client draws as HTML tables and ``to_text()`` draws as
box tables for the terminal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.benchmarking.catalog import ALGORITHMS
from src.data.registry import DatasetSpec

STATES = ("idle", "slot_filling", "confirming", "executing", "showing_results", "error")


@dataclass
class ChatResponse:
    text: str
    state: str = "idle"
    intent: str = "unknown"
    kind: str = "message"            # message | question | results | error | progress
    job_id: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        """Plain-text rendering for the terminal chat."""
        out = [strip_markup(self.text)]
        for t in self.tables:
            if t.get("title"):
                out.append("\n" + t["title"])
            out.append(ascii_table(t["columns"], t["rows"]))
        if self.suggestions:
            out.append("\nTry: " + " | ".join(self.suggestions))
        return "\n".join(out)


def strip_markup(text: str) -> str:
    return text.replace("**", "").replace("`", "")


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—"
        a = abs(v)
        if a != 0 and (a < 1e-3 or a >= 1e6):
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


def mb(v: Optional[float]) -> str:
    if not v:
        return "<0.1 MB"
    return "<0.1 MB" if v < 0.1 else f"{v:.1f} MB" if v < 10 else f"{v:.0f} MB"


def ascii_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(c) for c in columns]] + [[fmt(v, 3) if not isinstance(v, str) else v for v in r] for r in rows]
    widths = [min(42, max(len(r[i]) for r in cells)) for i in range(len(columns))]
    clip = lambda s, w: s if len(s) <= w else s[: w - 1] + "…"  # noqa: E731
    line = lambda l, m, r: l + m.join("─" * (w + 2) for w in widths) + r  # noqa: E731
    out = [line("┌", "┬", "┐"), "│" + "│".join(f" {clip(c, w):<{w}} " for c, w in zip(cells[0], widths)) + "│",
           line("├", "┼", "┤")]
    for r in cells[1:]:
        out.append("│" + "│".join(f" {clip(c, w):<{w}} " for c, w in zip(r, widths)) + "│")
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def table(title: str, columns: List[str], rows: List[List[Any]], note: str = "") -> Dict[str, Any]:
    return {"title": title, "columns": columns, "rows": rows, "note": note}


def human_task(task: Optional[str]) -> str:
    return (task or "").replace("_", " ")


# =============================================================================
# Benchmark results
# =============================================================================

COLS = {
    "regression": [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")],
    "classification": [("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC")],
    "clustering": [("silhouette", "Silhouette"), ("ari_external", "ARI"), ("nmi_external", "NMI"),
                   ("davies_bouldin", "DB"), ("n_clusters", "k found")],
    "forecasting": [("rmse", "RMSE"), ("mase", "MASE"), ("smape", "sMAPE %"), ("skill_vs_naive", "Skill vs naive")],
    "anomaly_detection": [("pr_auc", "PR-AUC"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                          ("roc_auc", "ROC-AUC")],
}
CAL_COLS = [("ece", "ECE ↓"), ("brier", "Brier ↓"), ("mean_confidence", "Mean conf."), ("accuracy", "Accuracy")]


def benchmark_table(result: Dict[str, Any], calibration: bool = False,
                    only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    task = result["task"]
    cols = CAL_COLS if calibration else COLS.get(task, [])
    pm = result.get("primary_metric", {})
    models = [m for m in result.get("models", []) if m.get("status") == "ok" and (not only or m["algorithm"] in only)]
    models.sort(key=lambda m: m.get("rank", 99))
    has_cv = any(m.get("cv") for m in models)
    columns = ["#", "Algorithm"] + [c[1] for c in cols] + (["CV mean ± std"] if has_cv else []) + ["Latency µs", "Train s"]
    rows = []
    for m in models:
        r = [m.get("rank", "—"), m["name"] + (" ★" if m.get("pareto_optimal") else "")]
        r += [m["metrics"].get(k) for k, _ in cols]
        if has_cv:
            cv = m.get("cv") or {}
            r.append(f"{fmt(cv.get('mean'))} ± {fmt(cv.get('std'))}" if cv.get("mean") is not None else "—")
        r += [m.get("latency_us_per_sample"), m.get("train_time_s")]
        rows.append(r)
    goal = "↓" if pm.get("goal") == "minimise" else "↑"
    title = ("Calibration (lower ECE/Brier = better calibrated)" if calibration
             else f"Leaderboard — primary metric: {pm.get('name', '?').upper()} {goal}")
    return table(title, columns, rows, "★ = Pareto-optimal (no other model is both better and faster)")


def benchmark_summary(result: Dict[str, Any], calibration: bool = False) -> str:
    task = result["task"]
    s = result.get("summary") or {}
    name = lambda aid: ALGORITHMS[aid].name if aid in ALGORITHMS else (aid or "—")  # noqa: E731
    lines = [f"Done: **{result['dataset_name']}** · {human_task(task)} · {len(result.get('models', []))} model(s) "
             f"in {fmt(result.get('duration_s'), 1)} s (run `{result['run_id']}`)."]
    split = result.get("split") or {}
    if split.get("strategy"):
        bits = [f"{k.replace('n_', '')}={v}" for k, v in split.items() if k in ("n_train", "n_val", "n_test", "n_rows")]
        lines.append(f"Split: {split['strategy']}" + (f" ({', '.join(bits)})" if bits else "") + ", same split and seed for every model.")
    if s.get("best_by_metric") and not calibration:
        pm = result["primary_metric"]
        lines.append(f"- Best {pm['name'].upper()}: **{name(s['best_by_metric'])}** ({fmt(s.get('best_value'))})")
        if s.get("fastest_inference") and s["fastest_inference"] != s["best_by_metric"]:
            lines.append(f"- Fastest inference: {name(s['fastest_inference'])}")
        if s.get("baseline"):
            losers = [m["name"] for m in result["models"] if m.get("beats_baseline") is False and m["algorithm"] != s["baseline"]]
            if losers:
                lines.append(f"- Did not beat the {name(s['baseline'])}: {', '.join(losers)}")
    if calibration:
        ok = [m for m in result["models"] if m.get("status") == "ok" and m["metrics"].get("ece") is not None]
        if ok:
            best = min(ok, key=lambda m: m["metrics"]["ece"])
            lines.append(f"- Best calibrated: **{best['name']}** (ECE {fmt(best['metrics']['ece'])}). "
                         "ECE = average gap between confidence and accuracy over 10 bins.")
        else:
            lines.append("- No model exposed probabilities, so calibration could not be measured.")
    fs = result.get("forecast_setup")
    if fs:
        lines.append(f"- Window {split.get('window')} → horizon {split.get('horizon')}; naive baseline RMSE {fmt(fs.get('naive_test_rmse'))}.")
    an = result.get("anomaly_setup")
    if an:
        lines.append(f"- Labels were **not** used for training; threshold from contamination prior {fmt(an.get('contamination_prior'))}.")
    for w in result.get("warnings", [])[:3]:
        lines.append(f"- Note: {w}")
    failed = [m for m in result.get("models", []) if m.get("status") != "ok"]
    for m in failed[:3]:
        lines.append(f"- Failed: {m['name']} — {m.get('error')}")
    for aid, why in list((result.get("skipped_algorithms") or {}).items())[:3]:
        lines.append(f"- Skipped {name(aid)}: {'; '.join(why)}")
    return "\n".join(lines)


# =============================================================================
# Phase 3 workflow results
# =============================================================================

def generation_reply(r: Dict[str, Any]) -> (str, List[Dict[str, Any]]):
    vae = r["vae"]
    dist = r["distribution"]
    vae_sw = dist[0]["sliced_w"]
    gauss = next(x for x in dist if x["model"].startswith("Baseline"))
    floor = next(x for x in dist if x["model"].startswith("Reference"))
    verdict = ("beats" if vae_sw < gauss["sliced_w"] else "does **not** beat")
    lines = [
        f"Trained a VAE on **{r['dataset_name']}** ({r['input_dim']}-dim input → {r['latent_dim']}-dim latent, "
        f"β={fmt(r['beta'])}, {r['epochs']} epochs) and generated {r['num_samples']} synthetic rows.",
        f"- Held-out reconstruction MSE {fmt(vae['recon_mse_test'])} (standardised units), KL {fmt(vae['kl_test'])}, "
        f"{fmt(vae['params'])} parameters, trained in {fmt(vae['train_time_s'], 1)} s.",
        f"- Distribution match: sliced-Wasserstein {fmt(vae_sw)} vs real-data floor {fmt(floor['sliced_w'])}; "
        f"the VAE {verdict} an independent-Gaussian baseline ({fmt(gauss['sliced_w'])}).",
    ]
    if r.get("mean_abs_corr_error") is not None:
        lines.append(f"- Mean absolute correlation error vs held-out data: {fmt(r['mean_abs_corr_error'])}.")
    if r.get("ae"):
        lines.append(f"- Plain AE for comparison: reconstruction MSE {fmt(r['ae']['recon_mse'])} — {r['ae']['note']}")
    if r.get("dropped_columns"):
        lines.append(f"- Non-numeric columns left out: {', '.join(r['dropped_columns'][:8])}.")
    lines.append(f"- Samples saved to `{r['samples_csv']}`.")
    tables = [
        table("Distribution fidelity on held-out rows (lower = closer)", ["Model", "Sliced-W", "MMD", "Note"],
              [[x["model"], x["sliced_w"], x["mmd"], x["note"]] for x in dist]),
        table("Per-feature comparison (original units)", ["Feature", "Real μ", "Synth μ", "Real σ", "Synth σ", "KS p"],
              [[f["feature"], f["real_mean"], f["synth_mean"], f["real_std"], f["synth_std"], f["ks_p"]]
               for f in r["features"][:12]],
              "KS p < 0.05: that feature's marginal differs significantly from the real one."),
    ]
    return "\n".join(lines), tables


def gen_compare_reply(r: Dict[str, Any]) -> (str, List[Dict[str, Any]]):
    best = r["models"][0]
    floor = next(x for x in r["references"] if x["model"].startswith("Reference"))
    gauss = next(x for x in r["references"] if x["model"].startswith("Baseline"))
    lines = [f"Compared diffusion and flow matching on **{r['dataset_name']}** ({r['input_dim']} features, "
             f"{r['n_train']} training rows, {r['n_eval']} held-out rows, {r['epochs']} epochs each).",
             f"- Closest to the held-out data: **{best['model']}** (sliced-W {fmt(best['sliced_w'])}; "
             f"real-data floor {fmt(floor['sliced_w'])}, Gaussian baseline {fmt(gauss['sliced_w'])}).",
             f"- Training time: flow matching {fmt(r['train_time_s']['flow_matching'], 1)} s, "
             f"diffusion {fmt(r['train_time_s']['diffusion'], 1)} s."]
    lines += [f"- {n}" for n in r["notes"][-2:]]
    rows = [[m["model"], m["nfe"], m["sliced_w"], m["mmd"], m["straightness"], m["latency_ms_per_sample"]]
            for m in r["models"]]
    rows += [[x["model"], "—", x["sliced_w"], x["mmd"], "—", "—"] for x in r["references"]]
    return "\n".join(lines), [table("Generative comparison (held-out)", ["Model", "NFE", "Sliced-W ↓", "MMD ↓",
                                                                         "Straightness", "ms/sample"], rows)]


def compression_reply(r: Dict[str, Any]) -> (str, List[Dict[str, Any]]):
    v = r["variants"]
    teacher = v[0]
    smallest = min(v, key=lambda x: x["size_kb"])
    lines = [f"Compressed a classifier trained on **{r['dataset_name']}** ({r['n_train']} train / {r['n_test']} test rows, "
             f"{r['classes']} classes, {r['epochs']} epochs).",
             f"- Teacher: accuracy {fmt(teacher['accuracy'])}, {fmt(teacher['size_kb'], 1)} KB.",
             f"- Smallest: **{smallest['variant']}** at {fmt(smallest['size_kb'], 1)} KB "
             f"({fmt(smallest['size_reduction'], 1)}× smaller), accuracy {fmt(smallest['accuracy'])} "
             f"({fmt(100 * (smallest['accuracy_retention'] or 0), 1)}% of the teacher)."]
    lines += [f"- {n}" for n in r["notes"][1:3]]
    rows = [[x["variant"], x["accuracy"], x["f1"], x["size_kb"], x["size_reduction"], x["latency_us"], x["params"]] for x in v]
    return "\n".join(lines), [table("Compression variants (same held-out split)",
                                    ["Variant", "Accuracy", "F1", "Size KB", "× smaller", "Latency µs", "Params"], rows)]


# =============================================================================
# Catalogue views
# =============================================================================

def dataset_rows(specs: Sequence[DatasetSpec], status: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    rows = [[s.id, s.name, ", ".join(human_task(t) for t in s.tasks), s.n_samples, s.size_class,
             (status or {}).get(s.id, "remote" if s.requires_network else "built-in")] for s in specs]
    return table("Datasets", ["id", "Name", "Tasks", "Rows", "Size", "Status"], rows)


def describe_spec(spec: DatasetSpec, status: Dict[str, Any]) -> str:
    lines = [f"**{spec.name}** (`{spec.id}`)", spec.description,
             f"- Tasks: {', '.join(human_task(t) for t in spec.tasks)}",
             f"- Size: {fmt(spec.n_samples)} rows × {spec.n_features} features ({spec.size_class}, ~{fmt(spec.approx_memory_mb, 1)} MB in memory)",
             f"- Target: `{spec.target}` ({spec.target_type})" + (f", {human_task(spec.label_role)}" if spec.label_role != "target" else "") if spec.target else "- Target: none",
             f"- Source: {spec.source} — {spec.source_url}",
             f"- Licence: {spec.license}",
             f"- Missing values: {spec.missing_values}",
             f"- Split: {spec.split_strategy}; status: {status.get('state')}"]
    for c in spec.caveats[:2]:
        lines.append(f"- Caveat: {c}")
    return "\n".join(lines)


HELP_TEXT = """I'm the EPM experiment assistant. Tell me what you want in plain English and I'll pick the task, dataset and algorithms, run them through EPM's benchmark pipeline, and report measured results.
- **Run experiments:** "classify iris with random forest", "regression on diabetes with 5-fold cv", "cluster the wine dataset"
- **Time series & anomalies:** "forecast air quality 12 steps ahead with lstm", "detect anomalies in the NAB data"
- **Generative & efficiency:** "generate 200 samples from wine with a VAE", "compare diffusion vs flow matching on sensor data", "compress a model trained on breast cancer"
- **Explore:** "list regression datasets", "describe adult", "preview iris", "which algorithms work on seeds?", "recommend a model for diabetes"
- **Results & control:** "show results", "compare random forest and svm", "how calibrated is the classifier on wine?", "status", "cancel"
- **Learn:** "explain flow matching", "what is backpropagation?"
I only call EPM's own registry, models and pipelines; I never run arbitrary code."""
