from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .io import complete_output, prepare_output, write_json

def ridge_errors(rho: float, penalty: float, a: float = 1., b: float = 1.) -> dict:

    if not 0 <= rho <= 1 or penalty <= 0:
        raise ValueError("require rho in [0,1] and penalty > 0")
    covariance = torch.diag(torch.tensor([2 - rho, rho], dtype=torch.float64))
    true_w = torch.tensor([a, b], dtype=torch.float64)
    w = torch.linalg.solve(covariance + penalty * torch.eye(2, dtype=torch.float64), covariance @ true_w)
    delta = w - true_w
    exact = (a * penalty / (2 - rho + penalty)) ** 2 + (b * penalty / (rho + penalty)) ** 2
    return {"rho": rho, "lambda": penalty, "train_distribution_mse": float(delta @ covariance @ delta),
            "independent_test_mse": float(delta @ delta), "analytic_test_mse": exact,
            "coefficient_common": float(w[0]), "coefficient_contrast": float(w[1])}

def analytic(output: str):
    out = prepare_output(output, "analytic", {"rho": 1e-4, "a": 1., "b": 1.})
    results = [ridge_errors(1e-4, float(l)) for l in np.logspace(-7, 0, 121)]
    if max(abs(r["independent_test_mse"] - r["analytic_test_mse"]) for r in results) > 1e-12:
        raise AssertionError("PyTorch solve does not match the analytical expression")
    write_json(out / "results.json", results)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key, label in [("train_distribution_mse", "Same-distribution error"),
                       ("independent_test_mse", "Independent-factor test error")]:
        ax.loglog([r["lambda"] for r in results], [r[key] for r in results], label=label)
    ax.axvline(1e-4, color="grey", ls="--", alpha=.5)
    ax.set(xlabel="Ridge penalty", ylabel="Population MSE", title="Calculated illustration, not a nonlinear experiment")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / "ridge.png", dpi=180); plt.close(fig)
    complete_output(out)
    return out

def clustered_values(rows: list[dict], metric) -> np.ndarray:

    clusters = defaultdict(list)
    for r in rows:
        clusters[r["teacher_seed"]].append(metric(r))
    return np.asarray([np.mean(clusters[k]) for k in sorted(clusters)], dtype=float)

def estimate(values: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(2000, len(values)), replace=True).mean(1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return mean, float(lo), float(hi)

def report(run: str) -> Path:
    out = Path(run)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("cannot summarize an incomplete run")
    results = json.loads((out / "results.json").read_text())
    if manifest["command"] == "sweep":
        return _sweep_report(out, results)
    if manifest["command"] != "active":
        raise ValueError("report supports sweep and active outputs")
    methods = list(dict.fromkeys(r["method"] for r in results))
    final_round = max(r["round"] for r in results)
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in methods:
        curve = []
        for round_id in sorted(set(r["round"] for r in results)):
            rows = [r for r in results if r["method"] == method and r["round"] == round_id]
            mean, lo, hi = estimate(clustered_values(rows, lambda r: r["ood"]["nmse"]))
            curve.append((rows[0]["acquired_labels"], mean, lo, hi))
        curve = np.asarray(curve)
        line, = axes[0].plot(curve[:, 0], curve[:, 1], marker="o", label=method)
        axes[0].fill_between(curve[:, 0], curve[:, 2], curve[:, 3], alpha=.12, color=line.get_color())
        rows = [r for r in results if r["method"] == method and r["round"] == final_round]
        stats = estimate(clustered_values(rows, lambda r: r["ood"]["nmse"]))
        summary[method] = {"heldout_nmse_mean": stats[0], "teacher_bootstrap_95_percent": list(stats[1:]),
                           "teachers": len(set(r["teacher_seed"] for r in rows)),
                           "acquired_labels": rows[0]["acquired_labels"],
                           "response_mse_mean": float(clustered_values(rows, lambda r: r["response_mse"]).mean()),
                           "predictor_and_committee_training_seconds": sum(r["compute"]["seconds"] for r in rows),
                           "training_runs_per_trial": rows[0]["compute"]["training_runs"],
                           "oracle_scoring_labels_per_trial": rows[0]["oracle_scoring_labels"]}

    initial = [r for r in results if r["method"] == methods[0] and r["round"] == 0]
    signal_names = ["path_pairs", "seed_pairs", "path_points"]
    signal_stats = []
    for signal in signal_names:
        vals = clustered_values(initial, lambda r: r["score_quality"][signal]["auroc_top_quartile_error"])
        signal_stats.append(estimate(vals))
    axes[1].bar(signal_names, [s[0] for s in signal_stats], color=["#168b89", "#526c99", "#d89c49"])
    axes[1].axhline(.5, color="grey", ls="--"); axes[1].set_ylim(0, 1)
    axes[1].set(title="Initial failure detection (held-out pairs)", ylabel="AUROC: highest-error quartile")
    axes[0].set(xlabel="New oracle endpoint labels", ylabel="Held-out normalized MSE", title="Measured acquisition pilot")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    fig.savefig(out / "results.png", dpi=180); plt.close(fig)
    write_json(out / "summary.json", summary)
    lines = ["# Measured acquisition results", "", "These are preliminary controlled-task measurements, not evidence of novelty or a general advantage.", "",
             "| Method | Final held-out NMSE | Teacher bootstrap 95% interval | New labels |", "|---|---:|---:|---:|"]
    for method, stats in summary.items():
        lo, hi = stats["teacher_bootstrap_95_percent"]
        lines.append(f"| {method} | {stats['heldout_nmse_mean']:.6f} | [{lo:.6f}, {hi:.6f}] | {stats['acquired_labels']} |")
    lines += ["", "Intervals resample independent teacher instances after averaging nested training seeds. With few teachers, intervals are descriptive and unreliable for broad inference.",
              "All arms train both committees; ID validation selects the initial predictor decay, then freezes it across rounds. The oracle arm, if present, uses extra privileged labels and is not a fair budget-matched baseline.",
              "", "## Direct paired comparison", ""]
    if "path_pairs" in methods and "random_pairs" in methods:
        a = {(r["teacher_seed"], r["training_seed"]): r["ood"]["nmse"] for r in results if r["method"] == "path_pairs" and r["round"] == final_round}
        b = {(r["teacher_seed"], r["training_seed"]): r["ood"]["nmse"] for r in results if r["method"] == "random_pairs" and r["round"] == final_round}
        diff = defaultdict(list)
        for key in a:
            diff[key[0]].append(a[key] - b[key])
        stats = estimate(np.asarray([np.mean(v) for v in diff.values()]))
        lines.append(f"Path-pair minus random-pair NMSE: {stats[0]:.6f}, teacher-bootstrap interval [{stats[1]:.6f}, {stats[2]:.6f}]. Negative favors path-pair acquisition.")
    lines += ["", "## Initial score quality", "", "| Signal | Mean AUROC |", "|---|---:|"]
    for signal, stats in zip(signal_names, signal_stats):
        lines.append(f"| {signal} | {stats[0]:.6f} |")
    lines += ["", "See `results.json` for every run, per-cell errors, selected checkpoints, rank diagnostics, score metrics, and compute accounting.",
              "The nonlinear rank measurements are descriptive; no causal conclusion about low rank is implied."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return out / "RESULTS.md"

def _sweep_report(out: Path, results: list[dict]) -> Path:
    coverages = sorted(set(r["coverage"] for r in results))
    decays = sorted(set(r["weight_decay"] for r in results))
    arrays = [np.zeros((len(coverages), len(decays))) for _ in range(2)]
    summary = []
    for i, p in enumerate(coverages):
        for j, d in enumerate(decays):
            rows = [r for r in results if r["coverage"] == p and r["weight_decay"] == d]
            for k, key in enumerate(["id", "ood"]):
                arrays[k][i, j] = clustered_values(rows, lambda r: r[key]["mse"]).mean()
            summary.append({"coverage": p, "weight_decay": d, "id_mse": arrays[0][i, j], "ood_mse": arrays[1][i, j]})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, values, title in zip(axes, arrays, ["Same-distribution MSE", "Independent-factor test MSE"]):
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(decays)), [f"{d:g}" for d in decays])
        ax.set_yticks(range(len(coverages)), [f"{p:g}" for p in coverages])
        ax.set(xlabel="Weight decay", ylabel="Independent-component probability", title=title)
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.savefig(out / "results.png", dpi=180); plt.close(fig)
    write_json(out / "summary.json", summary)
    lines = ["# Measured phase-map pilot", "", "Independent-factor test examples can be rare rather than strictly unseen in this unmasked experiment.", "",
             "| Coverage | Weight decay | Mean ID MSE | Mean OOD MSE |", "|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['coverage']:g} | {r['weight_decay']:g} | {r['id_mse']:.6f} | {r['ood_mse']:.6f} |")
    lines += ["", "Means first average training seeds within teachers. This pilot does not establish a phase transition or prove a causal role for low rank."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return out / "RESULTS.md"
