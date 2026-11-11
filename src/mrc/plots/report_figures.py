"""Report figures from the cached metrics artefacts.

Every figure imports `apply_style()`; no rcParams are set anywhere else. The
figures answer one question each: how close the curve is, where it is wrong, and
what the error costs in capacity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mrc.analysis import load_metrics
from mrc.config import ARTIFACTS
from mrc.plots.style import DOUBLE_COL, PALETTE, SINGLE_COL, apply_style, label_mrc

FIG_DIR = ARTIFACTS / "figures"


def _plt():
    import matplotlib.pyplot as plt

    apply_style()
    return plt


def plot_curves(clusters: list[str], path: Path | None = None) -> Path:
    """Ground truth, hazard model and Kaplan-Meier on the test block."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(clusters), figsize=(DOUBLE_COL, 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cluster in zip(axes, clusters):
        payload = load_metrics(cluster)
        caps = np.asarray(payload["curves"]["capacities"])
        ax.plot(caps, payload["curves"]["truth_test"], color="black", lw=1.3,
                label="Mattson (truth)")
        ax.plot(caps, payload["curves"]["gbdt_test"], color=PALETTE[1], lw=1.2,
                label="Hazard GBDT")
        ax.plot(caps, payload["curves"]["km_test"], color=PALETTE[2], lw=1.2, ls="--",
                label="Kaplan-Meier")
        label_mrc(ax)
        ax.set_title(cluster.replace("cluster", "C"), fontsize=8)
        ax.set_ylim(0.0, 1.02)
    axes[0].legend(loc="lower left", fontsize=7)
    path = Path(path or FIG_DIR / "fig_mrc.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error(clusters: list[str], path: Path | None = None) -> Path:
    """Absolute error of the two estimators against capacity."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(clusters), figsize=(DOUBLE_COL, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cluster in zip(axes, clusters):
        payload = load_metrics(cluster)
        caps = np.asarray(payload["curves"]["capacities"])
        truth = np.asarray(payload["curves"]["truth_test"])
        for key, colour, label in (("gbdt_test", PALETTE[1], "Hazard GBDT"),
                                   ("km_test", PALETTE[2], "Kaplan-Meier")):
            err = np.abs(np.asarray(payload["curves"][key]) - truth)
            ax.plot(caps, err, color=colour, lw=1.2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Cache capacity (objects)")
        ax.set_title(cluster.replace("cluster", "C"), fontsize=8)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel("Absolute error")
    axes[0].legend(loc="upper left", fontsize=7)
    path = Path(path or FIG_DIR / "fig_error.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_provisioning(clusters: list[str], path: Path | None = None) -> Path:
    """Capacity ratio (model / truth) for 80, 90 and 95 % hit targets."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.1))
    fractions = (0.60, 0.80, 0.95)
    width = 0.36
    x = np.arange(len(fractions))
    for offset, (model, colour, label) in enumerate(
            (("gbdt", PALETTE[1], "Hazard GBDT"), ("km", PALETTE[2], "Kaplan-Meier"))):
        ratios = []
        for target in fractions:
            vals = []
            for cluster in clusters:
                payload = load_metrics(cluster)
                entry = next((e for e in payload["metrics"]["provisioning"][model]
                              if abs(e["fraction"] - target) < 1e-9), None)
                vals.append(entry["ratio"] if entry and np.isfinite(entry["ratio"]) else np.nan)
            ratios.append(np.nanmedian(vals))
        ax.bar(x + (offset - 0.5) * width, ratios, width=width, color=colour, label=label)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t * 100)}% of ceiling" for t in fractions])
    ax.set_xlabel("Target as a fraction of the achievable hit ratio")
    ax.set_ylabel("Capacity ratio (model / truth)")
    ax.legend(loc="upper left", fontsize=7)
    path = Path(path or FIG_DIR / "fig_capacity.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_importance(clusters: list[str], path: Path | None = None) -> Path:
    """Mean absolute SHAP contribution, top six features per cluster."""
    import json

    plt = _plt()
    fig, axes = plt.subplots(1, len(clusters), figsize=(DOUBLE_COL, 2.0))
    axes = np.atleast_1d(axes)
    for ax, cluster in zip(axes, clusters):
        shap_path = ARTIFACTS / "tables" / f"{cluster}_shap.json"
        if not shap_path.exists():
            continue
        top = json.loads(shap_path.read_text())["top"][:6][::-1]
        names = [row["feature"].replace("_", " ") for row in top]
        vals = [row.get("mean_abs_contribution", 0.0) for row in top]
        ax.barh(names, vals, color=PALETTE[0], height=0.6)
        ax.set_title(cluster.replace("cluster", "C"), fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("mean |SHAP|")
    path = Path(path or FIG_DIR / "fig_importance.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(clusters: list[str]) -> list[Path]:
    """Every figure the cached metrics support; clusters without a run are skipped."""
    present = [c for c in clusters if (ARTIFACTS / "tables" / f"{c}_metrics.json").exists()]
    if not present:
        raise FileNotFoundError("no metrics artefacts; run `mrc train <cluster>` first")
    return [plot_curves(present), plot_error(present), plot_provisioning(present),
            plot_importance(present)]
