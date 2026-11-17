"""Report figures from the cached metrics artefacts.

Every figure imports `apply_style()`; no rcParams are set anywhere else. The
figures answer one question each: how close the curve is, where it is wrong, and
what the error costs in capacity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mrc.analysis import load_metrics
from mrc.config import ARTIFACTS, REPO
from mrc.plots.style import DOUBLE_COL, PALETTE, SINGLE_COL, apply_style, label_mrc

# One home for the figures. `report/main.tex` sets \graphicspath{{figures/}} and
# `report/sections/05-experiments.tex` includes `figures/<name>.pdf`, so the
# producer writes straight into the directory the report reads. Do not add a
# second output directory: a copy would drift from the LaTeX.
REPORT_FIG_DIR = REPO / "report" / "figures"


def _plt():
    import matplotlib.pyplot as plt

    apply_style()
    return plt


def _panels(n: int, sharey: bool = False, height: float = 2.0):
    """Axes for `n` traces: one row up to three, a 2x2 block beyond that.

    Four panels abreast at 7.0 in leave 1.7 in each, where the tick labels and the
    legend collide. The grid only changes the layout, never the data or the panel set.
    """
    plt = _plt()
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, height), sharey=sharey)
    else:
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(DOUBLE_COL, 1.55 * rows),
                                 sharey=sharey, squeeze=False)
        # The top row's tick labels sit directly above the bottom row's titles at the
        # default spacing, so the two collide in the PDF text layer.
        fig.subplots_adjust(hspace=0.55, wspace=0.22)
    return fig, np.atleast_1d(axes).ravel()


def plot_curves(clusters: list[str], path: Path | None = None) -> Path:
    """Ground truth, hazard model and Kaplan-Meier on the test block."""
    fig, axes = _panels(len(clusters), sharey=True)
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.06))
    path = Path(path or REPORT_FIG_DIR / "mrc-curves.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    _plt().close(fig)
    return path


FLOOR = 3e-5


def plot_error(clusters: list[str], path: Path | None = None) -> Path:
    """Absolute curve error against capacity on a log scale.

    A log axis is the only way to see both estimators in one panel: the hazard
    error sits one to two decades below the marginal curve. Exact zeros are
    floored at `FLOOR`, which is below every measured hazard error.
    """
    fig, axes = _panels(len(clusters), sharey=True)
    styles = (("gbdt_test", PALETTE[1], "Hazard GBDT", "-"),
              ("km_test", PALETTE[2], "Kaplan-Meier", "--"),
              ("hitcount_test", PALETTE[3], "Hit-count", ":"))
    for ax, cluster in zip(axes, clusters):
        payload = load_metrics(cluster)
        caps = np.asarray(payload["curves"]["capacities"])
        truth = np.asarray(payload["curves"]["truth_test"])
        for key, colour, label, style in styles:
            if key not in payload["curves"]:
                continue
            err = np.maximum(np.abs(np.asarray(payload["curves"][key]) - truth), FLOOR)
            ax.plot(caps, err, color=colour, lw=1.2, ls=style, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cache capacity (objects)")
        ax.set_title(cluster.replace("cluster", "C"), fontsize=8)
        ax.set_ylim(FLOOR, 0.5)
    axes[0].set_ylabel("Absolute error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.06))
    path = Path(path or REPORT_FIG_DIR / "cluster-error.pdf")
    fig.savefig(path)
    _plt().close(fig)
    return path


def plot_provisioning(clusters: list[str], path: Path | None = None) -> Path:
    """Capacity ratio (model / truth) at 60, 80 and 95 % of the hit ceiling."""
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
    ax.set_xticklabels([f"{int(t * 100)}%" for t in fractions], fontsize=7)
    ax.set_xlabel("Target hit ratio (% of ceiling)")
    ax.set_ylabel("Capacity ratio (model / truth)")
    # Below the axes, not inside: the bars reach 1.0, so an in-axes legend would
    # sit on a bar and on the 1.0 reference line.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
    path = Path(path or REPORT_FIG_DIR / "capacity-ratio.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_importance(clusters: list[str], path: Path | None = None) -> Path:
    """Mean absolute SHAP contribution, top six features per cluster."""
    import json

    plt = _plt()
    # Only clusters with a SHAP artefact get a panel. A cluster that has been trained
    # but not interpreted must not appear as a blank panel: an empty column reads as
    # "nothing ranks here" instead of "this trace was not interpreted".
    have = [c for c in clusters if (ARTIFACTS / "tables" / f"{c}_shap.json").exists()]
    if not have:
        raise FileNotFoundError("no <cluster>_shap.json artefacts; run `mrc interpret <cluster>`")
    fig, axes = _panels(len(have))
    for ax, cluster in zip(axes, have):
        shap_path = ARTIFACTS / "tables" / f"{cluster}_shap.json"
        top = json.loads(shap_path.read_text())["top"][:6][::-1]
        names = [row["feature"].replace("_", " ") for row in top]
        vals = [row.get("mean_abs_contribution", 0.0) for row in top]
        ax.barh(names, vals, color=PALETTE[0], height=0.6)
        ax.set_title(cluster.replace("cluster", "C"), fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("mean |SHAP|")
    path = Path(path or REPORT_FIG_DIR / "importance.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(clusters: list[str]) -> list[Path]:
    """Every figure the cached metrics support; clusters without a run are skipped."""
    present = [c for c in clusters if (ARTIFACTS / "tables" / f"{c}_metrics.json").exists()]
    if not present:
        raise FileNotFoundError("no metrics artefacts; run `mrc train <cluster>` first")
    return [plot_curves(present, REPORT_FIG_DIR / "mrc-curves.pdf"),
            plot_error(present, REPORT_FIG_DIR / "cluster-error.pdf"),
            plot_provisioning(present, REPORT_FIG_DIR / "capacity-ratio.pdf"),
            plot_importance(present, REPORT_FIG_DIR / "importance.pdf")]
