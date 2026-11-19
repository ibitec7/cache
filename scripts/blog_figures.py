"""Draw the blog's three data figures from artifacts, as SVGs that sit on any background.

Input contract
--------------
`artifacts/tables/cluster*_metrics.json`, `artifacts/tables/cluster*_curves.npz`,
`artifacts/oot/<cluster>/result.json`, `artifacts/loco/<cluster>/result.json` and
`artifacts/retrain/<cluster>-<windows>/result.json`, all reached through `blog_build`, which is
the single reader of the artifact tree.

Output contract
---------------
`blog/diagrams/<name>.svg`, transparent background, no grid, no legend, series labelled at the
end of the line. Colours are mid-tone on purpose: the post is read on a white page and on a
near-black one, and a figure that assumes a light background is wrong half the time. Text is
emitted as paths (`svg.fonttype = path`) so the figure does not depend on a font being installed.

This module deliberately does not import `src/mrc/plots/style.py`: that module styles report
figures (serif, 3.35 in, print), and the post is a sans-serif web document at 7 in. Keeping one
place that sets rcParams per medium is the rule; sharing a print figure's rcParams across both is
not.
"""

from __future__ import annotations

import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")  # headless: no display in the build environment

import blog_build as bb
import matplotlib.pyplot as plt
import numpy as np

OUT = bb.ROOT / "blog" / "diagrams"

# Mid-tone palette: each colour clears 3:1 against both #ffffff and #0d1117.
INK = "#6f757e"
GRID = "#c9ced6"
SERIES = {"cluster10": "#c2562a", "cluster26": "#3d6fa8",
          "cluster45": "#2f7f6a", "cluster52": "#7a4f92"}
LABEL = {"cluster10": "cluster10", "cluster26": "cluster26",
         "cluster45": "cluster45", "cluster52": "cluster52"}


def setup() -> None:
    plt.rcParams.update({
        "figure.figsize": (8.8, 3.3),
        "figure.dpi": 200,
        "savefig.transparent": True,
        "svg.fonttype": "path",
        "svg.hashsalt": "mrc-blog",  # deterministic ids, so a rebuild does not churn the repo
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "text.color": INK,
        "lines.linewidth": 1.7,
        "legend.frameon": False,
    })


def save(fig, name: str) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.06, transparent=True,
                metadata={"Date": None})
    plt.close(fig)
    return path


def fig_curves() -> pathlib.Path:
    """Miss ratio against capacity on a real log axis, one line per trace."""
    fig, ax = plt.subplots()
    for cluster in bb.CLUSTERS:
        d = bb.curves(cluster)
        caps = np.asarray(d["capacities"], dtype=float)
        miss = np.asarray(d["truth_test"], dtype=float)
        ax.plot(caps, miss, color=SERIES[cluster], label=LABEL[cluster],
                marker="o", markersize=3.0, markerfacecolor="none", markeredgewidth=0.9)
        ax.annotate(LABEL[cluster], (caps[-1], miss[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=SERIES[cluster], annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 3.0e6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("capacity (objects, log scale)")
    ax.set_ylabel("miss ratio")
    ax.spines["left"].set_color(INK)
    fig.tight_layout()
    return save(fig, "curves")


def fig_ratio() -> pathlib.Path:
    """One horizontal bar per trace and protocol: model error over Kaplan-Meier error."""
    rows = []
    for cluster in bb.CLUSTERS:
        for name, family in (("out of time", "oot"), ("leave one out", "loco")):
            s = bb.load_json(bb.ART / family / cluster / "result.json")["summary"]["test"]
            rows.append((f"{LABEL[cluster]} {name}",
                         float(s["mae_mean"]) / float(s["km_mae_mean"]), cluster))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots()
    ys = np.arange(len(rows))
    ax.barh(ys, [r[1] for r in rows], height=0.62,
            color=[SERIES[r[2]] for r in rows], edgecolor="none")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for y, (_, v, _c) in zip(ys, rows):
        ax.text(v + 0.006, float(y), f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(1.0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("model error / Kaplan-Meier error (lower is better)")
    ax.set_ylim(-1.1, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return save(fig, "ratio")


def fig_retrain() -> pathlib.Path:
    """Test error against the number of pooled training windows, fit rows held fixed."""
    fig, ax = plt.subplots()
    notes: list[tuple[float, float, str, str]] = []
    for cluster in bb.CLUSTERS[:3]:
        pts = {w: mae for _r, mae, w in bb.retrain_points(cluster)}
        wins = sorted(pts)
        ax.plot(wins, [pts[w] for w in wins], color=SERIES[cluster], marker="o",
                markersize=3.6, label=LABEL[cluster])
        notes.append((wins[-1], pts[wins[-1]], LABEL[cluster], SERIES[cluster]))
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 12])
    ax.set_xticklabels(["1", "5", "12"])
    ax.set_xlim(0.85, 17)
    ax.set_ylim(0, max(0.006, ax.get_ylim()[1]))
    ax.set_xlabel("training windows pooled, log scale (fit rows held at ~3.6M)")
    ax.spines["left"].set_color(INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    notes.sort(key=lambda t: t[1])
    last = None
    for x, y, text, color in notes:
        ty = y
        if last is not None and ty - last < 0.10 * ax.get_ylim()[1]:
            ty = last + 0.10 * ax.get_ylim()[1]
        last = ty
        ax.annotate(text, (x, y), xytext=(6, (ty - y) / (ax.get_ylim()[1] - ax.get_ylim()[0]) * 90),
                    textcoords="offset points", va="center", fontsize=8.5, color=color,
                    annotation_clip=False)
    ax.set_ylabel("mean test MAE")
    ax.spines["left"].set_color(INK)
    fig.tight_layout()
    return save(fig, "retrain")


def main() -> int:
    setup()
    for fn in (fig_curves, fig_ratio, fig_retrain):
        path = fn()
        print(f"  {path.name:<12} {path.stat().st_size:>7} bytes")
    print("wrote 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
