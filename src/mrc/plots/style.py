"""The single source of truth for figure style.

Every script that draws a figure imports `apply_style()` first and never sets
rcParams anywhere else. Rules (see AGENTS.md):
  * Times-compatible serif everywhere, STIX maths so equations match the text
  * 3.35 in single column, 7.0 in double, 300 dpi, PDF output
  * no grid, no top/right spines, at most one annotation per panel
  * no figure titles: the caption carries the claim
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

SERIF_CANDIDATES = ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"]

SINGLE_COL = 3.35
DOUBLE_COL = 7.0

PALETTE = ["#1f3b57", "#c1553b", "#4c7f6b", "#8a6d3b", "#5b5b8a"]


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_CANDIDATES,
        "font.size": 9,
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=PALETTE),
    })


def fig_single(height: float = 2.1):
    return plt.subplots(figsize=(SINGLE_COL, height))


def fig_double(height: float = 2.4):
    return plt.subplots(figsize=(DOUBLE_COL, height))


def label_mrc(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("Cache capacity (objects)")
    ax.set_ylabel("Miss ratio")
