"""The MRC comparison figure.

One panel, one line per curve, capacities on a log axis. The truth curve comes from
`mrc.gt.stackdist`, the estimates from `mrc.models.hazard`; this module only draws
them, so it owns no number of its own.

The style contract is not negotiable here: `apply_style()` and `fig_double()` come
from `mrc.plots.style`, no rcParam is set in this file, and the legend is capped at
four entries because the caption carries the claim for the fifth entry onwards.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from mrc.plots.style import PALETTE, apply_style, fig_double, label_mrc

MAX_LEGEND_ENTRIES = 4
"""Contract cap on series per panel; the report shows the truth plus the best three."""


def plot_mrc(
    capacities: Sequence[int] | np.ndarray,
    curves: Mapping[str, np.ndarray],
    path: str | Path,
    ylim: tuple[float, float] | None = None,
) -> Path:
    """Draw one line per entry of `curves` against `capacities`; write a PDF.

    capacities  strictly positive object counts, shared by every curve
    curves      {legend label: miss ratio per capacity}; at most four entries,
                drawn in the fixed palette order so a series keeps its colour
    path        output PDF; parent directories are created
    ylim        (low, high) miss-ratio limits; None keeps matplotlib's autoscale
    """
    import matplotlib.pyplot as plt

    caps = np.asarray(capacities, dtype=np.int64)
    if caps.size == 0:
        raise ValueError("capacities is empty")
    if np.any(caps <= 0):
        raise ValueError("capacities must be strictly positive for a log axis")
    if not curves:
        raise ValueError("curves is empty")
    if len(curves) > MAX_LEGEND_ENTRIES:
        raise ValueError(
            f"legend holds at most {MAX_LEGEND_ENTRIES} entries, got {len(curves)}"
        )

    apply_style()
    fig, ax = fig_double(2.4)
    for i, (label, values) in enumerate(curves.items()):
        y = np.asarray(values, dtype=float)
        if y.shape != caps.shape:
            raise ValueError(f"curve {label!r} has length {y.size}, capacities {caps.size}")
        ax.plot(caps, y, color=PALETTE[i], marker="o", markersize=2.5, label=label)

    label_mrc(ax)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc="upper right")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out
