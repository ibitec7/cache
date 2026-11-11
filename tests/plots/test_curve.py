"""Figure contract: a PDF over 1 KB, at most four legend entries, no title."""
from __future__ import annotations

import numpy as np
import pytest

from mrc.plots.curve import plot_mrc

CAPS = np.array([1, 2, 10, 100, 1000], dtype=np.int64)
TRUTH = np.array([1.0, 0.9, 0.8, 0.4, 0.2])


def test_plot_mrc_writes_pdf_and_respects_the_legend_cap(tmp_path):
    curves = {"Mattson (truth)": TRUTH, "Hazard GBDT": TRUTH + 0.02}
    path = plot_mrc(CAPS, curves, tmp_path / "mrc.pdf", ylim=(0.0, 1.05))
    assert path.exists() and path.stat().st_size > 1000
    assert path.suffix == ".pdf"


def test_plot_mrc_rejects_more_than_four_series(tmp_path):
    with pytest.raises(ValueError, match="legend"):
        plot_mrc(CAPS, {f"c{i}": TRUTH for i in range(5)}, tmp_path / "bad.pdf")


def test_plot_mrc_rejects_a_curve_of_the_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="length"):
        plot_mrc(CAPS, {"short": np.zeros(3)}, tmp_path / "bad.pdf")
