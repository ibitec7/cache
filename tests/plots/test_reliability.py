"""Reliability contract: equal-width probability bins, an exact partition, a PDF."""
from __future__ import annotations

import numpy as np
import pytest

from mrc.plots.reliability import bin_fractions, plot_reliability, reliability_table


def _sample(seed: int = 0, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pred = rng.random(n)
    y = (rng.random(n) < pred).astype(np.float32)
    return pred, y


def test_reliability_bins_and_writes_pdf(tmp_path):
    pred, y = _sample()
    frac = bin_fractions(pred, 12)
    assert len(frac) == 12
    assert abs(float(frac.sum()) - 1.0) < 1e-12
    path = plot_reliability(pred, y, tmp_path / "rel.pdf", n_bins=12)
    assert path.exists() and path.stat().st_size > 1000
    assert path.suffix == ".pdf"


def test_reliability_table_partitions_the_predictions():
    pred, y = _sample(seed=1)
    table = reliability_table(pred, y, n_bins=12)
    assert table["count"].shape == (12,)
    assert int(table["count"].sum()) == pred.size
    assert abs(float(table["fraction"].sum()) - 1.0) < 1e-12
    empty = table["count"] == 0
    assert np.all(np.isnan(table["pred_mean"][empty]))
    assert np.all(table["observed"][~empty] >= 0.0)
    assert np.all(table["observed"][~empty] <= 1.0)


def test_a_constant_forecast_lands_in_one_bin():
    pred = np.full(400, 0.5)
    y = np.zeros(400)
    y[::2] = 1.0
    table = reliability_table(pred, y, n_bins=12)
    occupied = table["count"] > 0
    assert int(occupied.sum()) == 1
    assert float(table["pred_mean"][occupied][0]) == pytest.approx(0.5)
    assert float(table["observed"][occupied][0]) == pytest.approx(0.5)


def test_probabilities_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bin_fractions(np.array([0.1, 1.5]), 4)
    with pytest.raises(ValueError, match="length"):
        reliability_table(np.array([0.1, 0.2]), np.array([1.0]), 4)
