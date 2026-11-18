"""Gates for the OOT sampling knobs: window stride and request subsampling.

Three claims are load-bearing and each gets an executable check:

1. `DistanceHistogram.mrc` is the same curve as `mrc_curve` on the raw distances, so
   a kept window's truth is exact and not a bin approximation.
2. Scoring a *sample* of a window's requests gives an unbiased estimate of the
   full-window MAE, and the published CI covers the full-window number. This is the
   only reason `--score-sample` is allowed to exist.
3. `thin_windows` never mixes blocks: every kept window keeps its own block label and
   its time order, so the train / valid / test split cannot move.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mrc.data.hist import DistanceHistogram
from mrc.eval.metrics import curve_metrics, curve_metrics_ci
from mrc.eval.mrc import mrc_curve

ROOT = Path(__file__).resolve().parents[1]


def _load_oot():
    spec = importlib.util.spec_from_file_location("oot_script", ROOT / "scripts" / "oot.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_histogram_truth_is_exact() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(50, 4000))
        scale = float(rng.choice([4.0, 40.0, 4000.0]))
        r = rng.geometric(1.0 / scale, size=n).astype(np.int64)
        r[rng.random(n) < 0.3] = 0                       # compulsory misses
        caps = np.array([1, 2, 4, 8, 16, 64, 256], dtype=np.int64)
        hist = DistanceHistogram().add(r)
        assert np.array_equal(hist.mrc(caps), mrc_curve(r, caps))


def mae_of(rows: np.ndarray, truth: np.ndarray) -> float:
    """MAE of the curve averaged from `rows`, the way `predict_mrc` reports it."""
    caps = np.arange(1, rows.shape[1] + 1)
    return float(curve_metrics(rows.mean(axis=0), truth, caps)["mae"])


def test_sampled_score_is_unbiased_and_covered() -> None:
    """A 10 % request sample reproduces the full-window MAE, and the CI covers it.

    The rows carry a real model error (0.02 above the truth) because that is the
    regime the sample is used in: sampling noise must stay small against it. A
    noiseless-error variant is the wrong test, since bootstrapping `|mean - truth|`
    is biased upward exactly when the error is smaller than the noise.
    """
    rng = np.random.default_rng(11)
    n, caps_n = 400_000, 6
    caps = np.arange(1, caps_n + 1)
    truth = np.linspace(0.6, 0.1, caps_n)
    rows = np.clip(truth[None, :] + 0.02 + rng.normal(0, 0.05, (n, caps_n)), 0, 1)
    full = mae_of(rows, truth)
    hits, deltas = 0, []
    for rep in range(100):
        idx = np.sort(rng.choice(n, 40_000, replace=False))
        sub = rows[idx]
        deltas.append(mae_of(sub, truth) - full)
        lo, hi = curve_metrics_ci(sub, truth, caps, seed=rep)
        hits += int(lo <= full <= hi)
    assert hits >= 95, hits
    assert abs(float(np.mean(deltas))) < 0.02 * full
    assert max(abs(d) for d in deltas) < 0.1 * full


def test_stride_keeps_blocks_and_order() -> None:
    oot = _load_oot()
    windows = ([("train", i, i + 10) for i in range(20)]
               + [("valid", 100 + i, 110 + i) for i in range(6)]
               + [("test", 200 + i, 210 + i) for i in range(10)])
    thinned = oot.thin_windows(windows, stride=3)
    for block in ("train", "valid", "test"):
        got = [w for w in thinned if w[0] == block]
        want = [w for w in windows if w[0] == block][::3]
        assert got == want                      # same block, same order, every 3rd
    # a block that is absent stays absent: striding cannot create a later period
    only_train = oot.thin_windows([w for w in windows if w[0] == "train"], stride=2)
    assert {w[0] for w in only_train} == {"train"}
    # max_windows is applied per block, not across blocks
    capped = oot.thin_windows(windows, stride=1, max_windows=2)
    assert [sum(1 for w in capped if w[0] == b) for b in ("train", "valid", "test")] == [2, 2, 2]
    # stride 1 with no cap is the identity, so the default path is unchanged
    assert oot.thin_windows(windows, stride=1) == windows


@pytest.mark.parametrize("stride", [1, 2, 7])
def test_stride_is_a_subsequence_of_each_block(stride: int) -> None:
    oot = _load_oot()
    windows = [(b, i, i + 1) for b in ("train", "valid", "test") for i in range(11)]
    thinned = oot.thin_windows(windows, stride=stride)
    for block in ("train", "valid", "test"):
        got = [w for w in thinned if w[0] == block]
        want = [w for w in windows if w[0] == block][::stride]
        assert got == want


def _toy_parts(rng, names: list[str], n: int = 20_000):
    parts = []
    for block, n_win in (("train", 2), ("valid", 2), ("test", 1)):
        for _ in range(n_win):
            r = rng.geometric(1.0 / 300.0, size=n).astype(np.int64)
            el = rng.random(n) < 0.75
            x = rng.normal(size=(n, len(names))).astype(np.float32)
            parts.append((block, {"R": r, "eligible": el}, x))
    return parts


def test_windows_objective_runs_and_keeps_exact_truth() -> None:
    """The study objective scores the kept valid windows and returns their mean MAE."""
    import optuna

    from mrc.eval.mrc import mrc_curve
    from mrc.pipeline import prepare_windows
    from mrc.tuning.search import make_windows_objective

    rng = np.random.default_rng(5)
    names = ["f0", "f1", "f2"]
    parts = _toy_parts(rng, names)
    prep = prepare_windows("toy", iter(parts), names, nbins=8, row_budget=50_000,
                           n_train_windows=2, keep_scored="valid",
                           scored_per_window=2_000)
    assert len(prep.scored) == 2
    # every kept window carries its whole histogram, so its truth is exact
    valid_r = [p[1]["R"] for p in parts if p[0] == "valid"]
    for w, r in zip(prep.scored, valid_r, strict=True):
        assert np.array_equal(w.histogram.mrc(prep.capacities),
                              mrc_curve(r, prep.capacities))

    study = optuna.create_study(direction="minimize")
    study.optimize(make_windows_objective(prep, num_boost_round=5), n_trials=1)
    assert np.isfinite(study.best_value)
    attrs = study.trials[0].user_attrs
    assert len(attrs["valid_mae_per_window"]) == 2
    assert attrs["n_requests_scored"] == [2_000, 2_000]


def test_objective_without_scored_windows_refuses() -> None:
    from mrc.pipeline import prepare_windows
    from mrc.tuning.search import make_windows_objective

    rng = np.random.default_rng(6)
    names = ["f0"]
    prep = prepare_windows("toy", iter(_toy_parts(rng, names, 2_000)), names, nbins=4,
                           row_budget=5_000, n_train_windows=2)
    with pytest.raises(ValueError, match="no scored windows"):
        make_windows_objective(prep)
