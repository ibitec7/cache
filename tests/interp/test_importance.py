"""Feature attribution: permutation importance and SHAP contributions (Phase G1, G2)."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from mrc.interp.importance import permutation_importance, shap_summary


def test_importance_ranks_the_informative_column_first() -> None:
    rng = np.random.default_rng(10)
    n = 4000
    signal = rng.normal(size=n).astype(np.float32)
    noise = rng.normal(size=(n, 3)).astype(np.float32)
    X = np.column_stack([signal, noise])
    y = (signal > 0).astype(np.float32)

    def score(Xv: np.ndarray) -> float:
        # a fixed, monotone-in-signal scoring rule; no model needed for the unit test
        return float(np.mean((Xv[:, 0] > 0).astype(np.float32) == y))

    imp = permutation_importance(
        score, X, names=["signal", "a", "b", "c"], n_repeats=5, seed=0
    )
    assert imp[0]["feature"] == "signal"
    assert imp[0]["drop"] > 0.2
    assert len(imp) == 4
    assert all("std" in row for row in imp)
    assert [row["drop"] for row in imp] == sorted((row["drop"] for row in imp), reverse=True)


@pytest.fixture(scope="module")
def shap_problem() -> tuple[lgb.Booster, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (X[:, 0] + 0.2 * rng.normal(size=3000) > 0).astype(np.float32)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 2},
        lgb.Dataset(X, label=y),
        num_boost_round=30,
    )
    return booster, X


def test_shap_writes_the_top_table(
    shap_problem: tuple[lgb.Booster, np.ndarray], tmp_path: Path
) -> None:
    booster, X = shap_problem
    out = shap_summary(
        booster, X, names=["a", "b", "c", "d"], top=2, path=tmp_path / "shap.json"
    )
    meta = json.loads(out.read_text())
    assert meta["top"][0]["feature"] == "a"
    assert len(meta["top"]) == 2
    assert meta["n_rows"] == 3000


def test_shap_full_ranking_and_name_count_guard(
    shap_problem: tuple[lgb.Booster, np.ndarray], tmp_path: Path
) -> None:
    booster, X = shap_problem
    names = ["a", "b", "c", "d"]
    out = shap_summary(booster, X, names=names, top=12, path=tmp_path / "full.json")
    meta = json.loads(out.read_text())
    assert len(meta["top"]) == len(names)
    assert meta["top"][0]["feature"] == "a"
    assert all(row["mean_abs_contribution"] >= 0.0 for row in meta["top"])
    with pytest.raises(ValueError):
        shap_summary(booster, X, names=["a", "b", "c"], path=tmp_path / "bad.json")
