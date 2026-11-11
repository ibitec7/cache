"""B3: the ground-truth artefact and its provenance manifest."""
from __future__ import annotations

import json

import numpy as np

from mrc.gt.write import written_truth


def _labels(seed: int = 2, n: int = 4000) -> dict:
    rng = np.random.default_rng(seed)
    return {"R": rng.integers(-1, 500, n), "B": rng.integers(-1, 500, n),
            "obj_size": rng.integers(8, 64, n), "n_keys": np.array([2000])}


def test_truth_artifact_roundtrip(tmp_path):
    labels = _labels()
    man = written_truth(tmp_path / "c45.npz", labels, n_keys=2000, overflow_edge=1024)
    meta = json.loads(man.read_text())
    assert meta["n_capacities"] == len(meta["capacities"])
    assert meta["b_M_minus_1"] == 1024
    z = np.load(tmp_path / "c45.npz")
    assert z["mrc"].shape == z["capacities"].shape
    assert np.all(np.diff(z["mrc"]) <= 1e-12)
    assert 0.0 <= float(z["mrc"].min()) and float(z["mrc"].max()) <= 1.0
    assert z["capacities"].max() <= meta["b_M_minus_1"] - 1


def test_manifest_recomputes_from_the_written_curve(tmp_path):
    labels = _labels(seed=3)
    man = written_truth(tmp_path / "t", labels, n_keys=2000, overflow_edge=1024, points=16,
                        source_labels=tmp_path / "labels.npz")
    meta = json.loads(man.read_text())
    z = np.load(tmp_path / "t.npz")
    assert meta["max_capacity"] == int(z["capacities"][-1])
    assert meta["grid_points_requested"] == 16
    assert meta["overflow_edge"] == 1024
    assert meta["n_requests"] == labels["R"].shape[0]
    assert meta["p_compulsory"] == float((labels["R"] < 0).mean())
    assert meta["source_labels"].endswith("labels.npz")
    assert meta["labels_sha256"] == ""          # no manifest beside the toy labels
    assert len(meta["git_sha"]) >= 7
    assert meta["created_utc"].endswith("Z")
