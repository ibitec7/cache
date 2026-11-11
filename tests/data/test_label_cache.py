"""Cache identity for the label columns: pickle-free npz plus a recomputable manifest."""
import json

import numpy as np

from mrc.data.labels import LABEL_KEYS, REQUEST_KEYS, build_labels, load_labels, save_labels

MANIFEST_FIELDS = {
    "source", "npz", "sha256", "n_requests", "n_keys", "limit", "cap", "eligible", "events",
    "p_compulsory", "feature_source", "git_sha", "created_utc",
}


def _toy_csv(tmp_path):
    p = tmp_path / "toy.csv"
    p.write_text("0,aaa,10,0,1,get,0\n1,bbb,10,0,1,get,0\n2,aaa,10,0,2,get,0\n"
                 "3,ccc,10,0,1,get,0\n4,aaa,10,0,1,get,0\n")
    return p


def test_build_save_load_roundtrip(tmp_path):
    src = _toy_csv(tmp_path)
    lab = build_labels(src, limit=None, cap=7)
    assert lab["R"].shape == (5,)
    assert lab["rtilde"].dtype == np.int64
    out = tmp_path / "labels.npz"
    man = save_labels(out, lab, source=src, limit=None, cap=7)
    meta = json.loads(man.read_text())
    assert meta["n_requests"] == 5 and meta["cap"] == 7
    assert len(meta["sha256"]) == 64
    assert np.array_equal(load_labels(out)["rtilde"], lab["rtilde"])


def test_cache_columns_are_pickle_free_and_carry_key_geometry(tmp_path):
    lab = build_labels(_toy_csv(tmp_path), limit=None, cap=7)
    assert lab["key"].dtype == np.int32
    assert lab["key_ns"].dtype == np.int32 and lab["key_len"].dtype == np.int32
    assert lab["key_ns"].min() >= 1            # "aaa", "bbb", "ccc" all hash above the reserve
    assert lab["key_len"].tolist() == [3, 3, 3, 3, 3]
    out = tmp_path / "l.npz"
    save_labels(out, lab, source="toy", limit=None, cap=7)
    reloaded = load_labels(out)
    assert not any(reloaded[k].dtype == object for k in reloaded)


def test_manifest_carries_every_field_and_recomputes_from_the_arrays(tmp_path):
    src = _toy_csv(tmp_path)
    lab = build_labels(src, limit=None, cap=7)
    meta = json.loads(save_labels(tmp_path / "labels.npz", lab, source=src, limit=5, cap=7)
                      .read_text())
    assert MANIFEST_FIELDS <= set(meta)
    assert meta["n_requests"] == len(lab["R"])
    assert meta["n_keys"] == int(lab["n_keys"][0]) == 3
    assert meta["eligible"] == int(lab["eligible"].sum())
    assert meta["events"] == int(lab["delta"].sum())
    assert meta["p_compulsory"] == float((~lab["eligible"]).mean())
    assert meta["limit"] == 5 and meta["cap"] == 7 and meta["feature_source"] == "key_vocab"
    assert meta["npz"] == str(tmp_path / "labels.npz")
    assert set(load_labels(tmp_path / "labels.npz")) == set(LABEL_KEYS) | set(REQUEST_KEYS) | {
        "n_keys"}


def test_key_codes_are_a_function_of_the_key_string_only(tmp_path):
    src = _toy_csv(tmp_path)
    lab = build_labels(src, limit=None, cap=7)
    # same key code -> same geometry, regardless of row position or target
    by_key = {}
    for code, ns, ln in zip(lab["key"], lab["key_ns"], lab["key_len"]):
        by_key.setdefault(int(code), set()).add((int(ns), int(ln)))
    assert all(len(v) == 1 for v in by_key.values())
    assert lab["key_ns"].max() <= 63 and lab["key_ns"].min() >= 1
    assert lab["eligible"].tolist() == [False, False, True, False, True]
    assert lab["delta"].tolist() == [False, False, True, False, False]
