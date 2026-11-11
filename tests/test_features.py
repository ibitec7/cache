"""The 12-column design matrix: fixed order, no target leakage, no silent constants."""
import time

import numpy as np
import pytest

from mrc.data.reuse import compute_reuse
from mrc.features import CATEGORICAL, FEATURE_NAMES, build_features


def _toy():
    keys = np.array([0, 1, 0, 2, 1, 3, 0, 1], dtype=np.int64)
    sizes = np.array([16, 20, 16, 24, 20, 12, 16, 20], dtype=np.int64)
    ttl = np.array([0, 0, 0, 0, 0, 1296000, 0, 0], dtype=np.int64)
    client = np.array([1, 2, 1, 3, 2, 1, 1, 2], dtype=np.int64)
    labels = compute_reuse(keys, sizes, 4, cap=1023)
    labels.update(obj_size=sizes, ttl=ttl, client=client,
                  key=np.array([f"q:q:1:{k}" for k in keys], dtype=object),
                  op=np.array([0, 0, 0, 1, 0, 2, 0, 0], dtype=np.int8),
                  ts=np.arange(8, dtype=np.int64))
    return labels


def test_shape_dtype_and_names():
    X, names, cat_idx = build_features(_toy())
    assert X.dtype == np.float32
    assert X.shape == (8, len(FEATURE_NAMES))
    assert names == list(FEATURE_NAMES)
    assert cat_idx == [names.index("ns_code"), names.index("op"), names.index("is_write")]
    assert not np.isnan(X).any()
    assert np.isfinite(X).all()


def test_values_are_what_the_docstring_says():
    X, names, _ = build_features(_toy())
    col = {n: X[:, i] for i, n in enumerate(names)}
    assert col["log_obj_size"][0] == np.float32(np.log1p(16))
    # op 0/1 are reads (get/gets); op 2 (set) is a write
    assert col["is_write"][5] == 1.0 and col["is_write"][0] == 0.0
    assert col["is_write"][3] == 0.0
    assert col["log_ttl"][5] > 0.0 and col["log_ttl"][0] == 0.0
    assert col["has_ttl"][5] == 1.0 and col["has_ttl"][0] == 0.0
    assert col["log_gap"][0] == 0.0                       # -1 clipped to 0, not log1p(-1)
    assert col["log_freq"][0] == np.float32(np.log1p(1))
    assert col["log_age"][0] == 0.0 and col["log_age"][7] == np.float32(np.log1p(7))


def test_integer_key_labels_use_the_cached_key_geometry():
    lab = _toy()
    lab["key"] = np.array([0, 1, 0, 2, 1, 3, 0, 1], dtype=np.int32)   # real-trace layout
    lab["key_ns"] = np.array([5, 6, 5, 7, 6, 5, 5, 6], dtype=np.int32)
    lab["key_len"] = np.array([7, 7, 7, 7, 7, 7, 7, 7], dtype=np.int32)
    X, names, _ = build_features(lab)
    col = {n: X[:, i] for i, n in enumerate(names)}
    assert len(np.unique(col["ns_code"])) == 3          # not constant
    assert col["log_key_size"][0] == np.float32(np.log1p(7))
    # log_val_size uses the real key length: log1p(16 - 7), not log1p(16 - 8)
    assert col["log_val_size"][0] == np.float32(np.log1p(9))


def test_integer_key_labels_without_geometry_raise():
    lab = _toy()
    lab["key"] = np.array([0, 1, 0, 2, 1, 3, 0, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        build_features(lab)


def test_no_column_is_a_function_of_the_target():
    lab = _toy()
    X, _, _ = build_features(lab)
    lab2 = dict(lab)
    lab2["R"] = lab["R"] * 3
    lab2["rtilde"] = lab["rtilde"] * 3
    lab2["delta"] = ~lab["delta"]
    X2, _, _ = build_features(lab2)
    assert np.array_equal(X, X2)        # R and friends are only ever read for their length


def test_names_and_categorical_contract():
    _, names, _cat_idx = build_features(_toy())
    assert len(FEATURE_NAMES) == 12 and len(set(FEATURE_NAMES)) == 12
    assert CATEGORICAL == ("ns_code", "op", "is_write")
    names.append("mutated")
    assert len(FEATURE_NAMES) == 12     # module tuple untouched by a caller appending


def test_real_key_tokens_code_the_first_component():
    from mrc.features import _namespace_codes
    keys = np.array(["q:q:1:abc", "q:q:1:def", "s:1:zzz", "q:q:1:ghi"], dtype=object)
    codes = _namespace_codes(keys)
    assert codes[0] == codes[1] == codes[3]
    assert codes[2] != codes[0]
    assert codes.min() >= 1
    # opaque tokens with no separator, the measured real-trace shape
    tokens = np.array(["tSM5KAkocE", "i8r0qbmiu3", "tSM5KAkocE"], dtype=object)
    t_codes = _namespace_codes(tokens)
    assert t_codes[0] == t_codes[2] != t_codes[1]


def test_the_two_coders_are_functions_of_the_key_string_only():
    from mrc.data.labels import _key_codes
    from mrc.features import _namespace_codes
    vocab = np.array(["q:q:1:a", "s:1:b", "q:q:1:c", "z:9:d"], dtype=object)
    keys = np.array([0, 1, 2, 3, 0, 2], dtype=np.int64)
    ns_col, len_col = _key_codes(vocab, keys)
    ns_str = _namespace_codes(vocab[keys])
    # the same key string gets the same code on both paths, whatever the row position
    for i in range(len(keys)):
        for j in range(len(keys)):
            if vocab[keys[i]] == vocab[keys[j]]:
                assert ns_col[i] == ns_col[j]
                assert ns_str[i] == ns_str[j]
    # the string path is the exact first-component partition; the cached path is a coarse
    # bucket of the whole key (a 63-bucket hash cannot be injective, so it may merge
    # distinct components, which is why the cross-path assertion is one-directional)
    assert len(np.unique(ns_str)) == 3                       # "q", "s", "z"
    assert ns_col.min() >= 1 and ns_col.max() <= 63
    assert np.array_equal(len_col, np.array([len(vocab[k].encode()) for k in keys]))


def test_two_hundred_thousand_rows_build_under_two_seconds():
    rng = np.random.default_rng(0)
    n = 200_000
    tokens = np.array([f"t{rng.integers(0, 1 << 40):010x}" for _ in range(2000)], dtype=object)
    codes = rng.integers(0, 2000, n)
    labels = {
        "R": rng.integers(-1, 5000, n).astype(np.int64),
        "key": tokens[codes],
        "obj_size": rng.integers(8, 1 << 16, n).astype(np.int64),
        "ttl": rng.integers(0, 1 << 21, n).astype(np.int64),
        "client": rng.integers(0, 1 << 20, n).astype(np.int32),
        "op": rng.integers(0, 3, n).astype(np.int8),
        "gap": rng.integers(-1, 5000, n).astype(np.int64),
        "freq": rng.integers(1, 100, n).astype(np.int64),
    }
    start = time.perf_counter()
    X, _names, _cat_idx = build_features(labels)
    elapsed = time.perf_counter() - start
    assert X.shape == (n, 12) and X.dtype == np.float32
    assert np.isfinite(X).all()
    assert elapsed < 2.0, f"200,000 rows took {elapsed:.3f} s, budget 2.0 s"
