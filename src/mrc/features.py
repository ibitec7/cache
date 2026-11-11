"""The 12-column design matrix the hazard model consumes.

Input contract: a label dict from `mrc.data.labels.build_labels` (or the same columns from a cached
npz). The columns read are `obj_size`, `ttl`, `client`, `op`, `gap`, `freq`, `R` (row count only) and
either `key` as an object array of key strings, or `key` as int32 codes together with the cached
`key_ns` and `key_len` columns. `delta`, `rtilde` and the rest of the target never enter the matrix.

Output contract: `(X, names, cat_idx)` with `X` float32 `(n, 12)`, finite, columns in `FEATURE_NAMES`
order, and `cat_idx` the indices of `CATEGORICAL`.

`key_ns` is a coarse token bucket, not a namespace: the measured Twitter keys are opaque 10-character
tokens with no `:` separator, so one category per key is unusable for a GBDT. On the object-key path
the first `:`-delimited component is coded instead, which is the exact partition; on the cached path
the bucket is `1 + crc32(key) % 63`. Equal components therefore always share a code, while distinct
components may share one. Reordering or renaming a column is a `FEATURE_VERSION` bump.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION: int = 1

FEATURE_NAMES: tuple[str, ...] = (
    "log_obj_size",
    "log_key_size",
    "log_val_size",
    "ns_code",
    "op",
    "is_write",
    "log_ttl",
    "has_ttl",
    "client_bucket",
    "log_gap",
    "log_freq",
    "log_age",
)

CATEGORICAL: tuple[str, ...] = ("ns_code", "op", "is_write")

_READ_OPS = (0, 1)          # trace.OPS: get, gets
CLIENT_BUCKETS = 512


def _namespace_codes(keys: np.ndarray) -> np.ndarray:
    """Code the first ':'-delimited component of each key string; empty component -> 0."""
    heads = np.array([str(k).split(":", 1)[0] for k in np.asarray(keys, dtype=object)],
                     dtype=object)
    codes = np.zeros(heads.shape[0], dtype=np.int32)
    known = np.fromiter((bool(h) for h in heads), dtype=bool, count=heads.shape[0])
    if known.any():
        _, inverse = np.unique(heads[known].astype(str), return_inverse=True)
        codes[known] = inverse.astype(np.int32) + 1     # 0 stays reserved for an empty component
    return codes


def _key_geometry(labels: dict) -> tuple[np.ndarray, np.ndarray]:
    """Namespace code and key byte length, from the key strings or from the cached columns."""
    key = np.asarray(labels["key"])
    if key.dtype == object:
        codes = _namespace_codes(key)
        lengths = np.fromiter((len(str(k).encode()) for k in key), dtype=np.int64,
                              count=key.shape[0])
        return codes, lengths
    if "key_ns" not in labels or "key_len" not in labels:
        raise ValueError("labels carry no key geometry; rebuild them with mrc labels")
    return np.asarray(labels["key_ns"]), np.asarray(labels["key_len"], dtype=np.int64)


def build_features(labels: dict) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (X, names, categorical_indices); raises on non-finite values or missing key geometry."""
    ns_code, key_len = _key_geometry(labels)

    n = int(np.asarray(labels["R"]).shape[0])
    obj_size = np.asarray(labels["obj_size"], dtype=np.float64)
    ttl = np.asarray(labels["ttl"], dtype=np.float64)
    client = np.asarray(labels["client"], dtype=np.int64)
    op = np.asarray(labels["op"])
    gap = np.asarray(labels["gap"], dtype=np.float64)
    freq = np.asarray(labels["freq"], dtype=np.float64)
    key_len_f = key_len.astype(np.float64)

    columns = {
        "log_obj_size": np.log1p(obj_size),
        "log_key_size": np.log1p(key_len_f),
        "log_val_size": np.log1p(np.maximum(obj_size - key_len_f, 0.0)),
        "ns_code": ns_code.astype(np.float64),
        "op": op.astype(np.float64),
        "is_write": (~np.isin(op, _READ_OPS)).astype(np.float64),
        "log_ttl": np.log1p(np.maximum(ttl, 0.0)),
        "has_ttl": (ttl > 0).astype(np.float64),
        "client_bucket": np.log1p(client % CLIENT_BUCKETS).astype(np.float64),
        "log_gap": np.log1p(np.maximum(gap, 0.0)),
        "log_freq": np.log1p(freq),
        "log_age": np.log1p(np.arange(n, dtype=np.float64)),
    }
    X = np.column_stack([columns[name] for name in FEATURE_NAMES]).astype(np.float32)
    if X.shape != (n, len(FEATURE_NAMES)):
        raise ValueError(f"design matrix has shape {X.shape}, expected {(n, len(FEATURE_NAMES))}")
    if not np.isfinite(X).all():
        raise ValueError("design matrix carries non-finite values")

    names = list(FEATURE_NAMES)
    cat_idx = [names.index(name) for name in CATEGORICAL]
    return X, names, cat_idx
