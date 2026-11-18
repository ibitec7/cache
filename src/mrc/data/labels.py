"""Label cache: reuse columns, the request columns they join to, and a sha256 manifest.

The cache is identified by the triple `(source, limit, cap)`. Its npz is pickle-free: every array is
numeric or boolean, so `np.load(..., allow_pickle=False)` always works. Array equality is the
reproducibility invariant; file bytes are not, because `np.savez_compressed` stamps zip entries with
the wall clock.

Input contract: a trace shard readable by `mrc.data.trace.read_trace` (plain CSV or `.zst`), or an in-memory `Trace` from a parquet store window (`mrc.data.store`).
Output contract: `LABEL_KEYS` plus `REQUEST_KEYS` plus `n_keys`, and a `<stem>.manifest.json` beside
the npz carrying sha256, row and key counts, the cache key, the event counts, the git sha and the
creation time.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mrc.config import CAP_QUANTILE, REPO
from mrc.data.reuse import compute_reuse
from mrc.data.trace import Trace, read_trace

LABEL_KEYS: tuple[str, ...] = (
    "R", "B", "gap", "freq", "gamma", "rtilde", "eligible", "delta", "compulsory",
    "key_ns", "key_len",
)
REQUEST_KEYS: tuple[str, ...] = ("ts", "key", "obj_size", "op", "client", "ttl")

# 63 token buckets plus 0 reserved for an empty key, so the column has 64 categorical values.
NS_BUCKETS = 63


def _key_codes(vocab: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coarse token bucket and byte length per request, from the key vocabulary.

    `key_ns` is a deterministic bucket of the key string, `1 + crc32(key) % 63`, with 0 reserved for
    an empty key. It is a coarse bucket, not a namespace: the measured Twitter keys are opaque
    10-character tokens with no `:` separator, so their "first component" is the whole key and a
    distinct category per key (173,874 on cluster45) is too many for a LightGBM categorical. The
    bucket is a function of the key string only and never of the target.
    """
    n_vocab = int(vocab.shape[0])
    ns_table = np.zeros(n_vocab, dtype=np.int32)
    len_table = np.zeros(n_vocab, dtype=np.int32)
    for i, entry in enumerate(vocab):
        raw = str(entry).encode()
        len_table[i] = len(raw)
        if raw:
            ns_table[i] = 1 + zlib.crc32(raw) % NS_BUCKETS
    idx = np.asarray(keys, dtype=np.int64)
    return ns_table[idx], len_table[idx]


def labels_from_trace(trace: Trace, cap: int = CAP_QUANTILE) -> dict[str, np.ndarray]:
    """Return label columns plus the request columns they join to, from an in-memory trace."""
    derived = compute_reuse(trace.key, trace.obj_size, trace.n_keys, cap=cap)
    key_ns, key_len = _key_codes(trace.key_vocab, trace.key)
    geometry = {"key_ns": key_ns, "key_len": key_len}

    labels: dict[str, np.ndarray] = {
        name: geometry[name] if name in geometry else derived[name] for name in LABEL_KEYS
    }
    labels.update({name: getattr(trace, name) for name in REQUEST_KEYS})
    labels["n_keys"] = np.array([trace.n_keys], dtype=np.int64)
    return labels


def build_labels(
    path: str | Path,
    limit: int | None = None,
    cap: int = CAP_QUANTILE,
) -> dict[str, np.ndarray]:
    """Read a shard prefix and return label columns plus the request columns they join to."""
    return labels_from_trace(read_trace(path, limit=limit), cap=cap)


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _git_sha() -> str:
    """Short HEAD of the repository this module lives in, or "unknown" outside a work tree."""
    done = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return done.stdout.strip() or "unknown"


def save_labels(
    out: str | Path,
    labels: dict[str, np.ndarray],
    *,
    source: str | Path,
    limit: int | None,
    cap: int,
) -> Path:
    """Write the npz and its manifest; return the manifest path."""
    npz_path = Path(out)
    if npz_path.suffix != ".npz":
        raise ValueError(f"label cache path must end in .npz, got {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **labels)  # type: ignore[arg-type]

    eligible = np.asarray(labels["eligible"], dtype=bool)
    delta = np.asarray(labels["delta"], dtype=bool)
    meta = {
        "source": str(source),
        "npz": str(npz_path),
        "sha256": _sha256(npz_path),
        "n_requests": int(np.asarray(labels["R"]).shape[0]),
        "n_keys": int(np.asarray(labels["n_keys"]).reshape(-1)[0]),
        "limit": None if limit is None else int(limit),
        "cap": int(cap),
        "eligible": int(eligible.sum()),
        "events": int(delta.sum()),
        "p_compulsory": float((~eligible).mean()),
        "feature_source": "key_vocab",
        "git_sha": _git_sha(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest = npz_path.with_name(npz_path.stem + ".manifest.json")
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    return manifest


def load_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Load a label npz without pickling; keys are the array names."""
    with np.load(Path(path), allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}
