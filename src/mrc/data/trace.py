"""Reading Twitter cache-trace CSV shards into typed columns.

Trace contract (fixed by the dataset):
    timestamp, anonymized key, key size (B), value size (B), client id, operation, TTL

The reader is deliberately the only place that knows this layout. Everything
downstream consumes the returned `Trace` arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

COLUMNS = ["timestamp", "key", "key_size", "value_size", "client_id", "operation", "ttl"]

OPS = ["get", "gets", "set", "add", "replace", "cas", "append", "prepend", "delete", "incr", "decr"]

_CONVERT = pacsv.ConvertOptions(
    column_types={
        "timestamp": pa.int64(),
        "key": pa.string(),
        "key_size": pa.int32(),
        "value_size": pa.int32(),
        "client_id": pa.int32(),
        "operation": pa.string(),
        "ttl": pa.int64(),
    },
    strings_can_be_null=False,
    null_values=[],
)


@dataclass(frozen=True)
class Trace:
    """Columnar trace. `key` holds integer codes, `key_vocab` maps code -> string."""

    ts: np.ndarray          # int64, seconds
    key: np.ndarray         # int32 codes into key_vocab
    obj_size: np.ndarray    # int64 bytes, key_size + value_size
    client: np.ndarray      # int32
    op: np.ndarray          # int8 codes into OPS
    ttl: np.ndarray         # int64 seconds, 0 when not a write
    key_vocab: np.ndarray   # object array of key strings

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def slice(self, start: int, stop: int) -> Trace:
        return Trace(
            ts=self.ts[start:stop],
            key=self.key[start:stop],
            obj_size=self.obj_size[start:stop],
            client=self.client[start:stop],
            op=self.op[start:stop],
            ttl=self.ttl[start:stop],
            key_vocab=self.key_vocab,
        )

    @property
    def n_keys(self) -> int:
        return len(self.key_vocab)


def _open(path: str | Path) -> pa.NativeFile:
    p = Path(path)
    raw = pa.OSFile(str(p), "rb")
    if p.suffix == ".zst":
        return pa.CompressedInputStream(raw, "zstd")
    return raw


def read_trace(path: str | Path, limit: int | None = None, block_size: int = 1 << 26) -> Trace:
    """Read a trace shard (plain CSV or .zst) into a `Trace`."""
    read_opts = pacsv.ReadOptions(column_names=COLUMNS, block_size=block_size)
    tbl = pacsv.read_csv(
        _open(path), read_options=read_opts, convert_options=_CONVERT,
        parse_options=pacsv.ParseOptions(delimiter=","),
    )
    if limit is not None and tbl.num_rows > limit:
        tbl = tbl.slice(0, limit)

    keys = tbl.column("key").combine_chunks()
    if not pa.types.is_string(keys.type):
        keys = keys.cast(pa.string())
    encoded = keys.dictionary_encode()  # one dictionary over the whole column
    key_codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    key_vocab = encoded.dictionary.to_numpy(zero_copy_only=False)
    del keys, encoded

    op_codes = pc.index_in(tbl.column("operation"), value_set=pa.array(OPS)).to_numpy(
        zero_copy_only=False).astype(np.int8, copy=False)

    obj = (tbl.column("key_size").to_numpy(zero_copy_only=False).astype(np.int64)
           + tbl.column("value_size").to_numpy(zero_copy_only=False).astype(np.int64))

    return Trace(
        ts=tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
        key=key_codes,
        obj_size=obj,
        client=tbl.column("client_id").to_numpy(zero_copy_only=False).astype(np.int32, copy=False),
        op=op_codes,
        ttl=tbl.column("ttl").to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
        key_vocab=key_vocab,
    )


def summary(t: Trace) -> dict:
    dt = int(t.ts[-1] - t.ts[0]) if len(t) else 0
    return {
        "requests": len(t),
        "distinct_keys": t.n_keys,
        "span_seconds": dt,
        "req_per_sec": round(len(t) / dt, 1) if dt else None,
        "mean_obj_size": float(t.obj_size.mean()) if len(t) else None,
        "working_set_bytes": int(np.unique(t.key).size and t.obj_size[np.unique(t.key, return_index=True)[1]].sum()),
        "one_hit_wonder_frac": None,
    }
