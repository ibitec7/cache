"""Sharded zstd-parquet store for one raw trace shard.

Why
---
A raw shard is a single 86.7 GiB zstd CSV that expands to 608 GiB and holds 13.4
billion requests. The pipeline only ever needs contiguous request windows, so
the store cuts the shard into fixed-size parts and reads a window by row group.

Input contract: a trace readable by `mrc.data.trace.read_trace` (plain CSV or `.zst`)
with the fixed eight-column layout.
Output contract: `<root>/manifest.json` plus `part-00000.parquet`, ... Each part carries
the same column names and types as the CSV, `key` and `operation` are dictionary
encoded, and the whole part is zstd compressed. `manifest.json` maps request ranges
to part files, so a reader never scans a part it does not need.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from mrc.data.trace import _CONVERT, COLUMNS, OPS, Trace, open_stream

STORE_VERSION = 1
PART_FILE = "part-{index:05d}.parquet"
MANIFEST = "manifest.json"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_store(
    source: str | Path,
    root: str | Path,
    *,
    limit: int | None = None,
    rows_per_part: int = 5_000_000,
    row_group_rows: int = 1_000_000,
    compression: str = "zstd",
    compression_level: int = 3,
    block_size: int = 1 << 26,
) -> dict:
    """Convert a CSV shard into zstd parquet parts and write the manifest.

    Parsing is streamed in Arrow batches; at most one part is held in memory. A
    part is written as soon as it reaches `rows_per_part`, so peak memory is set by
    the part size, not by the shard size.
    """
    src = Path(source)
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    read_opts = pacsv.ReadOptions(column_names=COLUMNS, block_size=block_size)
    parse_opts = pacsv.ParseOptions(delimiter=",")

    parts: list[dict] = []
    rows_total = 0
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    index = 0

    def flush() -> None:
        nonlocal pending, pending_rows, index
        if not pending:
            return
        table = pa.Table.from_batches(pending)
        path = out / PART_FILE.format(index=index)
        pq.write_table(
            table,
            path,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_rows,
            use_dictionary=["key", "operation"],
            dictionary_pagesize_limit=64 << 20,
        )
        parts.append({"file": path.name, "start": rows_total - pending_rows,
                      "rows": pending_rows, "bytes": path.stat().st_size})
        index += 1
        pending = []
        pending_rows = 0

    remaining = None if limit is None else int(limit)
    with pacsv.open_csv(open_stream(src), read_options=read_opts,
                        convert_options=_CONVERT, parse_options=parse_opts) as reader:
        for batch in reader:
            if remaining is not None:
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            while batch.num_rows > 0:
                take = min(rows_per_part - pending_rows, batch.num_rows)
                pending.append(batch.slice(0, take))
                pending_rows += take
                rows_total += take
                batch = batch.slice(take)
                if remaining is not None:
                    remaining -= take
                if pending_rows == rows_per_part:
                    flush()
            if remaining is not None and remaining <= 0:
                break
    flush()

    meta = {
        "store_version": STORE_VERSION,
        "source": str(src),
        "source_bytes": src.stat().st_size,
        "rows": rows_total,
        "limit": limit,
        "rows_per_part": rows_per_part,
        "row_group_rows": row_group_rows,
        "compression": compression,
        "compression_level": compression_level,
        "columns": COLUMNS,
        "parts": parts,
        "created_utc": _utc_now(),
        "pyarrow": pa.__version__,
    }
    (out / MANIFEST).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


@dataclass(frozen=True)
class Part:
    """One parquet file and the half-open request range it covers."""

    path: Path
    start: int
    rows: int

    @property
    def stop(self) -> int:
        return self.start + self.rows


class Store:
    """Read request windows from a store written by `write_store`."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / MANIFEST).read_text())
        self.parts = tuple(
            Part(self.root / p["file"], int(p["start"]), int(p["rows"]))
            for p in self.meta["parts"]
        )
        self.n_rows = int(self.meta["rows"])
        expected = 0
        for part in self.parts:
            if part.start != expected:
                raise ValueError(f"store part {part.path.name} starts at {part.start}, "
                                 f"expected {expected}")
            expected = part.stop
        if expected != self.n_rows:
            raise ValueError("store parts do not cover the manifest row count")

    def _covering(self, start: int, stop: int) -> Iterator[Part]:
        for part in self.parts:
            if part.stop <= start:
                continue
            if part.start >= stop:
                break
            yield part

    def read_range(self, start: int, stop: int,
                   columns: Sequence[str] | None = None) -> pa.Table:
        """Read requests `[start, stop)` as a table, touching only their row groups."""
        if not 0 <= start < stop <= self.n_rows:
            raise ValueError(f"range [{start}, {stop}) outside [0, {self.n_rows})")
        chunks: list[pa.Table] = []
        for part in self._covering(start, stop):
            local_start = max(start, part.start) - part.start
            local_stop = min(stop, part.stop) - part.start
            handle = pq.ParquetFile(part.path)
            offset = 0
            for group in range(handle.metadata.num_row_groups):
                group_rows = handle.metadata.row_group(group).num_rows
                group_start, group_stop = offset, offset + group_rows
                offset = group_stop
                if group_stop <= local_start:
                    continue
                if group_start >= local_stop:
                    break
                table = handle.read_row_group(group, columns=columns)
                lo = max(local_start, group_start) - group_start
                hi = min(local_stop, group_stop) - group_start
                chunks.append(table.slice(lo, hi - lo))
        if not chunks:
            raise ValueError(f"range [{start}, {stop}) selected no row groups")
        return pa.concat_tables(chunks)

    def _rows(self, start: int, stop: int) -> Iterator[tuple[pq.ParquetFile, int, int, int]]:
        """Yield `(handle, row_group, local_lo, local_hi)` for the row groups a range touches."""
        for part in self._covering(start, stop):
            local_start = max(start, part.start) - part.start
            local_stop = min(stop, part.stop) - part.start
            handle = pq.ParquetFile(part.path)
            offset = 0
            for group in range(handle.metadata.num_row_groups):
                rows = handle.metadata.row_group(group).num_rows
                group_start, group_stop = offset, offset + rows
                offset = group_stop
                if group_stop <= local_start or group_start >= local_stop:
                    continue
                yield (handle, group, max(local_start, group_start) - group_start,
                       min(local_stop, group_stop) - group_start)

    def _column(self, start: int, stop: int, name: str, dtype) -> np.ndarray:
        """One column as a preallocated numpy array, filled row group by row group."""
        out = np.empty(stop - start, dtype=dtype)
        pos = 0
        for handle, group, lo, hi in self._rows(start, stop):
            values = handle.read_row_group(group, columns=[name]).column(name)
            out[pos:pos + hi - lo] = values.slice(lo, hi - lo).to_numpy(zero_copy_only=False)
            pos += hi - lo
        return out

    def _key_codes(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        chunks = [handle.read_row_group(group, columns=["key"]).column("key").slice(lo, hi - lo)
                  for handle, group, lo, hi in self._rows(start, stop)]
        keys = pa.chunked_array(chunks).combine_chunks()
        if not pa.types.is_string(keys.type):
            keys = keys.cast(pa.string())
        encoded = keys.dictionary_encode()
        codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        vocab = encoded.dictionary.to_numpy(zero_copy_only=False)
        return codes, vocab

    def read_trace(self, start: int = 0, stop: int | None = None) -> Trace:
        """Read requests `[start, stop)` as a `Trace` with a window-local vocabulary.

        Columns are filled into preallocated arrays per row group, so peak memory is
        the window's arrays plus the key strings, not a materialised table of every
        column and a second copy from `pa.Table.from_batches`.
        """
        stop = self.n_rows if stop is None else stop
        if not 0 <= start < stop <= self.n_rows:
            raise ValueError(f"range [{start}, {stop}) outside [0, {self.n_rows})")
        key, vocab = self._key_codes(start, stop)
        values = pa.array(OPS)
        op = np.empty(stop - start, dtype=np.int8)
        pos = 0
        for handle, group, lo, hi in self._rows(start, stop):
            column = handle.read_row_group(group, columns=["operation"]).column("operation")
            codes = pc.index_in(column.slice(lo, hi - lo), value_set=values).to_numpy(
                zero_copy_only=False).astype(np.int8, copy=False)
            op[pos:pos + hi - lo] = codes
            pos += hi - lo
        obj = (self._column(start, stop, "key_size", np.int64)
               + self._column(start, stop, "value_size", np.int64))
        return Trace(
            ts=self._column(start, stop, "timestamp", np.int64),
            key=key,
            obj_size=obj,
            client=self._column(start, stop, "client_id", np.int32),
            op=op,
            ttl=self._column(start, stop, "ttl", np.int64),
            key_vocab=vocab,
        )

    def iter_blocks(self, rows: int, start: int = 0, stop: int | None = None,
                    columns: Sequence[str] | None = None) -> Iterator[tuple[int, pa.Table]]:
        """Yield `(first_request_index, table)` blocks of at most `rows` requests."""
        if rows <= 0:
            raise ValueError("block size must be positive")
        stop = self.n_rows if stop is None else stop
        for first in range(start, stop, rows):
            yield first, self.read_range(first, min(first + rows, stop), columns=columns)

@dataclass(frozen=True)
class StoreShard:
    """A request window in a store, addressed without touching the other parts."""

    root: str | Path
    start: int = 0
    rows: int | None = None

    def bounds(self) -> tuple[int, int]:
        store = Store(self.root)
        stop = (store.n_rows if self.rows is None
                else min(self.start + int(self.rows), store.n_rows))
        if not 0 <= self.start < stop <= store.n_rows:
            raise ValueError(f"shard [{self.start}, {stop}) outside [0, {store.n_rows})")
        return self.start, stop

    def read(self) -> Trace:
        start, stop = self.bounds()
        return Store(self.root).read_trace(start, stop)

    def key(self) -> str:
        start, stop = self.bounds()
        return f"store:{Path(self.root).resolve()}#{start}+{stop - start}"
