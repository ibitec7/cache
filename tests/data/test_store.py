"""Round-trip contract for the sharded parquet store.

The store is an alternative reader for the same trace, so the gate is array
equality against `read_trace`, not file equality: a window read from any part must
produce the same timestamps, key codes, sizes, clients, operations and TTLs.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from mrc.data.store import Store, StoreShard, write_store
from mrc.data.trace import read_trace

ROWS = [
    ["0", "k1", "10", "100", "1", "get", "0"],
    ["1", "k2", "20", "200", "2", "set", "60"],
    ["2", "k1", "10", "100", "1", "get", "0"],
    ["3", "k3", "30", "300", "3", "gets", "30"],
    ["4", "k1", "10", "100", "1", "get", "0"],
    ["5", "k2", "20", "200", "2", "get", "0"],
    ["6", "k4", "40", "400", "4", "set", "90"],
    ["7", "k3", "30", "300", "3", "get", "0"],
    ["8", "k1", "10", "100", "1", "get", "0"],
    ["9", "k2", "20", "200", "2", "get", "0"],
]


def _csv(tmp_path):
    path = tmp_path / "trace.csv"
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(ROWS)
    return path


def _store(tmp_path, rows_per_part=4, row_group_rows=2):
    return write_store(_csv(tmp_path), tmp_path / "store", rows_per_part=rows_per_part,
                       row_group_rows=row_group_rows)


def test_the_store_reproduces_the_csv(tmp_path):
    meta = _store(tmp_path)
    assert meta["rows"] == len(ROWS)
    assert [p["rows"] for p in meta["parts"]] == [4, 4, 2]
    full = Store(tmp_path / "store").read_trace()
    ref = read_trace(_csv(tmp_path))
    for name in ("ts", "key", "obj_size", "client", "op", "ttl"):
        assert np.array_equal(getattr(full, name), getattr(ref, name)), name
    assert np.array_equal(full.key_vocab, ref.key_vocab)


def test_a_window_matches_the_same_rows_from_the_csv(tmp_path):
    _store(tmp_path)
    window = Store(tmp_path / "store").read_trace(2, 7)
    assert window.ts.tolist() == [2, 3, 4, 5, 6]
    assert window.key_vocab.tolist() == ["k1", "k3", "k2", "k4"]  # block-local codes
    assert window.key.tolist() == [0, 1, 0, 2, 3]
    assert window.n_keys == 4


def test_blocks_cover_every_row_once(tmp_path):
    _store(tmp_path)
    seen = []
    for first, table in Store(tmp_path / "store").iter_blocks(3):
        assert table.num_rows == min(3, len(ROWS) - first)
        seen.extend(table.column("timestamp").to_pylist())
    assert seen == [int(row[0]) for row in ROWS]


def test_a_shard_is_addressed_without_reading_the_other_parts(tmp_path):
    _store(tmp_path)
    shard = StoreShard(tmp_path / "store", start=4, rows=4)
    assert shard.bounds() == (4, 8)
    assert shard.read().ts.tolist() == [4, 5, 6, 7]
    assert shard.key().endswith("#4+4")


def test_a_window_past_the_end_is_clipped(tmp_path):
    _store(tmp_path)
    assert StoreShard(tmp_path / "store", start=8, rows=10).bounds() == (8, 10)


def test_a_range_outside_the_store_is_refused(tmp_path):
    _store(tmp_path)
    with pytest.raises(ValueError):
        Store(tmp_path / "store").read_range(0, len(ROWS) + 1)
    with pytest.raises(ValueError):
        StoreShard(tmp_path / "store", start=12, rows=4).bounds()
