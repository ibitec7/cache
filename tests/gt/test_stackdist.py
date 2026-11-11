"""B2: exact stack-distance MRC and the identification-limited capacity grid.

The object curve is checked against LRU replay with `array_equal`: the stack
distance theorem is an identity, not an approximation.
"""
from __future__ import annotations

import numpy as np

from mrc.data.reference import naive_lru_mrc
from mrc.data.reuse import compute_reuse
from mrc.gt.stackdist import identified_grid, mrc_from_labels


def test_objects_mrc_is_exact_against_replay():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 30, 800)
    sizes = rng.integers(8, 64, 800)
    d = compute_reuse(keys, sizes, 30, cap=4095)
    caps = np.array([1, 2, 4, 8, 16, 64])
    got = mrc_from_labels(d, caps, by_bytes=False)
    want = np.array([naive_lru_mrc(keys, sizes, int(c)) for c in caps])
    assert np.array_equal(got, want)


def test_grid_respects_the_identification_limit():
    caps = identified_grid(n_keys=1_000_000, overflow_edge=1024, points=12)
    assert caps.max() < 1024
    assert caps.min() >= 1
    assert np.all(np.diff(caps) > 0)


def test_grid_is_unique_collapsed_not_padded():
    caps = identified_grid(n_keys=173_874, overflow_edge=837, points=24)
    assert caps.max() == 836
    assert len(caps) <= 24
    assert len(caps) == len(np.unique(caps))


def test_grid_never_exceeds_the_key_count():
    caps = identified_grid(n_keys=5, overflow_edge=1_000_000, points=12)
    assert caps.max() <= 5
    assert caps.min() >= 1
