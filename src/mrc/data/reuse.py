"""Reuse distance in O(n log n) with a Fenwick tree over last-occurrence positions.

Definition
----------
For request i to key k_i with previous access at position p(i), the reuse
distance R_i is the number of DISTINCT keys requested strictly between p(i)
and i. R_i = -1 marks a compulsory (first) access.

Why a Fenwick tree
------------------
Mark 1 at the most recent position of every key seen so far. Then for any
window (a, b] the number of distinct keys inside it is exactly
prefix(b) - prefix(a): a key appearing in the window has its mark inside it,
a key not appearing has its mark at or before a. Range-sum therefore answers
"distinct count since last access" in O(log n) per request.

A second tree carries object sizes, which yields the byte-weighted variant
(the quantity a real cache compares against capacity in bytes):
B_i = total size of the distinct objects requested since the last access.
"""
from __future__ import annotations

import numpy as np
from numba import njit

CB = -1  # compulsory / first access marker


@njit(cache=True, nogil=True)
def _fenwick_add(tree, n, pos, delta):
    while pos <= n:
        tree[pos] += delta
        pos += pos & (-pos)


@njit(cache=True, nogil=True)
def _fenwick_sum(tree, pos):
    s = 0
    while pos > 0:
        s += tree[pos]
        pos -= pos & (-pos)
    return s


@njit(cache=True, nogil=True)
def reuse_kernel(keys, sizes, n_keys):
    """Return (R, B, gap, freq) for a key-code sequence.

    R    distinct-key reuse distance, CB (-1) for first access
    B    distinct-bytes reuse distance, CB (-1) for first access
    gap  requests since the previous access, -1 for first access
    freq 1-based occurrence index of the key
    """
    n = keys.shape[0]
    R = np.empty(n, dtype=np.int64)
    B = np.empty(n, dtype=np.int64)
    gap = np.empty(n, dtype=np.int64)
    freq = np.empty(n, dtype=np.int64)

    cbit = np.zeros(n + 2, dtype=np.int64)
    sbit = np.zeros(n + 2, dtype=np.int64)
    last = np.full(n_keys, -1, dtype=np.int64)
    last_size = np.zeros(n_keys, dtype=np.int64)
    count = np.zeros(n_keys, dtype=np.int64)

    for i in range(n):
        k = keys[i]
        sz = sizes[i]
        p = last[k]
        count[k] += 1
        freq[i] = count[k]
        if p < 0:
            R[i] = CB
            B[i] = CB
            gap[i] = -1
        else:
            pos_p = p + 1
            pos_i = i + 1
            R[i] = _fenwick_sum(cbit, pos_i - 1) - _fenwick_sum(cbit, pos_p)
            B[i] = _fenwick_sum(sbit, pos_i - 1) - _fenwick_sum(sbit, pos_p)
            gap[i] = i - p
            _fenwick_add(cbit, n, pos_p, -1)
            _fenwick_add(sbit, n, pos_p, -last_size[k])
        _fenwick_add(cbit, n, i + 1, 1)
        _fenwick_add(sbit, n, i + 1, sz)
        last[k] = i
        last_size[k] = sz
    return R, B, gap, freq


def compute_reuse(keys: np.ndarray, sizes: np.ndarray, n_keys: int) -> dict:
    """Compute reuse distance columns plus censoring flags.

    A request is right-censored when its key is never requested again inside the
    observed window, i.e. it is the last occurrence of that key in the trace.
    """
    keys = np.ascontiguousarray(keys.astype(np.int64))
    sizes = np.ascontiguousarray(sizes.astype(np.int64))
    R, B, gap, freq = reuse_kernel(keys, sizes, int(n_keys))

    last_occ = np.full(int(n_keys), -1, dtype=np.int64)
    for i in range(len(keys)):
        last_occ[keys[i]] = i
    is_last = np.zeros(len(keys), dtype=bool)
    is_last[last_occ[last_occ >= 0]] = True

    return {
        "R": R,
        "B": B,
        "gap": gap,
        "freq": freq,
        "censor": is_last & (freq > 1),
        "compulsory": R == CB,
    }
