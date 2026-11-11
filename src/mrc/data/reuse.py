"""Reuse distance in O(n log n) with a Fenwick tree over last-occurrence positions.

Definition
----------
For request i to key k_i with previous access at position p(i), the reuse
distance R_i is the number of DISTINCT keys requested strictly between p(i)
and i. R_i = -1 marks a compulsory (first) access.

The model also needs the censoring bound. A request whose key is never
requested again inside the observed window has its next reference beyond the
window, so all we know is R_i >= gamma_i, where gamma_i is the number of
distinct keys requested strictly after i. Feeding the backward distance of a
censored request into a risk set would bias survival down.

Columns returned
----------------
R       backward reuse distance in distinct keys, -1 for first access
B       backward reuse distance in bytes, -1 for first access
gap     requests since the previous access, -1 for first access
freq    1-based occurrence index of the key
gamma   distinct keys requested strictly after i (the bound Gamma_i)
eligible  freq > 1: the request has a previous access, so it may enter a risk set
delta   eligible and the next access is observed inside the window
rtilde  min(R, cap) for an event, min(gamma, cap) for a censored request, -1 if
        not eligible. cap must equal b_{M-1} - 1, the last edge below overflow.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from mrc.config import CAP_QUANTILE

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


@njit(cache=True, nogil=True)
def suffix_distinct(keys, n_keys):
    """distinct[i] = number of distinct key codes in keys[i:]; distinct[n] = 0."""
    n = keys.shape[0]
    seen = np.zeros(n_keys, dtype=np.bool_)
    out = np.empty(n + 1, dtype=np.int64)
    out[n] = 0
    cnt = 0
    for i in range(n - 1, -1, -1):
        k = keys[i]
        if not seen[k]:
            seen[k] = True
            cnt += 1
        out[i] = cnt
    return out


def compute_reuse(
    keys: np.ndarray,
    sizes: np.ndarray,
    n_keys: int,
    cap: int = CAP_QUANTILE,
) -> dict:
    """Reuse distance, censoring bound and the truncation the model consumes.

    See the module docstring for the meaning of every returned column. `cap`
    must equal b_{M-1} - 1, the last bin edge below the overflow bin.
    """
    keys = np.ascontiguousarray(keys.astype(np.int64))
    sizes = np.ascontiguousarray(sizes.astype(np.int64))
    n_keys = int(n_keys)
    R, B, gap, freq = reuse_kernel(keys, sizes, n_keys)

    last_occ = np.full(n_keys, -1, dtype=np.int64)
    for i in range(len(keys)):
        last_occ[keys[i]] = i
    is_last = np.zeros(len(keys), dtype=bool)
    is_last[last_occ[last_occ >= 0]] = True

    gamma = suffix_distinct(keys, n_keys)[1:]        # strictly after i
    eligible = freq > 1
    delta = eligible & (~is_last)
    horizon = np.where(delta, R, gamma)
    rtilde = np.where(eligible, np.minimum(horizon, int(cap)), -1).astype(np.int64)

    return {
        "R": R,
        "B": B,
        "gap": gap,
        "freq": freq,
        "gamma": gamma.astype(np.int64),
        "eligible": eligible,
        "delta": delta,
        "rtilde": rtilde,
        "compulsory": R == CB,
        "censor": eligible & is_last,
    }
