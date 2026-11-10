"""Deliberately slow reference implementations.

These exist only to pin the fast code paths bit-exactly (see tests). Never use
them on anything larger than a few thousand requests.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


def naive_reuse(keys, sizes):
    """O(n * window) reuse distance, distinct keys and distinct bytes."""
    n = len(keys)
    R = np.empty(n, dtype=np.int64)
    B = np.empty(n, dtype=np.int64)
    last: dict[int, int] = {}
    for i, k in enumerate(keys):
        p = last.get(k)
        if p is None:
            R[i], B[i] = -1, -1
        else:
            seen: dict[int, int] = {}
            for j in range(p + 1, i):
                seen[keys[j]] = int(sizes[j])
            R[i] = len(seen)
            B[i] = sum(seen.values())
        last[k] = i
    return R, B


def naive_lru_mrc(keys, sizes, capacity, by_bytes=False):
    """Replay LRU and return the miss ratio. Capacity in objects or bytes."""
    cache: OrderedDict[int, int] = OrderedDict()
    occupied = 0
    hits = 0
    for k, s in zip(keys, sizes):
        if k in cache:
            cache.move_to_end(k)
            hits += 1
            continue
        need = int(s) if by_bytes else 1
        cap = capacity
        while cache and (occupied + need > cap):
            _, ev_size = cache.popitem(last=False)
            occupied -= ev_size
        if need <= cap:
            cache[k] = need
            occupied += need
    return 1.0 - hits / len(keys)
