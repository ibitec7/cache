"""Exact Mattson stack-distance MRC from reuse distances.

Under LRU a request hits at object capacity C iff R_i < C, where R_i is the
number of distinct keys requested since the previous access to the same key.
The miss ratio is therefore the empirical survival of R:

    MRC(C) = P(R >= C)

This is exact for object capacities, so it is the ground truth the hazard model
is scored against, and it never reads `rtilde`: the truth is a function of the
trace, not of the model's truncation. Compulsory accesses (R_i = -1) miss at
every capacity; `mrc_curve` sends them to a sentinel so they cannot be counted
as hits.

The byte variant is approximate: adding the requested object's size to B_i
ignores that an object larger than the cache cannot be admitted, so it is never
written into the primary artefact.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mrc.config import N_CAPACITIES
from mrc.eval.mrc import capacity_grid, mrc_curve


def mrc_from_labels(
    labels: Mapping[str, np.ndarray],
    capacities: Sequence[int],
    by_bytes: bool = False,
) -> np.ndarray:
    """Miss ratio at each capacity from a label dict produced by `compute_reuse`.

    Exact for object capacities; the byte variant is approximate (module
    docstring). `labels` may be the dict `compute_reuse` returns or the npz it
    was saved into.
    """
    if by_bytes:
        return mrc_curve(labels["B"], capacities, own_size=labels["obj_size"])
    return mrc_curve(labels["R"], capacities)


def identified_grid(
    n_keys: int,
    overflow_edge: int,
    points: int = N_CAPACITIES,
) -> np.ndarray:
    """Log-spaced object capacities strictly below the overflow bin edge.

    `overflow_edge` is b_{M-1} = `bin_edges(...)[-2]`, the first distance that
    falls in the overflow bin, so no returned capacity is the extrapolation the
    hazard model refuses. Capping the grid at `n_keys` keeps it inside the trace:
    above that every non-compulsory request hits and the curve is flat. Rounding
    collapses duplicates, so the realised count can be smaller than `points`;
    readers take the size from the returned array.
    """
    top = min(int(n_keys), int(overflow_edge) - 1)
    if top < 1:
        raise ValueError(f"no identified capacity below overflow_edge={overflow_edge}")
    return capacity_grid(top, points)
