"""Exact integer histograms over reuse distance.

Why a histogram and not an array
--------------------------------
An out-of-time split needs quantities from every window of the train period, and
the train period of the full trace is 8B requests: 64 GB as int64, impossible to
hold in one process. Every quantity the split needs from ``R`` is a function of
its order statistics or of a count below a threshold, and both survive an add.
So each window reduces to a histogram as it is read, and windows merge with ``+``.

The two identities that make this exact rather than approximate:

* ``mrc_curve`` is ``1 - #(R < C) / n``, and ``#(R < C)`` is a prefix sum of the
  histogram at index ``C``. Compulsory accesses (``R = -1``) always miss, so they
  live at index 0 and are counted below every capacity.
* ``np.percentile(r, q)`` is linear interpolation between two order statistics.
  A histogram recovers any order statistic exactly, so the interpolation is
  reproduced bit for bit, which matters because the bin grid is derived from it
  and a one-unit change in the top edge moves every later curve.

Layout: index 0 holds ``R = -1`` (compulsory), index ``v + 1`` holds ``R = v`` for
``0 <= v < limit``, and the last index holds every ``R >= limit``. The overflow
bucket cannot be interpolated, so the percentile raises when the target rank
lands in it instead of returning a value that is silently too small.
"""
from __future__ import annotations

import numpy as np

HIST_LIMIT = 1 << 23           # 8,388,608 distinct distances, 64 MB of int64


class DistanceHistogram:
    """Additive histogram of reuse distances, one per window or per block."""

    __slots__ = ("counts", "limit")

    def __init__(self, limit: int = HIST_LIMIT) -> None:
        self.limit = int(limit)
        self.counts = np.zeros(self.limit + 2, dtype=np.int64)

    def add(self, distance) -> DistanceHistogram:
        """Accumulate one window's ``R`` array in place."""
        r = np.asarray(distance, dtype=np.int64)
        idx = r + 1
        np.clip(idx, 0, self.limit + 1, out=idx)
        self.counts += np.bincount(idx, minlength=self.limit + 2)
        return self

    def merge(self, other: DistanceHistogram) -> DistanceHistogram:
        if other.limit != self.limit:
            raise ValueError(f"histogram limits differ: {self.limit} vs {other.limit}")
        self.counts += other.counts
        return self

    @property
    def n(self) -> int:
        """Requests represented, compulsory included."""
        return int(self.counts.sum())

    @property
    def n_eligible(self) -> int:
        """Requests with ``R >= 0``."""
        return int(self.counts[1:].sum())

    def cumulative(self) -> np.ndarray:
        """``cum[i]`` is the count of requests with ``R < i``, compulsory included."""
        return np.cumsum(self.counts)

    def below(self, capacities) -> np.ndarray:
        """Count of requests with ``R < C`` for every capacity in the grid.

        Compulsory accesses are excluded: ``mrc_curve`` parks them at ``int64`` max
        so they are below no capacity, which is what makes them always a miss.
        """
        cum = self.cumulative()
        caps = np.asarray(capacities, dtype=np.int64)
        if caps.size and int(caps.min()) < 1:
            raise ValueError(f"capacities must be >= 1, got {int(caps.min())}")
        pos = np.minimum(caps, self.limit + 1)
        return cum[pos] - self.counts[0]

    def mrc(self, capacities) -> np.ndarray:
        """Empirical miss ratio at each capacity, identical to ``mrc_curve``."""
        n = self.n
        if n == 0:
            raise ValueError("empty histogram has no miss-ratio curve")
        # mrc_curve computes `1.0 - idx / n` with integer idx; same expression here.
        return 1.0 - self.below(capacities) / n

    def percentile(self, q: float) -> float:
        """``np.percentile(R[eligible], q)`` without holding ``R``."""
        cum = np.cumsum(self.counts[1:])          # eligible only, index v holds R = v
        n = int(cum[-1])
        if n == 0:
            raise ValueError("no eligible requests, no distance percentile")
        rank = (q / 100.0) * (n - 1)
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        frac = rank - lo
        # order statistic k is the first index whose cumulative count exceeds k
        v_lo = int(np.searchsorted(cum, lo + 1))
        v_hi = int(np.searchsorted(cum, hi + 1))
        if v_hi >= self.limit:
            raise ValueError(
                f"percentile {q} lands in the overflow bucket of a histogram with "
                f"limit {self.limit}: raise the limit to resolve it")
        return float(v_lo) + frac * float(v_hi - v_lo)

    def bin_counts(self, edges: np.ndarray) -> np.ndarray:
        """Eligible requests per hazard bin, the same bins ``assign_bins`` makes.

        Bin 0 is ``R = 0``, bin b is ``edges[b] <= R < edges[b+1]``, and the last
        bin collects every capped distance, so a Kaplan-Meier hazard computed from
        these counts is the one ``pipeline.km_curve`` computes from the raw array.
        """
        e = np.asarray(edges, dtype=np.int64)
        nbins = len(e) - 1
        # edges[0] is 0, below every capacity; below() refuses a capacity below 1
        # and below(1) is exactly count(R = 0), which is bin 0.
        below = self.below(np.maximum(e, 1))
        counts = np.empty(nbins, dtype=np.int64)
        counts[0] = below[1]
        if nbins > 2:
            counts[1:-1] = np.diff(below[1:-1])
        counts[-1] = self.n_eligible - below[-2]
        return counts
