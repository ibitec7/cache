"""Discrete-time hazard model over reuse distance.

Construction
------------
Reuse distance is binned into log-spaced intervals. Request i with bin b_i
contributes one row per hazard step 0..b_i (the risk set). The label is 1 at
step b_i when the re-access is observed inside the trace, and 0 on every row
when the request is right-censored. Fitting a binary classifier on those rows
estimates h_b(x) = P(R = b | R >= b, X = x), and

    S_R(k | x) = prod_{b=0..k} (1 - h_b(x))

is the survival function, from which

    MRC(C) = mean_i S_R(bin(C - 1) | x_i)

because LRU hits exactly when the reuse distance is below capacity.

Censoring enters through the risk set: a censored request still contributes
rows for steps it survived, it just never contributes a positive label. That is
the discrete-time analogue of the Kaplan-Meier correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

CB = -1


def bin_edges(max_distance: int, nbins: int = 32) -> np.ndarray:
    """Log-spaced bin edges, starting at 1 (distance 0 is its own bin)."""
    edges = np.unique(np.round(np.geomspace(2, max(3, max_distance), nbins)).astype(np.int64))
    return np.concatenate(([0, 1], edges))


def assign_bins(distance: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per request; compulsory accesses get -1."""
    d = np.asarray(distance)
    b = np.searchsorted(edges, d, side="right") - 1
    b[d < 0] = CB
    return b


def expand_risk_set(X: np.ndarray, bins: np.ndarray, event: np.ndarray, nbins: int):
    """Expand one row per request into one row per hazard step.

    Returns (X_rows, step_rows, y_rows, request_index_rows).
    """
    keep = bins >= 0
    X, bins, event = X[keep], bins[keep], event[keep]
    orig_idx = np.flatnonzero(keep)

    counts = bins + 1
    total = int(counts.sum())
    rep_req = np.repeat(np.arange(X.shape[0]), counts)
    step = np.empty(total, dtype=np.int32)
    pos = 0
    for c in counts:                       # cumulative fill, O(n) python loop
        step[pos:pos + c] = np.arange(c, dtype=np.int32)
        pos += c
    y = (step == np.repeat(bins, counts)) & np.repeat(event, counts)
    return X[rep_req], step, y.astype(np.float32), orig_idx[rep_req]


@dataclass
class HazardModel:
    params: dict
    nbins: int = 32
    booster: lgb.Booster | None = None
    edges: np.ndarray | None = None
    feature_names: list[str] | None = None

    def fit(self, X, bins, event, edges, feature_names, num_boost_round=400, valid_frac=0.2):
        self.edges, self.feature_names = edges, feature_names
        Xr, step, y, req = expand_risk_set(X, bins, event, self.nbins)
        Xr = np.column_stack([Xr, step]).astype(np.float32)
        n = Xr.shape[0]
        cut = int(n * (1 - valid_frac))
        dtrain = lgb.Dataset(Xr[:cut], label=y[:cut], feature_name=feature_names + ["hazard_step"])
        dvalid = lgb.Dataset(Xr[cut:], label=y[cut:], reference=dtrain)
        self.booster = lgb.train(self.params, dtrain, num_boost_round=num_boost_round,
                                 valid_sets=[dvalid], callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def predict_survival(self, X, steps):
        """S_R(k | x) for every request at every step k in `steps` (sorted)."""
        X = np.asarray(X, dtype=np.float32)
        n, K = X.shape[0], len(steps)
        grid = np.repeat(X, K, axis=0)
        grid = np.column_stack([grid, np.tile(np.asarray(steps, dtype=np.int64), n)]).astype(np.float32)
        h = self.booster.predict(grid).reshape(n, K).astype(np.float64)
        return np.cumprod(1.0 - np.clip(h, 1e-9, 1 - 1e-9), axis=1)

    def predict_mrc(self, X, capacities, edges, p_compulsory=0.0):
        """MRC at each capacity.

        MRC(C) = p_comp + (1 - p_comp) * E[S_R(bin(C - 1) | x)]

        The first term is the mass of first-time accesses, which no capacity can
        turn into hits; the model is fitted only on requests that have a previous
        access, so that mass has to be added back explicitly.
        """
        kmax = int(min(len(edges) - 2, assign_bins(np.array([int(max(capacities))]), edges)[0])) + 1
        steps = np.arange(kmax + 1)
        S = self.predict_survival(X, steps)
        bins_c = assign_bins(np.asarray(capacities, dtype=np.int64) - 1, edges)
        bins_c = np.clip(bins_c, 0, kmax)
        return p_compulsory + (1.0 - p_compulsory) * S[:, bins_c].mean(axis=0)


def km_hazard(distance: np.ndarray, nbins: int = 32):
    """Nonparametric (Kaplan-Meier style) baseline: h_k = d_k / n_k on raw distance.

    Returns (h, edges, p_compulsory). The compulsory mass must be added back when
    converting survival to a miss ratio: those requests can never hit at any
    finite capacity, so MRC(C) = p_comp + (1 - p_comp) * S(bin(C-1)).
    """
    d = np.asarray(distance)
    p_comp = float(np.mean(d < 0))
    d = d[d >= 0]
    max_d = int(np.percentile(d, 99.9)) if d.size else 1
    edges = bin_edges(max_d, nbins)
    bins = assign_bins(np.asarray(distance), edges)
    nb = len(edges)
    at_risk = np.zeros(nb, dtype=np.int64)
    events = np.zeros(nb, dtype=np.int64)
    valid = bins >= 0
    events += np.bincount(bins[valid], minlength=nb)
    # at risk at step k = number of requests whose bin >= k
    hist = np.bincount(bins[valid], minlength=nb)
    at_risk = np.cumsum(hist[::-1])[::-1]
    h = np.where(at_risk > 0, events / np.maximum(at_risk, 1), 0.0)
    return h, edges, p_comp
