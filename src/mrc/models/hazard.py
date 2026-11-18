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
    """Bin index per request; compulsory accesses get -1.

    A distance at or above the last edge lands in the last real bin, index
    ``len(edges) - 2``, never in ``len(edges) - 1``. That bin is the overflow
    bin: it collects every capped distance, so survival below it is the
    identification limit.
    """
    d = np.asarray(distance)
    b = np.searchsorted(edges, d, side="right") - 1
    b = np.clip(b, -1, len(edges) - 2)
    b[d < 0] = CB
    return b


def block_masks(n: int, train: float | None = None,
                valid: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous request-index masks for train / valid / test.

    The split is on request index, not on expanded rows: neighbouring requests in
    a trace share keys, so any interleaved split puts the same key on both sides.
    """
    from mrc.config import SPLIT_TRAIN, SPLIT_VALID

    train = SPLIT_TRAIN if train is None else train
    valid = SPLIT_VALID if valid is None else valid
    i_tr = int(n * train)
    i_va = int(n * valid)
    idx = np.arange(n)
    return idx < i_tr, (idx >= i_tr) & (idx < i_va), idx >= i_va


def _split_rows(rows: dict, n_requests: int, train: float | None = None,
                valid: float | None = None):
    """Row masks for train / valid / test, split on ORIGINAL request index.

    ``rows["request_index"]`` indexes the kept eligible rows and
    ``rows["eligible_index"]`` maps them back to trace positions, so the frozen
    70/15/15 request-index rule is applied once, in ``block_masks``.
    """
    orig = rows["eligible_index"][rows["request_index"]]
    if orig.size and int(orig.max()) >= int(n_requests):
        raise ValueError(
            f"request index {int(orig.max())} outside n_requests={int(n_requests)}"
        )
    tr, va, te = block_masks(int(n_requests), train, valid)
    return tr[orig], va[orig], te[orig]


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
    validation_logloss: float | None = None
    split_n: int | None = None

    def fit(self, X, bins, event, edges, feature_names, num_boost_round=400, valid_frac=0.2):
        self.edges, self.feature_names = edges, feature_names
        Xr, step, y, _req = expand_risk_set(X, bins, event, self.nbins)
        Xr = np.column_stack([Xr, step]).astype(np.float32)
        n = Xr.shape[0]
        cut = int(n * (1 - valid_frac))
        dtrain = lgb.Dataset(Xr[:cut], label=y[:cut], feature_name=feature_names + ["hazard_step"])
        dvalid = lgb.Dataset(Xr[cut:], label=y[cut:], reference=dtrain)
        self.booster = lgb.train(self.params, dtrain, num_boost_round=num_boost_round,
                                 valid_sets=[dvalid], callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def fit_grouped(self, rows: dict, edges: np.ndarray, feature_names: list[str],
                    num_boost_round: int = 400, n_requests: int | None = None,
                    train: float | None = None, valid: float | None = None) -> HazardModel:
        """Fit on grouped risk-set rows with contiguous time-block splits.

        ``rows`` is the output of ``risk_set.grouped_rows``. The split is applied
        to the original request index, never to the row order, so neighbouring
        requests cannot put the same key on both sides.
        """
        self.edges = edges
        self.feature_names = list(feature_names)
        n_req = (int(rows["eligible_index"].max()) + 1
                 if n_requests is None else int(n_requests))
        self.split_n = n_req
        tr, va, _te = _split_rows(rows, n_req, train, valid)

        names = self.feature_names + ["hazard_step"]
        # One allocation for the design matrix. `column_stack` followed by
        # `.astype(float32)` made two copies of a 20M x 13 block, about 2 GB.
        X = np.empty((rows["X"].shape[0], len(names)), dtype=np.float32)
        X[:, :-1] = rows["X"]
        X[:, -1] = rows["bin"]
        y = np.asarray(rows["y"], dtype=np.float32)
        w = np.asarray(rows["w"], dtype=np.float32)

        dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], feature_name=names)
        dvalid = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtrain)
        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        score = self.booster.best_score.get("valid_0", {})
        self.validation_logloss = float(score.get("binary_logloss", np.nan))
        return self

    def predict_survival(self, X, steps, max_grid_rows: int = 2_000_000):
        """S_R(k | x) for every request at every step k in `steps` (sorted).

        Prediction is chunked so peak memory is `max_grid_rows` float32 rows rather
        than n_requests * len(steps): at 650k eligible requests and 33 steps the
        unchunked grid is 1.1 GB, which is what a 20M-request run can no longer hold.
        """
        X = np.asarray(X, dtype=np.float32)
        n, K = X.shape[0], len(steps)
        if self.booster is None:
            raise RuntimeError("predict_survival needs a fitted model")
        step = np.asarray(steps, dtype=np.float32)
        d = X.shape[1]
        out = np.empty((n, K), dtype=np.float64)
        per = max(1, int(max_grid_rows) // max(K, 1))
        for start in range(0, n, per):
            block = X[start:start + per]
            m = block.shape[0]
            # One float32 allocation. `column_stack` of a float32 block and an int64
            # step column promotes the whole block to float64 before the
            # `.astype(float32)` copies it back, which is a 96 MB round trip per
            # 2M-row chunk for values that were float32 to begin with. The widened
            # float64 values are exact, so the float32 grid is the same bytes.
            grid = np.empty((m * K, d + 1), dtype=np.float32)
            grid[:, :d].reshape(m, K, d)[:] = block[:, None, :]
            grid[:, d].reshape(m, K)[:] = step
            h = np.asarray(self.booster.predict(grid), dtype=np.float64).reshape(m, K)
            # In place: the clip, the complement and the prefix scan are each
            # elementwise (or a running product along the axis), so writing into
            # the same buffer gives the same values and drops three temporaries.
            np.clip(h, 1e-9, 1 - 1e-9, out=h)
            np.subtract(1.0, h, out=h)
            np.cumprod(h, axis=1, out=h)
            out[start:start + m] = h
        return out

    def predict_mrc(self, X, capacities, edges, p_compulsory=0.0):
        """MRC at each capacity.

        MRC(C) = p_comp + (1 - p_comp) * E[S_R(bin(C - 1) | x)]

        The first term is the mass of first-time accesses, which no capacity can
        turn into hits; the model is fitted only on requests that have a previous
        access, so that mass has to be added back explicitly.
        """
        steps, bins_c = self._steps_and_bins(capacities, edges)
        S = self.predict_survival(X, steps)
        return p_compulsory + (1.0 - p_compulsory) * S[:, bins_c].mean(axis=0)

    def predict_mrc_rows(self, X, capacities, edges, p_compulsory=0.0):
        """The same curve per request, shape `(n, len(capacities))`.

        `predict_mrc` averages these values first and that order is kept: averaging
        then scaling is not the same floating-point sequence as scaling then
        averaging, and the committed curves are the averaged ones.
        """
        steps, bins_c = self._steps_and_bins(capacities, edges)
        S = self.predict_survival(X, steps)
        return p_compulsory + (1.0 - p_compulsory) * S[:, bins_c]

    def _steps_and_bins(self, capacities, edges) -> tuple[np.ndarray, np.ndarray]:
        """Hazard steps to predict and the step each capacity reads, with the checks."""
        caps = np.asarray(capacities, dtype=np.int64)
        limit = int(edges[-2])
        if caps.size and int(caps.max()) >= limit:
            raise ValueError(
                f"capacity {int(caps.max())} is at or above the identification limit "
                f"{limit}: the overflow bin makes every capacity from {limit} upward "
                "unidentified. Truncate the grid below the limit."
            )
        kmax = int(min(len(edges) - 2, assign_bins(np.array([int(caps.max())]), edges)[0])) + 1
        steps = np.arange(kmax + 1)
        bins_c = np.clip(assign_bins(caps - 1, edges), 0, kmax)
        return steps, bins_c


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
