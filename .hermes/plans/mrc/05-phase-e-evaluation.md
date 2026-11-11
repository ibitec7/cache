# Phase E: evaluation and baselines

**Phase:** E. **Owner profile:** `mrc-run`. **Defects closed:** D5 in part (no eval, no metrics, no
baselines; the artefact half was Phase B).
**Upstream:** `04-phase-d-estimator.md` (the model these metrics score), `02-phase-b-truth.md` (the grid
and `b_{M-1}`).
**Downstream:** `06-phase-f-pipeline.md` (edge 8, `E1 -> F1`), `08-phase-h-report.md` (H3 writes these
numbers as tables).

**Estimated wall time:** 3 tasks at 2 to 5 minutes each (master §4). E3 fits three small LightGBM models
on 8,000 synthetic requests.
**Compute:** one process; E3's three models use `num_threads: 4` from `PARAMS`, everything else is numpy.

---

## 1. Input and output contract

### Inputs

| Input | Shape and dtype | Source |
|---|---|---|
| `pred`, `truth` | float64 `(K,)` | a model curve and the Mattson curve, on the same `capacities` |
| `capacities` | int64 `(K,)` | `truth.npz` (Phase B); every capacity `< b_{M-1}` |
| `rtilde` | int64 `(n,)` | labels (A3) |
| `delta` | bool `(n,)` | labels (A3) |
| `X` | float32 `(n, 12)` | features (C1) |
| `edges` | int64 `(M+1,)` | the fit-time edges, `b_{M-1} = edges[-2]` |
| `p_compulsory` | float | `(~eligible).mean()` |

### Outputs

`curve_metrics(pred, truth, capacities, mre_floor=0.01) -> dict`:

| Key | Type | Definition | Invariant |
|---|---|---|---|
| `mae` | float | `mean(|pred - truth|)` | `>= 0` |
| `max_abs` | float | `max(|pred - truth|)` | `>= mae` |
| `bias` | float | `mean(pred - truth)` | `|bias| <= max_abs` |
| `mre` | float | `mean(|err| / truth)` over points with `truth >= mre_floor` | `nan` when no point qualifies |
| `mre_n` | int | number of points kept for `mre` | `0 <= mre_n <= K` |
| `abc` | float | area between curves in `log C`, divided by `log(C_max) - log(C_min)` | `>= 0` |
| `n` | int | `K` | `== len(pred) == len(truth)` |

`ipcw_weights(rtilde, delta, capacities) -> (w, y)`, both `(n, K)`; `w` float64 `>= 0`, `y` bool.
`integrated_brier(pred, rtilde, delta, capacities) -> float`, the mean over capacities of the
IPCW-weighted Brier score.

Baselines return float64 `(K,)`, non-increasing in capacity, in `[p_compulsory, 1]`.

Metric semantics, one line each, as the report states them:

| Metric | What it measures | Why it is right here | What failing it means |
|---|---|---|---|
| MAE | average absolute miss-ratio displacement over the grid | the primary quantity is a curve, and MAE is the one scalar Optuna can minimise | the curve is displaced on average; with |bias| small, the error is local |
| MAX | worst pointwise displacement | a curve that is right on average can still be wrong where the operator runs | a local blow-up, usually at the knee |
| bias | signed mean displacement | separates random error from systematic optimism | systematically optimistic or pessimistic |
| MRE | relative error above a floor | large capacities have small miss ratios where absolute error is meaningless | relative error concentrated at small capacities |
| ABC | area between curves in log capacity | one number for the whole curve, comparable across grids of the same log range | a shape error that MAE dilutes |
| IBS | IPCW-weighted Brier score over capacities | the only metric that scores the hit/miss prediction itself under censoring | the hazard probabilities are miscalibrated even if the curve looks right |

---

## 2. Module interface contract

```python
# src/mrc/eval/metrics.py
def curve_metrics(pred, truth, capacities, mre_floor: float = 0.01) -> dict:
    """MAE, MAX, bias, MRE and log-normalised ABC on one capacity grid."""

def abc(pred, truth, capacities) -> float:
    """Area between curves, integrated in log C and normalised by the log range."""

def ipcw_weights(rtilde, delta, capacities) -> tuple[np.ndarray, np.ndarray]:
    """Graf weights and hit labels, shape (n_requests, n_capacities)."""

def integrated_brier(pred, rtilde, delta, capacities) -> float:
    """Mean over capacities of the IPCW-weighted Brier score."""
```

Callers: `curve_metrics` from `run_one` (F1), `cli.py` `fit` (existing, via `mrc_error`; leave that call
alone) and `report_tables.metrics_table` (H3). `integrated_brier` from `run_one` (F1).
`ipcw_weights` from `integrated_brier` and tests only.

```python
# src/mrc/eval/baselines.py
def km_curve(rtilde, delta, edges, capacities, p_compulsory: float = 0.0) -> np.ndarray:
    """Kaplan-Meier style hazard on the raw distances, no covariates."""

def single_row_ablation(X, rtilde, delta, edges, capacities, p_compulsory: float = 0.0,
                        *, eval_X=None, eval_rtilde=None, eval_delta=None) -> np.ndarray:
    """One row per request, label only at the event bin: biased but cheap."""

def hitcount_regression(X, rtilde, delta, edges, capacities, p_compulsory: float = 0.0,
                        *, eval_X=None, eval_rtilde=None, eval_delta=None) -> np.ndarray:
    """Regress the hit indicator on (x, capacity): the practice this project argues against."""
```

The three `eval_*` keywords are added by this plan (audit M-22). The master's baselines fit and evaluate
on the same arrays, so when F1 passes the test block they would train on the test block, which is the one
thing the protocol forbids. With the keywords, the caller passes the train block for fitting and the test
block for evaluation; when they are omitted the behaviour is the master's, which is what the unit tests
exercise.

Callers: `run_one` (F1). `km_curve` is also called by `cli.py fit` indirectly through `km_hazard`; that
path is unchanged.

Non-goals: no model fitting, no Optuna, no figures, no tables under `report/`.

---

## 3. Ordered tasks

### E1: the metric set with hand-checked values

**Purpose:** one place that defines every error number, with a hand example that fixes the arithmetic.
**Files:** create `src/mrc/eval/metrics.py`, `tests/eval/test_metrics.py`.
**Depends on:** D5 (the curves it scores).

**TDD steps**

1. Failing test, as master E1 gives it (three tests). The arithmetic is verified and the constants are
   correct (audit M-19):
   - `mae`: errors `[0, 0.1, 0, -0.1]`, mean of absolutes `0.05`.
   - `mre`: kept points are `C=1` (error 0) and `C=10` (error `0.1/0.5 = 0.2`), mean `0.1`, `mre_n = 2`;
     `C=1000` is dropped because `truth = 0.004 < 0.01`.
   - `abc`: gaps `[0, 0.2, 0]` over `log C = [0, 2.3026, 4.6052]` give area `0.46052`, divided by
     `4.6052` gives `0.1`.
   The master's "note for the implementer" says the implementation is right if it disagrees with the
   test. That note is wrong here. **Do not edit these constants.** If an implementation disagrees, the
   implementation is wrong.
2. Add one positive-control test (planner addition, so a metric that silently returns zeros cannot pass):

```python
def test_perfect_prediction_gives_zero_error_and_bias():
    caps = np.array([1, 10, 100])
    truth = np.array([0.9, 0.5, 0.1])
    m = curve_metrics(truth, truth, caps)
    assert m["mae"] == 0.0 and m["max_abs"] == 0.0 and m["bias"] == 0.0 and m["abc"] == 0.0
    assert m["n"] == 3
```

3. Verify failure: `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.eval.metrics'`.
4. Implement as master E1 gives it, with `np.trapezoid` preferred and `np.trapz` as the fallback (numpy 2
   removed neither, but the guard is already in the sketch; keep it).
5. Verify pass: `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` -> `4 passed`.

**Gate:** `G-E1-metrics` (`4 passed`).
**Commit:** `feat(eval): MAE, MAX, bias, MRE and log-normalised ABC`
**Rollback:** `git revert <sha>`; nothing imports it yet.
**Done when:** the four tests pass and every metric is a plain Python float or int, ready for MLflow.

---

### E2: Graf IPCW weights and the integrated Brier score

**Purpose:** score the hit/miss prediction itself, with the censoring correction the observation scheme
requires.
**Files:** modify `src/mrc/eval/metrics.py`; append to `tests/eval/test_metrics.py`.
**Depends on:** E1.

**TDD steps**

1. Failing test, as master E2 gives it (two tests).
2. Verify failure: `ImportError: cannot import name 'ipcw_weights'`.
3. Implement `_km_censoring`, `ipcw_weights`, `integrated_brier` as master E2 gives them. The censoring
   survival is `S_Gamma(t^-)`, hence the `np.concatenate(([1.0], S[:-1]))` shift; keep it, it is what
   makes a request censored at step 0 contribute weight 1.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` -> `6 passed`.

**Gate:** `G-E2-ipcw` (`6 passed`).
**Commit:** `feat(eval): Graf IPCW weights and integrated Brier score`
**Rollback:** `git revert <sha>`; `curve_metrics` still returns `abc` and the rest.
**Done when:** a censored request below capacity contributes zero weight, and a perfect prediction gives
IBS 0.

---

### E3: three baselines behind one interface

**Purpose:** give the hazard model something to beat, at three levels of sophistication, and quantify what
the risk set buys.
**Files:** create `src/mrc/eval/baselines.py`, `tests/eval/test_baselines.py`.
**Depends on:** E2.

**TDD steps**

1. Failing test, as master E3 gives it, with one corrected assertion (audit M-21). The master writes
   `assert np.all(np.diff(c) >= -1e-12)` for `km_curve`; the curve is non-increasing in capacity, so the
   sign must be `<=`:

```python
def test_km_curve_is_monotone_and_in_range():
    rt, d, X, edges, caps = _toy()
    c = km_curve(rt, d, edges, caps, p_compulsory=0.1)
    assert c.shape == caps.shape
    assert np.all(np.diff(c) <= 1e-12)                  # non-increasing in capacity (audit M-21)
    assert np.all((c >= 0.1 - 1e-9) & (c <= 1.0))
```

   The other two tests (shape only) stand.
2. Add one fairness test (planner addition, audit M-22):

```python
def test_baselines_do_not_train_on_the_evaluation_block():
    rt, d, X, edges, caps = _toy()
    n = rt.shape[0]
    tr = np.zeros(n, dtype=bool); tr[: int(n * 0.7)] = True
    te = ~tr
    c_fair = single_row_ablation(X[tr], rt[tr], d[tr], edges, caps, p_compulsory=0.1,
                                 eval_X=X[te], eval_rtilde=rt[te], eval_delta=d[te])
    c_leaky = single_row_ablation(X, rt, d, edges, caps, p_compulsory=0.1)
    assert c_fair.shape == caps.shape
    assert not np.array_equal(c_fair, c_leaky)    # the evaluation set changes the answer
```

3. Verify failure: `ModuleNotFoundError: No module named 'mrc.eval.baselines'`.
4. Implement as master E3 gives it, with `eval_X` / `eval_rtilde` / `eval_delta` wiring: fitting uses
   `X, rtilde, delta`; prediction uses `eval_X or X`, and `eval_rtilde` / `eval_delta` replace `rtilde` /
   `delta` when the curve is averaged over a different block.
5. Verify pass: `env -u PYTHONPATH uv run pytest tests/eval/test_baselines.py -q` -> `4 passed`.

**Gate:** `G-E3-baselines` (`4 passed`).
**Commit:** `feat(eval): KM, single-row and hit-count baselines with evaluation-block control`
**Rollback:** `git revert <sha>`; F1 then reports no baselines and its gate fails loudly.
**Done when:** all three return curves on the grid, KM is monotone and bounded, and no baseline fits on
the arrays it is scored on.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-E1-metrics` | `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` | `4 passed` | the arithmetic differs from the hand example; the test is right |
| `G-E2-ipcw` | `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` | `6 passed` | the censoring survival is shifted the wrong way, so censored requests get event weight |
| `G-E3-baselines` | `env -u PYTHONPATH uv run pytest tests/eval/test_baselines.py -q` | `4 passed` | monotonicity is inverted, or a baseline ignores the evaluation block |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| MRE with a vanishing denominator | dividing by a miss ratio of 0.004 produces a number that measures nothing | `mre_floor = 0.01` and `mre_n` reported beside it, so a reader sees how many points the number rests on |
| ABC depending on the grid | the area is normalised by the log range, so a shorter grid gives a different number | every ABC is reported with its `K` and the grid's log range; comparisons are only made on the same grid, which Phase B freezes |
| Baselines trained on the block they are scored on | master's signatures fit and evaluate on the same arrays | `eval_*` keywords and E3's fairness test; F1 passes train for fitting and test for scoring |
| KM baseline computed with a different censoring convention than the hazard | KM uses `delta` per bin; the hazard uses the risk set | both read the same `delta` and the same `edges`; D2 already shows grouped and per-step agree |
| IBS computed on the wrong scale | IBS is a Brier score over hit indicators, not over miss ratios | `pred` is the hit probability per request per capacity, and the test asserts 0 for a perfect predictor; the note in `run_one` must pass hit probabilities, not the MRC curve (see Phase F) |
| Metric with a silently nan value reaching MLflow | `mre` is `nan` when no point clears the floor | F1 logs only finite values and records `mre_n`; the reviewer checks the JSON for `NaN` |

---

## 6. Reviewer checklist

- [ ] All six metrics are plain `float` or `int`, no numpy scalars leaking into JSON.
- [ ] `mre` uses `|err| / truth`, not `|err| / pred`, and only on `truth >= mre_floor`.
- [ ] `abc` integrates in `log C` and divides by the log range; a perfect prediction gives exactly 0.
- [ ] `ipcw_weights` returns `(w, y)` in that order, `w >= 0`, `y` bool.
- [ ] `_km_censoring` evaluates `S(t^-)`, so a request censored at its first step gets weight 1.
- [ ] No baseline fits on `eval_X`; the only fitting arrays are the first three positional arguments.
- [ ] `km_curve` is non-increasing in capacity (audit M-21 sign).
- [ ] The three `PARAMS` in `baselines.py` carry `deterministic` and `force_row_wise`, like the hazard model.
- [ ] No metric is computed against a truth that is itself a prediction (no proxy metric).

---

## 7. Handoff note

Before `06-phase-f-pipeline.md` starts (edge 8):

1. `curve_metrics` and `integrated_brier` exist with the schemas above.
2. `km_curve`, `single_row_ablation`, `hitcount_regression` return `(K,)` curves and accept the `eval_*`
   keywords.
3. The metric set the report will present is fixed: MAE primary, MAX, bias, MRE (`mre_floor = 0.01`),
   ABC, IBS. No metric is added later without a `logs/DECISIONS.md` line.
4. F1 must compute the truth curve **restricted to the block it reports on** (`mrc_curve(R[mask], caps)`),
   because `truth.npz` covers the whole prefix; a whole-prefix truth scored against a test-block curve
   mixes two denominators (audit M-23, resolved in the Phase F plan).
