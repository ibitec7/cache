# Phase G: interpretability

**Phase:** G. **Owner profile:** `mrc-run`. **Defects closed:** none numbered; this is the "explainable
results" half the master's goal names.
**Upstream:** `06-phase-f-pipeline.md` (edge 7, `F1 -> G1`), `03-phase-c-features.md` (the 12 columns and
their names).
**Downstream:** `08-phase-h-report.md` (edge 9, `G3 -> H3`; the ablation table and the importance figure).

**Estimated wall time:** 3 tasks at 2 to 5 minutes each (master §4). G1's permutation loop is
12 columns x 5 repeats = 60 scoring passes over the validation block; G2 runs one `pred_contrib` call;
G3 is column deletion only.
**Compute:** one process. G2's `pred_contrib` on a validation block of a few hundred thousand rows is the
only heavy call, and it is bounded by the block, not by the whole prefix.

---

## 1. Input and output contract

### Inputs

| Input | Shape and dtype | Source |
|---|---|---|
| `X` | float32 `(n, 12)` | features (C1) |
| `names` | `list[str]` length 12 | `FEATURE_NAMES` |
| validation mask | bool `(n,)` | `block_masks(n_requests)[1]` |
| fitted booster | `lgb.Booster` | `HazardModel.booster` from F1 |
| validation `rtilde` | int64 | labels (A3) |

The test block is not read by any task in this phase.

### Outputs

| Artefact | Path | Producer | Invariant |
|---|---|---|---|
| permutation importance | returned as a list, logged by F1 or written as `artifacts/tables/<cluster>_importance.json` | G1 | one row per feature; sorted by mean drop descending; every row has `feature`, `drop`, `std` |
| SHAP summary | `artifacts/tables/<cluster>_shap.json` | G2 | `{"top": [{"feature", "mean_abs_contribution"}], "n_rows": int}`, `len(top) == top` |
| ablation matrix | in memory `dict[str, np.ndarray]` | G3 | six groups; column counts equal `12 - len(dropped)` |

`importance.json` schema (written by whoever calls G1, which is F1's driver or a small CLI step):

```
cluster        str
block          "valid"
n_rows         int
n_repeats      int
features       [{"feature": str, "drop": float, "std": float}, ...]   sorted by drop desc
base_score     float
git_sha        str
created_utc    str
```

Invariants:

- `I-G1` importance is computed on the validation block only; the test mask is never passed in.
- `I-G2` `drop` is `base_score - score(permuted)`, so a positive value means the column helped.
- `I-G3` SHAP contributions drop the final bias column (`contrib[:, :-1]`), so the feature axis matches
  `names`.
- `I-G4` the ablation groups are exactly the six frozen in `logs/DECISIONS.md`: `all`, `no_namespace`,
  `no_sizes`, `no_timing`, `no_popularity`, `no_client`.
- `I-G5` no group silently drops a column that is already absent; a missing name is a `KeyError` at build
  time, not a smaller matrix.

---

## 2. Module interface contract

```python
# src/mrc/interp/importance.py
def permutation_importance(score, X: np.ndarray, names: list[str],
                           n_repeats: int = 5, seed: int = 0) -> list[dict]:
    """Return rows {feature, drop, std} sorted by mean drop, descending."""

def shap_summary(booster, X: np.ndarray, names: list[str], top: int = 10, path=None):
    """Mean |contribution| per feature from LightGBM's pred_contrib, plus a json dump."""
```

Callers: `permutation_importance` from the G1 driver (a function in `pipeline.py` or the CLI, decided in
F4's handoff) and `tests/interp/test_importance.py`. `shap_summary` from the same driver and the tests.

```python
# src/mrc/pipeline.py  (added by G3)
GROUPS: dict[str, tuple[str, ...]]

def ablate(X: np.ndarray, names: list[str], groups: dict = GROUPS) -> dict[str, np.ndarray]:
    """Return {group: X with the dropped columns removed}."""
```

Callers: the ablation driver and `tests/test_pipeline.py`.

Non-goals: no retraining with a different feature set, no test-block access, no per-run tuning, no
change to `FEATURE_NAMES`.

---

## 3. Ordered tasks

### G1: permutation importance on the validation block

**Purpose:** say which columns the fitted hazard actually uses, with a method whose failure mode
(permutation breaks the association but keeps the marginal) is stated in the report.
**Files:** create `src/mrc/interp/__init__.py`, `src/mrc/interp/importance.py`; create
`tests/interp/test_importance.py`.
**Depends on:** F1 (edge 7).

**TDD steps**

1. Failing test, as master G1 gives it (one test, using a scoring rule instead of a model so the test does
   not depend on LightGBM).
2. Verify failure: `ModuleNotFoundError: No module named 'mrc.interp'`.
3. Implement as master G1 gives it. `base = score(X)` is computed once; each repeat copies `X` and
   permutes one column, so the marginal distribution is preserved.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/interp/test_importance.py -q` -> `1 passed`.
5. Real-data check, recorded in the card comment: run the driver on the `sample45` validation block with
   `score(X) = -mae` of the resulting MRC curve against the validation truth, and report the ranking. The
   expectation, stated before running, is that `log_gap` and `log_freq` lead and `log_age` ranks high for
   the collinearity reason in Phase C. A ranking that puts `log_age` first is a finding about the split,
   not about reuse, and must be reported as such.

**Gate:** `G-G1-perm` (`1 passed`).
**Commit:** `feat(interp): permutation importance on the validation block`
**Rollback:** `git revert <sha>`; nothing imports it.
**Done when:** the informative column ranks first on the synthetic test and the real ranking is recorded
with its block and row count.

---

### G2: SHAP contributions and the hazard-by-feature table

**Purpose:** a second attribution that is additive within a tree and therefore agrees or disagrees with
permutation in an interpretable way.
**Files:** modify `src/mrc/interp/importance.py`; append to `tests/interp/test_importance.py`.
**Depends on:** G1.

**TDD steps**

1. Failing test, as master G2 gives it (trains a 30-round booster on 3,000 rows, asserts the top feature
   and the length).
2. Verify failure: `ImportError: cannot import name 'shap_summary'`.
3. Implement as master G2 gives it. Keep the bias-column drop; assert in the docstring that
   `contrib.shape[1] == len(names) + 1`.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/interp -q` -> `2 passed`.
5. Real run: write `artifacts/tables/sample45_shap.json` with `top = len(FEATURE_NAMES)` so the report can
   show the full ranking, and record `n_rows`.

**Gate:** `G-G2-shap` (`2 passed`).
**Commit:** `feat(interp): SHAP mean-absolute contributions with json dump`
**Rollback:** `git revert <sha>`; the permutation table stands alone.
**Done when:** the json has one row per feature and the ranking is comparable to G1's.

---

### G3: feature-group ablation matrix

**Purpose:** show each feature family earns its place, per skeleton §8.3, with the list frozen before any
tuning run (`logs/DECISIONS.md`).
**Files:** modify `src/mrc/pipeline.py`; append to `tests/test_pipeline.py`.
**Depends on:** G2.

**TDD steps**

1. Failing test:

```python
def test_ablation_groups_have_the_frozen_column_counts():
    from mrc.features import FEATURE_NAMES
    from mrc.pipeline import GROUPS, ablate
    X = np.zeros((5, len(FEATURE_NAMES)), dtype=np.float32)
    out = ablate(X, list(FEATURE_NAMES))
    assert set(out) == {"all", "no_namespace", "no_sizes", "no_timing", "no_popularity", "no_client"}
    assert out["all"].shape == X.shape
    assert out["no_sizes"].shape[1] == X.shape[1] - 3
    assert out["no_namespace"].shape[1] == X.shape[1] - 1
    for g, drop in GROUPS.items():
        assert out[g].shape[1] == len(FEATURE_NAMES) - len(drop)
    with pytest.raises(KeyError):
        ablate(X, list(FEATURE_NAMES), {"bogus": ("not_a_column",)})
```

2. Verify failure: `ImportError: cannot import name 'GROUPS'`.
3. Implement `GROUPS` and `ablate` as master G3 gives them, with the `KeyError` behaviour from `I-G5`
   (the master's `idx[n]` lookup already raises `KeyError`, so no extra code is needed; the test pins it).
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` -> `4 passed`.
5. Ablation driver, in the same module, run once per cluster with the frozen `_params` and the same split:
   for each group, fit on the train block, score validation MAE, write
   `artifacts/tables/<cluster>_ablation.json` with `{group: {"mae": float, "n_cols": int}}`. The driver is
   the only place that refits 6 times; it is bounded by the same row guard as F1.

**Gate:** `G-G3-ablate` (`4 passed`).
**Commit:** `feat(pipeline): feature-group ablation matrix`
**Rollback:** `git revert <sha>`; the six-group list is recorded in DECISIONS and can be rebuilt from it.
**Done when:** the group list matches the frozen one exactly and the driver writes one json per cluster.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-G1-perm` | `env -u PYTHONPATH uv run pytest tests/interp/test_importance.py -q` | `1 passed` | the shuffle does not break the association, or the sort order is ascending |
| `G-G2-shap` | `env -u PYTHONPATH uv run pytest tests/interp -q` | `2 passed` | the bias column is not dropped, so names and columns are misaligned by one |
| `G-G3-ablate` | `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` | `4 passed` | a group drops the wrong column, or a missing name silently shrinks the matrix |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| Importance measured on the test block | the cheap thing to do is score everything once | `I-G1`; the driver takes the validation mask as an argument and the reviewer checks the call site |
| `log_age` dominating the ranking | request index is collinear with the block split, so the tree can identify the block | reported, not hidden: G3's `no_timing` group removes it and the report states the collinearity (Phase C risk row) |
| Permutation on a correlated pair understating both | permuting one of two correlated columns leaves the model able to recover the signal from the other | reported as a limitation; SHAP is the second view, and the ablation is the third |
| SHAP memory on a large block | `pred_contrib` allocates `n x (F+1)` float64 | the driver subsamples the validation block to a fixed row count recorded in the json, or runs in chunks; the count is written, never assumed |
| Ablation refits changing the split | a driver that rebuilds masks differently would confound the ablation with the split | the driver takes the masks from `block_masks(n_requests)`, the same call F1 uses |
| Ablation list retrofitted after tuning | the list would be chosen to flatter the model | frozen in `logs/DECISIONS.md` before any Optuna run; the reviewer checks the dates |

---

## 6. Reviewer checklist

- [ ] No task in this phase reads the test mask.
- [ ] `permutation_importance` returns `base - permuted`, not the reverse.
- [ ] `shap_summary` drops the last column and asserts the shape.
- [ ] `GROUPS` matches the six frozen groups name for name.
- [ ] `ablate` raises on an unknown column instead of skipping it.
- [ ] The ablation driver refits with the frozen `_params`, not with a re-tuned parameter set.
- [ ] Every json written carries `git_sha`, `created_utc`, `cluster`, `n_rows`.
- [ ] No importance number is quoted in the report without its block name and row count.

---

## 7. Handoff note

Before `08-phase-h-report.md` starts (edge 9):

1. `artifacts/tables/<cluster>_importance.json`, `<cluster>_shap.json` and `<cluster>_ablation.json` exist
   for at least `sample45`, each with the schemas above.
2. The ranking is recorded together with the collinearity caveat on `log_age`, so H3's table caption can
   state it.
3. The six ablation groups are unchanged from `logs/DECISIONS.md`.
4. H1's MRC figure and H2's reliability diagram do not depend on this phase, so Phase H may start H1
   before G3 finishes; only H3 and H4 need the ablation table.
