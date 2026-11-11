# Phase C: features

**Phase:** C. **Owner profile:** `mrc-data`. **Defects closed:** D4 (only five coarse features).
**Upstream:** `01-phase-a-labels.md` (edge 2, `A4 -> C1`; and A3's `key_ns`/`key_len` columns).
**Downstream:** `04-phase-d-estimator.md` (edge 3, `C1 -> D1`), `07-phase-g-interpretability.md`
(G3 ablates these exact columns).

**Estimated wall time:** 3 tasks at 2 to 5 minutes each (master §4). The throughput task is the only one
with a runtime bound: 200,000 rows in under 2.0 s (master C3).
**Compute:** one process, no GPU. The real-trace path is vectorised over `key_ns` and `key_len`; the
string path is a Python loop over the key column and is the slow one.

---

## 1. Input and output contract

### Inputs

`labels: dict[str, np.ndarray]` with the Phase A columns. The columns this phase consumes:

| Column | Used for | Requirement |
|---|---|---|
| `key` | type discriminator, namespace coding on the string path | int32 codes, or an object array of strings in test fixtures |
| `key_ns` | `ns_code` on the real path | int32, `>= 1` for a non-empty key, `0` reserved for unknown |
| `key_len` | `log_key_size` and `log_val_size` on the real path | int32, `>= 1` |
| `obj_size` | `log_obj_size`, and `log_val_size = log1p(obj_size - key_len)` | int64 bytes |
| `ttl` | `log_ttl`, `has_ttl` | int64 seconds, `0` when absent |
| `client` | `client_bucket` | int32 |
| `op` | `op`, `is_write` | int8 code into `trace.OPS` |
| `gap` | `log_gap` | int64, `-1` for a first access |
| `freq` | `log_freq` | int64 `>= 1` |
| `R` | row count only (`n = R.shape[0]`) | int64 |
| `delta`, `rtilde` | **not read** | the target never enters the design matrix |

### Outputs

In memory, returned by `build_features`:

| Value | dtype | Shape | Invariant |
|---|---|---|---|
| `X` | float32 | `(n, 12)` | every entry finite; no column is a function of `R`, `delta` or `rtilde` |
| `names` | `list[str]` | 12 | exactly `list(FEATURE_NAMES)`, in order |
| `cat_idx` | `list[int]` | 3 | exactly the indices of `ns_code`, `op`, `is_write` |

Column definitions, fixed order. Reordering or renaming is a `FEATURE_VERSION` bump and a
`logs/DECISIONS.md` line.

| # | Column | Definition | Source |
|---|---|---|---|
| 0 | `log_obj_size` | `log1p(obj_size)` | `obj_size` |
| 1 | `log_key_size` | `log1p(key_len)` | `key_len` |
| 2 | `log_val_size` | `log1p(max(obj_size - key_len, 0))` | `obj_size`, `key_len` |
| 3 | `ns_code` | integer code of the first `:`-component of the key | `key_ns` |
| 4 | `op` | operation code | `op` |
| 5 | `is_write` | `op` is not `get` or `gets` (`_READ_OPS = (0, 1)`) | `op` |
| 6 | `log_ttl` | `log1p(max(ttl, 0))` | `ttl` |
| 7 | `has_ttl` | `ttl > 0` | `ttl` |
| 8 | `client_bucket` | `log1p(client % 512)` | `client` |
| 9 | `log_gap` | `log1p(max(gap, 0))` | `gap` |
| 10 | `log_freq` | `log1p(freq)` | `freq` |
| 11 | `log_age` | `log1p(request index)` | row position |

On-disk artefact `data/processed/<cluster>/features.npz`, written by F1, schema fixed here:

| Array | dtype | Shape | Invariant |
|---|---|---|---|
| `X` | float32 | `(n, 12)` | as above |
| `names` | `<U32` | `(12,)` | equals `FEATURE_NAMES` |
| `cat_idx` | int64 | `(3,)` | equals the categorical indices |

`data/processed/<cluster>/features.manifest.json` (required by the frozen cache-key contract; the
master's F1 sketch omits it, audit M-6):

```
sha256            str   64 hex of the npz file bytes
n_rows            int   == X.shape[0]
n_cols            int   == X.shape[1]
feature_version   int   == FEATURE_VERSION
names             list[str]
source_labels     str
labels_sha256     str
git_sha           str
created_utc       str
```

Invariants:

- `I-C1` `X` is float32 and finite; `build_features` raises `ValueError` rather than emitting NaN/inf.
- `I-C2` column order is `FEATURE_NAMES`; `names` is a copy, so a caller cannot mutate the module tuple.
- `I-C3` `ns_code` is a partition of the first `:`-component: equal components get equal codes, different
  components get different codes. The integer values themselves are not fixed.
- `I-C4` on real labels (`key` is int32), `ns_code` and `log_key_size` are **not constant**. If the
  labels carry no `key_ns`/`key_len`, `build_features` raises rather than silently emitting constants
  (audit M-17).
- `I-C5` the string path and the column path induce the same partition on the same keys.
- `I-C6` no column is a deterministic function of `R`, `delta` or `rtilde`.

---

## 2. Module interface contract

```python
# src/mrc/features.py
FEATURE_VERSION: int = 1
FEATURE_NAMES: tuple[str, ...]   # 12 names in the fixed order above
CATEGORICAL: tuple[str, ...]     # ("ns_code", "op", "is_write")

def _namespace_codes(keys: np.ndarray) -> np.ndarray:
    """Code the first ':'-delimited component of each key string; unknown -> 0."""

def build_features(labels: dict) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (X, names, categorical_indices); raises on non-finite values or missing key geometry."""
```

Callers: `src/mrc/pipeline.py::run_one` (F1), `src/mrc/cli.py` `tune` (F4), `src/mrc/pipeline.py::ablate`
(G3, which only needs `names`), and `tests/test_features.py`.

The producer of `key_ns`/`key_len` is `src/mrc/data/labels.py::_key_codes` from Phase A. The two coders
must agree on the partition (`I-C5`), which C2 tests across the module boundary.

Non-goals: no row filtering (compulsory requests stay in the matrix; the risk set drops them later), no
target construction, no LightGBM parameters, no `FEATURE_VERSION` bump, no reordering.

---

## 3. Ordered tasks

### C1: `build_features` with a fixed column order

**Purpose:** turn label columns into one float32 matrix that a GBDT can consume, with the target
excluded by construction.
**Files:** create `src/mrc/features.py`, `tests/test_features.py`.
**Depends on:** A4 (edge 2).

**TDD steps**

1. Failing test. The master's two tests, plus three that close the silent-constant hole (audit M-17):

```python
import numpy as np
import pytest

from mrc.data.reuse import compute_reuse
from mrc.features import FEATURE_NAMES, build_features


def _toy():
    keys = np.array([0, 1, 0, 2, 1, 3, 0, 1], dtype=np.int64)
    sizes = np.array([16, 20, 16, 24, 20, 12, 16, 20], dtype=np.int64)
    ttl = np.array([0, 0, 0, 0, 0, 1296000, 0, 0], dtype=np.int64)
    client = np.array([1, 2, 1, 3, 2, 1, 1, 2], dtype=np.int64)
    labels = compute_reuse(keys, sizes, 4, cap=1023)
    labels.update(obj_size=sizes, ttl=ttl, client=client,
                  key=np.array([f"q:q:1:{k}" for k in keys], dtype=object),
                  op=np.array([0, 0, 0, 1, 0, 2, 0, 0], dtype=np.int8),
                  ts=np.arange(8, dtype=np.int64))
    return labels


def test_shape_dtype_and_names():
    X, names, cat_idx = build_features(_toy())
    assert X.dtype == np.float32
    assert X.shape == (8, len(FEATURE_NAMES))
    assert names == list(FEATURE_NAMES)
    assert cat_idx == [names.index("ns_code"), names.index("op"), names.index("is_write")]
    assert not np.isnan(X).any()


def test_values_are_what_the_docstring_says():
    X, names, _ = build_features(_toy())
    col = {n: X[:, i] for i, n in enumerate(names)}
    assert col["log_obj_size"][0] == np.float32(np.log1p(16))
    assert col["is_write"][3] == 1.0 and col["is_write"][0] == 0.0
    assert col["log_ttl"][5] > 0.0 and col["log_ttl"][0] == 0.0
    assert col["log_gap"][0] == 0.0
    assert col["log_freq"][0] == np.float32(np.log1p(1))


def test_integer_key_labels_use_the_cached_key_geometry():
    lab = _toy()
    lab["key"] = np.array([0, 1, 0, 2, 1, 3, 0, 1], dtype=np.int32)   # real-trace layout
    lab["key_ns"] = np.array([5, 6, 5, 7, 6, 5, 5, 6], dtype=np.int32)
    lab["key_len"] = np.array([7, 7, 7, 7, 7, 7, 7, 7], dtype=np.int32)
    X, names, _ = build_features(lab)
    col = {n: X[:, i] for i, n in enumerate(names)}
    assert len(np.unique(col["ns_code"])) == 3          # not constant
    assert col["log_key_size"][0] == np.float32(np.log1p(7))


def test_integer_key_labels_without_geometry_raise():
    lab = _toy()
    lab["key"] = np.array([0, 1, 0, 2, 1, 3, 0, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        build_features(lab)


def test_no_column_is_a_function_of_the_target():
    lab = _toy()
    X, names, _ = build_features(lab)
    lab2 = dict(lab)
    lab2["R"] = lab["R"] * 3
    X2, _, _ = build_features(lab2)
    assert np.array_equal(X, X2)        # R is only ever read for its length
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/test_features.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.features'`.
3. Implement. The master's body stands, with two changes:

   - the integer-key branch reads `labels["key_ns"]` and `labels["key_len"]` and raises
     `ValueError("labels carry no key geometry; rebuild them with mrc labels")` when they are absent,
     instead of emitting `ns = zeros` and `key_size = 8.0` (audit M-17);
   - `is_write` is computed directly in the column list instead of being written as `... * 0.0` and then
     overwritten, so the list and the final matrix cannot drift.

   `log_val_size` uses `key_len` on both paths, so the byte accounting is exact rather than
   `obj_size - 8`.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/test_features.py -q` -> `5 passed`.

**Gate:** `G-C1-features` (`5 passed`: the master's two tests plus three that close the silent-constant
hole).
**Commit:** `feat(features): 12-column design matrix with cached key geometry`
**Rollback:** `git revert <sha>`; nothing imports it yet.
**Done when:** all five tests pass, the matrix is finite, and no column reads the target.

---

### C2: namespace coding on real keys, both paths agreeing

**Purpose:** the namespace feature must be real, not a constant, and the two code paths must not drift
apart.
**Files:** `tests/test_features.py` (append).
**Depends on:** C1.

**TDD steps**

1. Failing test:

```python
def test_real_twitter_keys_code_the_first_component():
    from mrc.features import _namespace_codes
    keys = np.array(["q:q:1:abc", "q:q:1:def", "s:1:zzz", "q:q:1:ghi"], dtype=object)
    codes = _namespace_codes(keys)
    assert codes[0] == codes[1] == codes[3]
    assert codes[2] != codes[0]
    assert codes.min() >= 1


def test_string_path_and_cached_column_path_agree_on_the_partition():
    from mrc.data.labels import _key_codes
    from mrc.features import _namespace_codes
    vocab = np.array(["q:q:1:a", "s:1:b", "q:q:1:c", "z:9:d"], dtype=object)
    keys = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    ns_col, _ = _key_codes(vocab, keys)
    ns_str = _namespace_codes(vocab[keys])
    # same partition: equal columns iff equal strings
    assert np.array_equal(ns_col == ns_col[0], ns_str == ns_str[0])
    assert len(np.unique(ns_col)) == len(np.unique(ns_str))
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/test_features.py -q` -> `1 failed`
   (`ImportError` or `AttributeError` on `_key_codes`).
3. Implementation: `_namespace_codes` stays as the master writes it. `_key_codes` in `labels.py` builds
   its code table from the vocabulary in vocabulary order; the test asserts a partition, not integer
   equality, because the two iterate in different orders.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/test_features.py -q` -> `7 passed`.

**Gate:** `G-C2-namespace` (`7 passed`).
**Commit:** `test(features): namespace partition agrees across the string and cached paths`
**Rollback:** `git revert <sha>`; test only.
**Done when:** both paths partition the keys identically and no code is `0` for a parseable key.

---

### C3: throughput gate on the slow path

**Purpose:** keep the design matrix out of the critical path of a 5 M-request run.
**Files:** `tests/test_features.py` (append).
**Depends on:** C2.

**TDD steps**

1. Failing evidence: the master's throughput test, verbatim (200,000 rows, object keys, budget 2.0 s).
   It exercises the **string** path, which is the slow one, so passing it implies the real path
   (`key_ns` lookup) is faster.
2. Run: `env -u PYTHONPATH uv run pytest tests/test_features.py -q` -> `8 passed`.
3. If it fails, vectorise the string path with `np.unique(keys, return_inverse=True)` over the first
   components rather than a per-row Python dict lookup. Do not raise the 2.0 s budget: the budget is
   the master's constant and a contract change.

**Gate:** `G-C3-throughput` (`8 passed`).
**Commit:** `test(features): 200k-row throughput gate`
**Rollback:** `git revert <sha>`; test only.
**Done when:** 200,000 rows build in under 2.0 s on this box.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-C1-features` | `env -u PYTHONPATH uv run pytest tests/test_features.py -q` | `5 passed` | a column is missing, misordered, non-finite, or reads the target |
| `G-C2-namespace` | `env -u PYTHONPATH uv run pytest tests/test_features.py -q` | `7 passed` | the two namespace coders disagree, so `ns_code` means different things depending on the path |
| `G-C3-throughput` | `env -u PYTHONPATH uv run pytest tests/test_features.py -q` | `8 passed` | the string path is too slow for 5 M rows; vectorise, do not relax the budget |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| Vacuous features on real data | the trace stores int key codes; a naive implementation emits constant `ns_code` and `key_size` and the ablation then measures nothing | A3 supplies `key_ns`/`key_len`; C1 raises when they are missing; two tests assert non-constant values |
| Target leakage through a feature | any column computed from `R`, `delta` or `rtilde` would make validation meaningless | `I-C6` and `test_no_column_is_a_function_of_the_target`, which perturbs `R` and asserts the matrix is unchanged |
| `log_age` collinear with the time-block split | request index rises monotonically, so the tree can partly identify the block; importance for it is inflated and validation gains may be optimistic | it stays because the master fixes the 12 columns, but G1 reports it and G3's `no_timing` group removes it, so the report can quantify the effect. Flagged, not silently removed |
| Two namespace coders drifting | `labels._key_codes` and `features._namespace_codes` both define "namespace" | C2 asserts partition equality across the boundary |
| Silent `FEATURE_VERSION` drift | a column reordered in place makes old `.npz` files unreadable in a way that looks like a model regression | the version is in the file name's manifest; F1 records it; reordering requires a bump and a DECISIONS line |
| Numpy string dtype surprise | `names` written as a Python list becomes an object array, which then needs `allow_pickle` | F1 writes `np.array(names)` of fixed-width `<U32`; the reader converts to `list[str]` |

---

## 6. Reviewer checklist

- [ ] `FEATURE_NAMES` has exactly 12 entries in the documented order and `FEATURE_VERSION == 1`.
- [ ] No column is derived from `R`, `delta`, `censor` or `rtilde`.
- [ ] `build_features` raises on integer-key labels without `key_ns`/`key_len`; no silent constants.
- [ ] `log_val_size` uses the real key length, so `obj_size - key_len >= 0` is clamped, not assumed.
- [ ] `log_gap` clips `-1` to `0` before `log1p`; `log_ttl` clips negatives likewise.
- [ ] `cat_idx` matches `CATEGORICAL` and the names list is not mutated by callers.
- [ ] The matrix has no NaN and no inf, and the check is an assertion, not a warning.
- [ ] The throughput test runs on 200,000 rows and fails loudly with the measured seconds.
- [ ] The diff touches only `src/mrc/features.py` and `tests/test_features.py`, plus the `key_ns`/
      `key_len` reads.

---

## 7. Handoff note

Before `04-phase-d-estimator.md` starts (edge 3):

1. `build_features` returns a float32 `(n, 12)` matrix for the `sample45` labels, and the real-trace path
   yields a non-constant `ns_code` and `log_key_size`.
2. `FEATURE_NAMES` and `CATEGORICAL` are frozen; G3's ablation dict is written against these names.
3. `features.npz` and `features.manifest.json` are F1's obligation, not C's; the schema in section 1 is
   what F1 must satisfy, including `feature_version`.
4. The `log_age` collinearity with the block split is stated in the handoff, so the Phase G worker does
   not present its importance as a finding about reuse.
