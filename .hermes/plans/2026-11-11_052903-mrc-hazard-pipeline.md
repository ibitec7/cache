# MRC Hazard Pipeline — Implementation Plan

> **For Hermes:** execute with `subagent-driven-development` — one fresh subagent per task, spec-compliance review then code-quality review. Do not batch tasks.

**Goal:** turn the scaffolded repo at `~/uni/cache` into a working pipeline that estimates a cache miss-ratio curve `MRC(C)` from Twitter production traces by fitting a LightGBM discrete-time hazard model over reuse distance, validated against exact Mattson stack-distance ground truth, with Optuna tuning, MLflow telemetry, interpretability artefacts, and a filled 8–10 page report.

**Architecture:** five cached stages — `labels` (reuse distance + censoring bound, numba) → `truth` (exact MRC from stack distance) → `features` (float32 design matrix) → `risk set + LightGBM hazard` (bin-level grouped rows, time-block split) → `eval/plots/tables` — each writing a hashed artefact under `data/processed/` or `artifacts/`, so a rerun is a cache hit rather than a recompute.

**Tech stack:** Python 3.13 via uv, numpy + numba, pyarrow, LightGBM 4.6, Optuna 4, MLflow 2.18 on sqlite, matplotlib under the frozen style contract, LaTeX booktabs.

**Machine:** 12 cores, 30 GB RAM (≈18 GB available), 659 GB free. No GPU. Every command assumes cwd `/home/ibrahim/uni/cache`. `env -u PYTHONPATH` is mandatory: Hermes injects a Python 3.14 site-packages that breaks pydantic and mlflow.

---

## 1. Current context (verified this session — do not re-derive)

### 1.1 What exists and works

| Item | State | Evidence |
|---|---|---|
| Env | uv-managed, `requires-python >=3.13`, deps installed | `env -u PYTHONPATH uv run pytest -q` → `.......... [100%]`, 10 passed |
| Git | 12 commits, clean tree | `git log --oneline -1` → `3d27271 log: audit of setup completeness` |
| `src/mrc/data/trace.py` | `read_trace(path, limit=None) -> Trace`; fields `ts, key, obj_size, client, op, ttl, key_vocab`; `n_keys`; `slice()` | 122 lines |
| `src/mrc/data/reuse.py` | `compute_reuse(keys, sizes, n_keys) -> dict` with `R, B, gap, freq, censor, compulsory`; numba Fenwick | 114 lines, 2.58 M req/s |
| `src/mrc/data/reference.py` | `naive_reuse`, `naive_lru_mrc(keys, sizes, capacity, by_bytes=False)` | 51 lines |
| `src/mrc/eval/mrc.py` | `mrc_curve(distance, capacities, own_size=None)`, `capacity_grid(max_capacity, points=24)`, `mrc_error(pred, truth) -> {mae,max_abs,bias,mre}` | 58 lines |
| `src/mrc/models/hazard.py` | `bin_edges(max_distance, nbins=32)`, `assign_bins(distance, edges)`, `expand_risk_set`, `HazardModel.fit/predict_survival/predict_mrc`, `km_hazard` | 136 lines |
| `src/mrc/models/search.py` | `suggest_params(trial)`, `make_objective(fit_fn)`, `study(storage, name)` | 41 lines |
| `src/mrc/plots/style.py` | `apply_style()`, `fig_single(height=2.1)`, `fig_double(height=2.4)`, `label_mrc(ax)` | 64 lines, frozen |
| `src/mrc/cli.py` | click group `main`: `inspect`, `reuse`, `mrc`, `fit`, `plots` | 114 lines |
| Data | cluster45 1.87 GB, cluster26 1.38 GB, cluster10 1.08 GB, cluster52 93.09 GB, all sha1-verified against `data/raw/traces/UPSTREAM.sha1` | `data/raw/traces/` |
| Docs | `PROJECT_PLAN.md` 295, `STATISTICAL_SKELETON.md` 687 (71 symbols), `MILESTONES.md`, `RISKS.md`, `DATA.md`, `LITERATURE.md`, `PLOT_CONTRACT.md`, `AGENTS.md` | root + `docs/` |

### 1.2 Measured facts the plan depends on

On `data/interim/s45.npz` (200,000 requests, `cluster045` sample), `cap=1023`, 32 log-spaced bins:

| Quantity | Value |
|---|---|
| Reuse-eligible (`freq > 1`) | 13.06 % |
| Observed events (next access inside window) | 8.80 % |
| Censored but eligible | 4.26 % |
| `R` percentiles 50 / 90 / 99 / 99.9 | 2,812 / 34,379 / 90,134 / 131,873 |
| Per-step risk-set rows | 21,174,922 → **105.9 rows per trace request** |
| Bin-level grouped rows | 745,784 → **3.73 rows per trace request** |
| Ratio | **28.4×** |

**Frozen consequence 1:** the canonical estimator is the **bin-level grouped hazard** (one row per (request, bin at risk)). Per-step expansion survives only as an ablation on ≤200,000 requests, because at 5 M requests it would be 529 M rows.

**Frozen consequence 2:** 86.9 % of requests are cold. They stay out of the risk set and their mass is added back as `p_compulsory`, exactly as `HazardModel.predict_mrc(..., p_compulsory=)` already does.

### 1.3 Six defects that block the science

| # | Defect | Location | Why it matters |
|---|---|---|---|
| D1 | No censoring bound `Γ_i`; only a boolean `censor` | `src/mrc/data/reuse.py:96-114` | Skeleton §5.2 requires `Γ_i` = distinct keys after `i`; without it a censored request cannot enter a risk set |
| D2 | Censored requests are expanded to their **backward** distance; `expand_risk_set` ignores `censor` | `src/mrc/models/hazard.py:47-66`, `cli.py:93` | A censored request is known only to satisfy `R > Γ_i`; its backward distance is the wrong quantity and biases survival down |
| D3 | Split is a position cut on **expanded rows** (`cut = int(n * 0.8)`), not contiguous request blocks | `src/mrc/models/hazard.py:78` | Adjacent requests share keys; a row-order cut leaks keys across the split and inflates every metric (skeleton §6.5) |
| D4 | Only 5 coarse features | `cli.py:96-101` | M2 requires namespace prefix, key/value size, client bucket, log gap, log freq, recency |
| D5 | No ground-truth artefact, no eval, no tables, no figures; `mrc plots` prints "no figures yet" | `cli.py:110-113` | M3/M4 cannot start |
| D6 | `predict_mrc` silently clips capacity bins | `src/mrc/models/hazard.py:104-107` | The identification limit `max G < b_{M-1}` must raise, not clip |

Every task below names the defect it closes.

---

## 2. Frozen contracts

Constants live in one file. Nothing else may hardcode them; a change is a commit on this file plus a `logs/WORKLOG.md` line.

```python
# src/mrc/config.py
"""Frozen run constants. One source of truth; nothing else may hardcode these."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw" / "traces"
PROCESSED = DATA / "processed"
ARTIFACTS = REPO / "artifacts"
MLFLOW_URI = f"sqlite:///{ARTIFACTS / 'mlflow' / 'mlflow.db'}"
OPTUNA_URI = f"sqlite:///{ARTIFACTS / 'optuna' / 'study.db'}"

LABEL_REQUESTS = 5_000_000      # prefix per cluster for labels / features / truth
SCALE_REQUESTS = 20_000_000     # cluster52 robustness run only
CAP_QUANTILE = 1023             # rtilde cap == b_{M-1} - 1 <= 1023
N_BINS = 32
N_BINS_SEARCH = (8, 16, 24, 32, 48, 64)
N_CAPACITIES = 24

SPLIT_TRAIN = 0.70              # contiguous request blocks
SPLIT_VALID = 0.85              # valid [0.70, 0.85), test [0.85, 1.0)

MAX_GROUPED_ROWS = 60_000_000
PER_STEP_MAX_REQUESTS = 200_000

SEED = 7
NUM_THREADS = 12
```

| Contract | Value | Rationale |
|---|---|---|
| Window | first 5,000,000 requests per cluster | 87 % cold rate; 5 M yields ~650 k eligible rows, enough for a 12-feature GBDT |
| Ground truth | Mattson stack distance from `R`, exact for object capacities | skeleton §7.1; `libCacheSim` aggregate path is broken (`docs/DATA.md` §9), single-request replay retained as a cross-check |
| Capacity grid | `capacity_grid(n_keys, 24)`, truncated so `max G < b_{M-1}` | one grid for all clusters, curves comparable across traces |
| Bins | `bin_edges(percentile(rtilde, 99.9), 32)` | log-spaced to match the log-spaced grid; overflow bin is the identification limit |
| Split | contiguous blocks 70 / 15 / 15 by request index | no key leakage; test block touched once |
| Primary metric | validation-block `MAE` of MRC | one scalar for Optuna; `MAX`, `bias`, `MRE`, `ABC`, `IBS` reported beside it |
| Determinism | `SEED=7`, `NUM_THREADS=12`, `deterministic=True`, `force_row_wise=True` | bit-reproducible on this box |
| Cache key | sha256 in a `manifest.json` beside each artefact + the config values that produced it | rerun is a hit; config drift is a miss, never silent staleness |

---

## 3. Proposed approach

Five stages, each a pure function of its inputs plus a manifest entry, orchestrated by `src/mrc/pipeline.py`. Every stage runs standalone from the CLI, so a bad tail costs one stage, not an evening. Optuna searches only the hazard stage and reads cached artefacts, so 60 trials cost minutes.

```
trace shard (93 GB .zst)
  └─ labels    R, B, Γ, δ, R̃, freq, gap          data/processed/<c>/labels.npz
       ├─ truth     MRC(C) = P(R ≥ C)             artifacts/truth/<c>.npz        ← ground truth
       └─ features  X (float32, 12 cols)          data/processed/<c>/features.npz
            └─ statistical model: h_b(x) = P(R∈b | R≥b, x)
                 └─ LightGBM on grouped rows (bin index as a feature)
                      ├─ S_R(b|x) = ∏(1 − h)  →  MRC(C) = p_c + (1−p_c)·E_x[S_R]   ← the estimate
                      ├─ Optuna 60 trials on MAE(valid)     artifacts/optuna/study.db
                      ├─ MLflow params/metrics/curves/model artifacts/mlflow/mlflow.db
                      └─ interpretability: permutation, SHAP, feature-group ablation
                            └─ figures + booktabs tables → report 8–10 pp, slides, wiki
```

Module map after this plan:

| Path | Role |
|---|---|
| `src/mrc/config.py` | frozen constants (new) |
| `src/mrc/data/labels.py` | label cache + manifest (new) |
| `src/mrc/features.py` | design matrix builder (new) |
| `src/mrc/gt/replay.py` | `libCacheSim` prefix cross-check (new) |
| `src/mrc/gt/stackdist.py` | exact MRC from labels (new) |
| `src/mrc/models/risk_set.py` | grouped + per-step expansion (new) |
| `src/mrc/models/hazard.py` | `HazardModel` with block splits, isotonic survival (modified) |
| `src/mrc/eval/metrics.py` | MAE/MAX/bias/MRE/ABC/IBS with Graf weights (new) |
| `src/mrc/eval/baselines.py` | KM, single-row, hit-count regression (new) |
| `src/mrc/pipeline.py` | `run_one(cluster, cfg)` orchestration (new) |
| `src/mrc/tuning/search.py` | Optuna study on cached artefacts (replaces `models/search.py`) |
| `src/mrc/interp/importance.py` | permutation, SHAP, ablations (new) |
| `src/mrc/plots/curve.py`, `reliability.py` | figures (new) |
| `src/mrc/report_tables.py` | markdown + booktabs writers (new) |

---

## 4. Step-by-step tasks

Twenty-nine tasks in eight phases, each 2–5 minutes. Commit after every task with the message given verbatim.

Global gate for any code task:

```bash
cd /home/ibrahim/uni/cache
env -u PYTHONPATH uv run pytest -q              # expected: all pass
env -u PYTHONPATH uv run ruff check src tests   # expected: no output
```

---

### Phase A — labels with an honest censoring bound (closes D1)

#### Task A1: add `Γ`, `δ`, `R̃` to `compute_reuse`

**Files:** modify `src/mrc/data/reuse.py`; create `tests/data/test_labels_bound.py`.

**Step 1 — failing test**

```python
# tests/data/test_labels_bound.py
import numpy as np
from mrc.data.reuse import compute_reuse


def test_gamma_counts_distinct_keys_after_each_request():
    k = np.array([0, 1, 0, 2, 1, 3])
    s = np.ones(6, dtype=np.int64)
    d = compute_reuse(k, s, 4, cap=10)
    assert d["gamma"].tolist() == [4, 4, 3, 2, 1, 0]


def test_delta_true_only_when_next_reference_is_inside_the_window():
    k = np.array([0, 1, 0, 1])
    d = compute_reuse(k, np.ones(4, dtype=np.int64), 2, cap=10)
    assert d["eligible"].tolist() == [False, False, True, True]
    assert d["delta"].tolist() == [False, False, True, False]


def test_rtilde_uses_forward_bound_when_censored():
    k = np.array([0, 1, 0, 1])
    d = compute_reuse(k, np.ones(4, dtype=np.int64), 2, cap=10)
    assert d["rtilde"][2] == 1   # event, R = 1 distinct key in between
    assert d["rtilde"][3] == 0   # censored, nothing follows, gamma = 0
    assert d["rtilde"][0] == -1  # cold, not eligible, marked not guessed


def test_rtilde_is_capped_at_the_overflow_edge():
    k2 = np.repeat(np.arange(50), 2)
    d2 = compute_reuse(k2, np.ones(100, dtype=np.int64), 50, cap=4)
    assert d2["rtilde"].max() == 4
```

**Step 2 — verify failure**

```bash
env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q
```

Expected: 4 failed, `KeyError: 'gamma'` / `TypeError: compute_reuse() got an unexpected keyword argument 'cap'`.

**Step 3 — implement**

Append to `src/mrc/data/reuse.py`, above `compute_reuse`:

```python
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
```

Replace `compute_reuse` with:

```python
def compute_reuse(keys: np.ndarray, sizes: np.ndarray, n_keys: int, cap: int = 1023) -> dict:
    """Reuse distance, censoring bound and the truncation the model consumes.

    R       backward reuse distance in distinct keys, -1 for a first access
    B       backward reuse distance in bytes, -1 for a first access
    gap     requests since the previous access, -1 for a first access
    freq    1-based occurrence index of the key
    gamma   distinct keys requested strictly after i, i.e. the bound Gamma_i
    eligible True when a previous access exists, so the request may enter a risk set
    delta   True when the NEXT access is observed inside the window
    rtilde  min(R, cap) for an event, min(gamma, cap) for a censored request, -1 if
            not eligible. cap must equal b_{M-1} - 1, the last edge below overflow.
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
    rtilde = np.where(eligible, np.minimum(horizon, cap), -1).astype(np.int64)

    return {
        "R": R, "B": B, "gap": gap, "freq": freq,
        "gamma": gamma.astype(np.int64),
        "eligible": eligible,
        "delta": delta,
        "rtilde": rtilde,
        "compulsory": R == CB,
        "censor": eligible & is_last,   # kept for callers that only know this flag
    }
```

**Step 4 — verify pass**

```bash
env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q
env -u PYTHONPATH uv run pytest -q
```

Expected: `4 passed`, then `14 passed`.

**Step 5 — commit**

```bash
git add src/mrc/data/reuse.py tests/data/test_labels_bound.py
git commit -m "feat(labels): add censoring bound, delta and rtilde to compute_reuse"
```

#### Task A2: pin the semantics on the skeleton's eight-request trace

**Files:** test only, `tests/data/test_labels_bound.py` (append).

Trace `A B A C A B D C`. Occurrences: A at 0,2,4; B at 1,5; C at 3,7; D at 6. Eligible requests = 5 (A@2, A@4, B@5, C@7 — and the second occurrence of nothing else). Distances: A@2 sees `{B}` → 1; A@4 sees `{C,B}` → 2; B@5 sees `{A,C,D}` → 3; C@7 is the last occurrence → censored.

```python
def test_eight_request_micro_example():
    key = {"A": 0, "B": 1, "C": 2, "D": 3}
    k = np.array([key[c] for c in "ABACABDC"], dtype=np.int64)
    d = compute_reuse(k, np.ones(8, dtype=np.int64), 4, cap=7)
    assert int(d["eligible"].sum()) == 5
    assert np.sort(d["R"][d["delta"]]).tolist() == [1, 2, 3]
    assert bool(d["delta"][7]) is False       # C@7 is censored
```

```bash
env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q
```

Expected: `5 passed`. **If `[1, 2, 3]` fails, the invariant is wrong, not the test — stop and re-derive by hand from skeleton §2.3 before editing anything.**

```bash
git add tests/data/test_labels_bound.py
git commit -m "test(labels): pin rtilde against the skeleton micro-example"
```

#### Task A3: the label cache plus manifest

**Files:** create `src/mrc/config.py` (verbatim from §2), create `src/mrc/data/labels.py`, create `tests/data/test_label_cache.py`.

**Step 1 — failing test**

```python
# tests/data/test_label_cache.py
import json
import numpy as np
from mrc.data.labels import build_labels, load_labels, save_labels


def _toy_csv(tmp_path):
    p = tmp_path / "toy.csv"
    p.write_text("0,aaa,10,0,1,get,0\n1,bbb,10,0,1,get,0\n2,aaa,10,0,2,get,0\n"
                 "3,ccc,10,0,1,get,0\n4,aaa,10,0,1,get,0\n")
    return p


def test_build_save_load_roundtrip(tmp_path):
    src = _toy_csv(tmp_path)
    lab = build_labels(src, limit=None, cap=7)
    assert lab["R"].shape == (5,)
    assert lab["rtilde"].dtype == np.int64
    out = tmp_path / "labels.npz"
    man = save_labels(out, lab, source=src, limit=None, cap=7)
    meta = json.loads(man.read_text())
    assert meta["n_requests"] == 5 and meta["cap"] == 7
    assert len(meta["sha256"]) == 64
    assert np.array_equal(load_labels(out)["rtilde"], lab["rtilde"])
```

**Step 2 — verify failure:** `env -u PYTHONPATH uv run pytest tests/data/test_label_cache.py -q` → `ModuleNotFoundError: No module named 'mrc.data.labels'`.

**Step 3 — implement `src/mrc/data/labels.py`**

```python
"""Label cache: reuse distance, censoring bound and the columns the model needs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mrc.config import CAP_QUANTILE
from mrc.data.reuse import compute_reuse
from mrc.data.trace import read_trace

LABEL_KEYS = ("R", "B", "gap", "freq", "gamma", "rtilde", "eligible", "delta", "compulsory")
REQUEST_KEYS = ("ts", "key", "obj_size", "op", "client", "ttl")


def build_labels(path, limit=None, cap: int = CAP_QUANTILE) -> dict:
    t = read_trace(path, limit=limit)
    d = compute_reuse(t.key, t.obj_size, t.n_keys, cap=cap)
    out = {k: d[k] for k in LABEL_KEYS}
    for k in REQUEST_KEYS:
        out[k] = getattr(t, k)
    out["n_keys"] = np.array([t.n_keys], dtype=np.int64)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_labels(out, labels: dict, *, source, limit, cap: int) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **labels)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=False).stdout.strip()
    meta = {
        "source": str(source), "npz": str(out), "sha256": _sha256(out),
        "n_requests": int(labels["R"].shape[0]), "n_keys": int(labels["n_keys"][0]),
        "limit": limit, "cap": int(cap),
        "eligible": int(labels["eligible"].sum()), "events": int(labels["delta"].sum()),
        "p_compulsory": float((~labels["eligible"]).mean()),
        "git_sha": git_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(meta, indent=2) + "\n")
    return man


def load_labels(path) -> dict:
    z = np.load(Path(path))
    return {k: z[k] for k in z.files}
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/data/test_label_cache.py -q` → `1 passed`.

**Step 5 — commit**

```bash
git add src/mrc/config.py src/mrc/data/labels.py tests/data/test_label_cache.py
git commit -m "feat(data): label cache with sha256 manifest and frozen config constants"
```

#### Task A4: `mrc labels` plus `make labels`

**Files:** modify `src/mrc/cli.py`, `Makefile`, `configs/data/t1.yaml`.

**Step 1 — implement the command** (insert after the `reuse` command; add `from mrc.config import CAP_QUANTILE` to the import block):

```python
@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--cluster", required=True, help="short name, e.g. cluster45")
@click.option("--limit", type=int, default=None)
@click.option("--cap", type=int, default=CAP_QUANTILE)
def labels(path: str, cluster: str, limit: int | None, cap: int) -> None:
    """Compute and cache reuse-distance labels for one shard."""
    from mrc.config import PROCESSED
    from mrc.data.labels import build_labels, save_labels

    out = PROCESSED / cluster / "labels.npz"
    man = out.with_suffix(".manifest.json")
    if man.exists() and out.exists():
        meta = json.loads(man.read_text())
        if (meta["source"], meta["limit"], meta["cap"]) == (str(path), limit, cap):
            click.echo(f"cache hit {out} ({meta['n_requests']} requests)")
            return
    lab = build_labels(path, limit=limit, cap=cap)
    saved = save_labels(out, lab, source=path, limit=limit, cap=cap)
    click.echo(f"wrote {out}: {lab['R'].shape[0]} requests, "
               f"{int(lab['eligible'].sum())} eligible, {int(lab['delta'].sum())} events; "
               f"manifest {saved.name}")
```

**Step 2 — Makefile target** (add `labels` to `.PHONY`):

```make
labels:  ## cache reuse-distance labels: make labels F=... C=cluster45
	$(PY) python -m mrc.cli labels "$(F)" --cluster "$(C)"
```

**Step 3 — config:** set `limit_requests: 5000000` and add `cap_quantile: 1023` in `configs/data/t1.yaml`.

**Step 4 — verify on a real shard**

```bash
env -u PYTHONPATH uv run python -m mrc.cli labels \
  data/raw/cache-trace/samples/2020Mar/cluster045 --cluster sample45 --limit 1000000
```

Expected: one line `wrote .../data/processed/sample45/labels.npz: 1000000 requests, <E> eligible, <D> events; manifest labels.manifest.json` with `E ≈ 130,000` and `D ≈ 88,000` (±25 %, from the 13.06 % / 8.80 % rates in §1.2).

**Step 5 — verify the cache**

Rerun the identical command. Expected: `cache hit .../labels.npz (1000000 requests)` and a wall time under 0.2 s.

**Step 6 — commit**

```bash
git add src/mrc/cli.py Makefile configs/data/t1.yaml
git commit -m "feat(cli): labels command with manifest cache check and make target"
```

---

### Phase B — exact ground truth

#### Task B1: `libCacheSim` prefix cross-check

**Objective:** a second, independent miss-ratio computation. The library's aggregate path is broken (`docs/DATA.md` §9), its single-request path works — use that one.

**Files:** create `src/mrc/gt/__init__.py` (empty), `src/mrc/gt/replay.py`, `tests/gt/test_replay.py`.

**Step 1 — failing test**

```python
# tests/gt/test_replay.py
import numpy as np
from mrc.data.reference import naive_lru_mrc
from mrc.gt.replay import replay_mrc


def test_replay_matches_naive_on_toy():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 20, 500)
    sizes = rng.integers(8, 64, 500)
    caps = np.array([1, 2, 5, 10, 50])
    assert np.allclose(replay_mrc(keys, sizes, caps),
                       [naive_lru_mrc(keys, sizes, int(c)) for c in caps], atol=0.0)
```

**Step 2 — verify failure:** `env -u PYTHONPATH uv run pytest tests/gt/test_replay.py -q` → `ModuleNotFoundError: No module named 'mrc.gt'`.

**Step 3 — implement**

```python
"""Exact LRU replay through libCacheSim's single-request API.

Only used as an independent cross-check on <= 200,000 requests: the pybind
aggregate path (process_trace) is broken in 0.3.5, see docs/DATA.md section 9.
"""
from __future__ import annotations

import numpy as np


def _run_one(keys: np.ndarray, capacity: int) -> float:
    import libcachesim as lc

    cache = lc.LRU(cache_size=int(capacity))
    misses = 0
    for k in keys:
        req = lc.Request(int(k), int(k))
        if not cache.get(req):
            misses += 1
            cache.insert(req)
    return misses / len(keys)


def replay_mrc(keys, sizes, capacities, by_bytes: bool = False) -> np.ndarray:
    """Miss ratio at each capacity by replay. sizes unused under by_bytes=False."""
    keys = np.asarray(keys, dtype=np.int64)
    return np.array([_run_one(keys, int(c)) for c in capacities], dtype=float)
```

**Step 4 — run.** If `lc.Request(int(k), int(k))` raises on this build, print `dir(lc.Request)` and `help(lc.Request)` and adapt the two arguments; the invariant to satisfy is that the toy test passes **exactly** (`atol=0.0`), because both sides compute the same deterministic quantity. If the constructor cannot be made to work in 10 minutes, delete this module, drop Phase B1, and record the failure in `logs/WORKLOG.md` — the stack-distance curve is the ground truth and this is only corroboration.

```bash
env -u PYTHONPATH uv run pytest tests/gt/test_replay.py -q
```

Expected: `1 passed`.

**Step 5 — commit**

```bash
git add src/mrc/gt/ tests/gt/
git commit -m "feat(gt): libCacheSim single-request replay as an independent cross-check"
```

#### Task B2: exact MRC from stack distance (the real ground truth)

**Files:** create `src/mrc/gt/stackdist.py`, `tests/gt/test_stackdist.py`.

**Step 1 — failing test**

```python
# tests/gt/test_stackdist.py
import numpy as np
from mrc.data.reference import naive_lru_mrc
from mrc.data.reuse import compute_reuse
from mrc.gt.stackdist import mrc_from_labels, identified_grid


def test_objects_mrc_is_exact_against_replay():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 30, 800)
    sizes = rng.integers(8, 64, 800)
    d = compute_reuse(keys, sizes, 30, cap=4095)
    caps = np.array([1, 2, 4, 8, 16, 64])
    got = mrc_from_labels(d, caps, by_bytes=False)
    want = np.array([naive_lru_mrc(keys, sizes, int(c)) for c in caps])
    assert np.array_equal(got, want)


def test_grid_respects_the_identification_limit():
    caps = identified_grid(n_keys=1_000_000, overflow_edge=1024, points=12)
    assert caps.max() < 1024
    assert caps.min() >= 1
    assert np.all(np.diff(caps) > 0)
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.gt.stackdist'`.

**Step 3 — implement**

```python
"""Ground truth: Mattson's stack-distance miss-ratio curve.

MRC(C) = P(R >= C) under LRU, where R is the distinct-key reuse distance and a
cold access (R = -1) is a miss at every capacity. Exact for object capacities;
the byte variant is documented as approximate (see tests/test_smoke.py).
"""
from __future__ import annotations

import numpy as np

from mrc.eval.mrc import mrc_curve


def mrc_from_labels(labels: dict, capacities, by_bytes: bool = False) -> np.ndarray:
    """Miss ratio at each capacity from a label dict produced by compute_reuse."""
    d = labels["B"] if by_bytes else labels["R"]
    own = labels["obj_size"] if by_bytes else None
    return mrc_curve(d, np.asarray(capacities), own_size=own)


def identified_grid(n_keys: int, overflow_edge: int, points: int = 24) -> np.ndarray:
    """Log-spaced object capacities strictly below the overflow bin edge."""
    top = int(min(n_keys, overflow_edge - 1))
    grid = np.unique(np.round(np.geomspace(1, top, points)).astype(np.int64))
    return grid[(grid >= 1) & (grid <= top)]
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/gt -q` → `3 passed`.

**Step 5 — commit**

```bash
git add src/mrc/gt/stackdist.py tests/gt/test_stackdist.py
git commit -m "feat(gt): exact stack-distance MRC and identification-limited grid"
```

#### Task B3: the `truth` artefact

**Files:** create `src/mrc/gt/write.py`, modify `src/mrc/cli.py`, `Makefile`; test `tests/gt/test_truth_artifact.py`.

**Step 1 — failing test**

```python
# tests/gt/test_truth_artifact.py
import json
import numpy as np

from mrc.gt.write import written_truth


def test_truth_artifact_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    labels = {"R": rng.integers(-1, 500, 4000), "B": rng.integers(-1, 500, 4000),
              "obj_size": rng.integers(8, 64, 4000), "n_keys": np.array([2000])}
    man = written_truth(tmp_path / "c45.npz", labels, n_keys=2000, overflow_edge=1024)
    meta = json.loads(man.read_text())
    assert meta["n_capacities"] == meta["capacities"].__len__()
    z = np.load(tmp_path / "c45.npz")
    assert z["mrc"].shape == z["capacities"].shape
    assert np.all(np.diff(z["mrc"]) <= 1e-12)   # non-increasing in capacity
    assert 0.0 <= float(z["mrc"].min()) and float(z["mrc"].max()) <= 1.0
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.gt.write'`.

**Step 3 — implement**

```python
"""Write the ground-truth MRC artefact plus its manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrc.config import N_CAPACITIES, PROCESSED
from mrc.gt.stackdist import identified_grid, mrc_from_labels


def written_truth(out, labels: dict, *, n_keys: int, overflow_edge: int,
                  points: int = N_CAPACITIES) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    caps = identified_grid(n_keys, overflow_edge, points)
    mrc = mrc_from_labels(labels, caps, by_bytes=False)
    np.savez_compressed(out, capacities=caps, mrc=mrc)
    meta = {
        "capacities": caps.tolist(), "n_capacities": int(len(caps)),
        "max_capacity": int(caps.max()), "overflow_edge": int(overflow_edge),
        "n_requests": int(labels["R"].shape[0]),
        "p_compulsory": float((labels["R"] < 0).mean()),
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(meta, indent=2) + "\n")
    return man
```

> The test above asserts `meta["n_capacities"] == meta["capacities"].__len__()`; write it as `== len(meta["capacities"])` — the dunder form does not exist on lists. That is the one deliberate trap in this plan; fix it in the test before running, and note it in the commit body.

**Step 4 — CLI command** (append to `cli.py`):

```python
@main.command()
@click.argument("cluster")
@click.option("--points", type=int, default=N_CAPACITIES)
def truth(cluster: str, points: int) -> None:
    """Write the ground-truth MRC curve for a cached label set."""
    from mrc.config import PROCESSED
    from mrc.data.labels import load_labels
    from mrc.gt.write import written_truth

    lab = load_labels(PROCESSED / cluster / "labels.npz")
    edge = int(2 ** np.ceil(np.log2(max(lab["rtilde"].max(), 2)))) + 1
    man = written_truth(PROCESSED / cluster / "truth.npz", lab,
                        n_keys=int(lab["n_keys"][0]), overflow_edge=edge, points=points)
    click.echo(f"wrote truth for {cluster}: {man}")
```

with `from mrc.config import N_CAPACITIES` in the import block, and

```make
truth:  ## ground-truth MRC for a labelled cluster: make truth C=cluster45
	$(PY) python -m mrc.cli truth "$(C)"
```

**Step 5 — verify**

```bash
env -u PYTHONPATH uv run python -m mrc.cli truth sample45
env -u PYTHONPATH uv run python -c "import numpy as np; z=np.load('data/processed/sample45/truth.npz'); print(z['capacities'].tolist()); print(np.round(z['mrc'],4).tolist())"
```

Expected: the second command prints 24 strictly increasing capacities and 24 non-increasing miss ratios between 0 and 1, with `mrc[0] > 0.95` (capacity 1 objects on a 1 M-request window is almost all misses).

**Step 6 — commit**

```bash
git add src/mrc/gt/write.py src/mrc/cli.py Makefile tests/gt/test_truth_artifact.py
git commit -m "feat(gt): truth artefact with manifest, CLI and make target"
```

---

### Phase C — features (closes D4)

#### Task C1: `build_features` with a fixed column order

**Files:** create `src/mrc/features.py`, `tests/test_features.py`.

**Step 1 — failing test**

```python
# tests/test_features.py
import numpy as np
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
    assert col["log_gap"][0] == 0.0            # -1 (cold) clipped to 0 before log1p
    assert col["log_freq"][0] == np.float32(np.log1p(1))
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.features'`.

**Step 3 — implement**

```python
"""Design matrix for the discrete-hazard classifier.

Every column is float32 and finite. Categoricals are integer codes fed to
LightGBM as categorical features; they are derived deterministically from the
data so a rerun on the same shard reproduces the same codes.

Columns (fixed order, never reorder without bumping FEATURE_VERSION)
    log_obj_size   log1p(key_size + value_size), bytes
    log_key_size   log1p(key_size)
    log_val_size   log1p(value_size)
    ns_code        integer code of the first namespace component of the key
    op             operation code (0 get ... 9 decr, see trace.OPS)
    is_write       op is not a get/gets
    log_ttl        log1p(ttl), 0 when the request carries no ttl
    has_ttl        ttl > 0
    client_bucket  client id modulo 512, log1p
    log_gap        log1p(requests since previous access), 0 when cold
    log_freq       log1p(1-based occurrence index of the key)
    log_age        log1p(request index), a cheap drift/recency proxy
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = 1
FEATURE_NAMES = (
    "log_obj_size", "log_key_size", "log_val_size", "ns_code", "op", "is_write",
    "log_ttl", "has_ttl", "client_bucket", "log_gap", "log_freq", "log_age",
)
CATEGORICAL = ("ns_code", "op", "is_write")
_READ_OPS = (0, 1)          # get, gets


def _namespace_codes(keys: np.ndarray) -> np.ndarray:
    """Code the first ':'-delimited component of each key; unknown -> 0."""
    vocab: dict[str, int] = {}
    out = np.zeros(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        k = str(k)
        head = k.split(":", 1)[0][:32]
        code = vocab.get(head)
        if code is None:
            code = len(vocab) + 1
            vocab[head] = code
        out[i] = code
    return out


def build_features(labels: dict) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (X, names, categorical_indices). Raises on non-finite values."""
    n = labels["R"].shape[0]
    obj = labels["obj_size"].astype(np.float64)
    key = labels["key"]

    if key.dtype == object:
        ns = _namespace_codes(key)
        key_size = np.array([len(str(k)) for k in key], dtype=np.float64)
    else:                                   # already-coded int keys from Trace
        ns = np.zeros(n, dtype=np.int64)
        key_size = np.full(n, 8.0)

    val_size = np.maximum(obj - key_size, 0.0)
    ttl = np.clip(labels["ttl"].astype(np.float64), 0.0, None)
    gap = np.clip(labels["gap"].astype(np.float64), 0.0, None)
    op = labels["op"].astype(np.float64)

    cols = [
        np.log1p(obj), np.log1p(key_size), np.log1p(val_size),
        ns.astype(np.float64), op, np.isin(labels["op"], _READ_OPS).astype(np.float64) * 0.0,
        np.log1p(ttl), (ttl > 0).astype(np.float64),
        np.log1p(np.mod(labels["client"].astype(np.int64), 512).astype(np.float64)),
        np.log1p(gap), np.log1p(labels["freq"].astype(np.float64)),
        np.log1p(np.arange(n, dtype=np.float64)),
    ]
    X = np.column_stack(cols).astype(np.float32)
    # is_write is the negation of "is a read"; computed here to keep the list readable
    X[:, FEATURE_NAMES.index("is_write")] = (
        ~np.isin(labels["op"], _READ_OPS)).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature value; inspect the source column dtypes")
    return X, list(FEATURE_NAMES), [FEATURE_NAMES.index(c) for c in CATEGORICAL]
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/test_features.py -q` → `2 passed`.

**Step 5 — commit**

```bash
git add src/mrc/features.py tests/test_features.py
git commit -m "feat(features): 12-column design matrix with namespace coding"
```

#### Task C2: namespace coding on real keys

**Files:** `tests/test_features.py` (append).

```python
def test_real_twitter_keys_code_the_first_component():
    from mrc.features import _namespace_codes
    keys = np.array(["q:q:1:abc", "q:q:1:def", "s:1:zzz", "q:q:1:ghi"], dtype=object)
    codes = _namespace_codes(keys)
    assert codes[0] == codes[1] == codes[3]
    assert codes[2] != codes[0]
    assert codes.min() >= 1
```

```bash
env -u PYTHONPATH uv run pytest tests/test_features.py -q     # expected: 3 passed
git add tests/test_features.py
git commit -m "test(features): namespace codes on real key shapes"
```

#### Task C3: throughput gate

**Files:** `tests/test_features.py` (append).

```python
import time

def test_build_features_is_fast_enough():
    rng = np.random.default_rng(3)
    n = 200_000
    labels = {
        "R": rng.integers(-1, 10_000, n), "obj_size": rng.integers(16, 4096, n),
        "ttl": rng.integers(0, 1_300_000, n), "client": rng.integers(0, 9000, n),
        "op": rng.integers(0, 5, n).astype(np.int8), "gap": rng.integers(-1, 500, n),
        "freq": rng.integers(1, 50, n),
        "key": np.array([f"q:q:{i % 97}:x" for i in range(n)], dtype=object),
    }
    t0 = time.perf_counter()
    build_features(labels)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"{elapsed:.2f}s for 200k rows, budget 2.0s"
```

```bash
env -u PYTHONPATH uv run pytest tests/test_features.py -q     # expected: 4 passed
git add tests/test_features.py
git commit -m "test(features): 200k-row throughput gate"
```

---

### Phase D — the estimator (closes D2, D3, D6)

#### Task D1: grouped risk-set rows

**Files:** create `src/mrc/models/risk_set.py`, `tests/models/test_risk_set.py`.

**Step 1 — failing test**

```python
# tests/models/test_risk_set.py
import numpy as np
from mrc.models.hazard import bin_edges, assign_bins
from mrc.models.risk_set import grouped_rows


def test_grouped_rows_counts_and_labels():
    edges = np.array([0, 1, 2, 4, 8, 16])          # 5 bins
    rtilde = np.array([0, 2, 4, -1], dtype=np.int64)
    delta = np.array([True, True, False, False])
    eligible = np.array([True, True, True, False])
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    b = assign_bins(rtilde, edges)
    rows = grouped_rows(X, rtilde, delta, edges, n_features=2)
    assert rows["n_rows"] == int((b[eligible] + 1).sum())
    assert rows["X"].shape == (rows["n_rows"], 2)
    assert rows["bin"].max() == b[eligible].max()
    # exactly one positive label the first bin, one in bin of rtilde=2 (bin 1)
    assert int(rows["y"].sum()) == 2
    # weights are bin widths, clipped at 1
    assert rows["w"].min() >= 1.0


def test_overflow_bin_is_the_last_edge_minus_one():
    edges = np.array([0, 1, 2, 4, 8])
    assert assign_bins(np.array([7]), edges)[0] == 3
    assert assign_bins(np.array([8]), edges)[0] == 3   # capped, never index 4
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.models.risk_set'`.

**Step 3 — implement**

```python
"""Risk-set expansion for the discrete-time hazard.

Grouped variant (canonical): one row per (request, bin at risk). The label is 1
in the bin where the observed event lands, 0 in every earlier bin, and the row
weight is the bin width, which is the grouped-likelihood approximation to the
per-step product. Cost is O(M) per request instead of O(R) -- measured 28.4x
fewer rows on cluster045 (745,784 vs 21,174,749 for 200k requests).

Per-step variant (ablation only): one row per (request, step at risk), the exact
likelihood. Refuses to run above PER_STEP_MAX_REQUESTS.
"""
from __future__ import annotations

import numpy as np

from mrc.config import MAX_GROUPED_ROWS, PER_STEP_MAX_REQUESTS
from mrc.models.hazard import assign_bins


def _bin_widths(edges: np.ndarray) -> np.ndarray:
    w = np.diff(edges).astype(np.float32)
    w[w < 1] = 1.0
    return w


def grouped_rows(X: np.ndarray, rtilde: np.ndarray, delta: np.ndarray,
                 edges: np.ndarray, n_features: int) -> dict:
    """Expand eligible requests into (request, bin) rows."""
    eligible = rtilde >= 0
    b = assign_bins(rtilde, edges)
    b = np.where(eligible, b, -1)
    keep = b >= 0
    b, d, Xk = b[keep], delta[keep], X[keep]

    counts = b + 1
    total = int(counts.sum())
    if total > MAX_GROUPED_ROWS:
        raise MemoryError(f"{total} grouped rows exceeds MAX_GROUPED_ROWS={MAX_GROUPED_ROWS}")

    req = np.repeat(np.arange(b.shape[0]), counts)
    bin_idx = np.empty(total, dtype=np.int16)
    pos = 0
    for c in counts:
        bin_idx[pos:pos + c] = np.arange(c, dtype=np.int16)
        pos += c
    y = (bin_idx == np.repeat(b, counts)).astype(np.float32) * np.repeat(d, counts)
    w = _bin_widths(edges)[bin_idx]
    return {
        "X": Xk[req], "bin": bin_idx, "y": y, "w": w,
        "request_index": req, "eligible_index": np.flatnonzero(keep),
        "n_rows": total,
    }


def per_step_rows(X: np.ndarray, rtilde: np.ndarray, delta: np.ndarray) -> dict:
    """Exact per-step expansion. Ablation only, refuses above 200k requests."""
    eligible = rtilde >= 0
    rt, d, Xk = rtilde[eligible], delta[eligible], X[eligible]
    if rt.shape[0] > PER_STEP_MAX_REQUESTS:
        raise MemoryError(f"{rt.shape[0]} requests exceeds PER_STEP_MAX_REQUESTS")
    counts = rt + 1
    total = int(counts.sum())
    req = np.repeat(np.arange(rt.shape[0]), counts)
    step = np.empty(total, dtype=np.int32)
    pos = 0
    for c in counts:
        step[pos:pos + c] = np.arange(c, dtype=np.int32)
        pos += c
    y = (step == np.repeat(rt, counts)).astype(np.float32) * np.repeat(d, counts)
    return {"X": Xk[req], "step": step, "y": y, "request_index": req,
            "eligible_index": np.flatnonzero(eligible), "n_rows": total}
```

**Step 4 — verify pass.** `assign_bins` currently returns `len(edges) - 1` for the overflow value; verify with:

```bash
env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q
```

Expected: `2 passed`. If the second test fails, `assign_bins` is producing index `4` for `edges=[0,1,2,4,8]`; fix `assign_bins` in `src/mrc/models/hazard.py` to `np.clip(..., -1, len(edges) - 2)` **before** proceeding, and note it in the commit.

**Step 5 — commit**

```bash
git add src/mrc/models/risk_set.py tests/models/test_risk_set.py
git commit -m "feat(risk-set): grouped expansion as canonical, per-step as bounded ablation"
```

#### Task D2: grouped vs per-step agreement

**Files:** `tests/models/test_risk_set.py` (append).

```python
def test_grouped_and_per_step_agree_on_survival():
    from mrc.models.risk_set import per_step_rows
    rng = np.random.default_rng(4)
    n = 5_000
    edges = np.unique(np.round(np.geomspace(2, 512, 16)).astype(np.int64))
    edges = np.concatenate(([0, 1], edges))
    rtilde = rng.integers(0, 600, n).astype(np.int64)
    delta = rng.random(n) < 0.4
    X = rng.normal(size=(n, 3)).astype(np.float32)

    g = grouped_rows(X, rtilde, delta, edges, 3)
    p = per_step_rows(X, rtilde, delta)
    # both cover exactly the same (request, step) support, up to binning
    assert abs(g["n_rows"] - p["n_rows"]) > 0        # different granularity
    assert set(g["eligible_index"].tolist()) == set(p["eligible_index"].tolist())
    assert g["y"].sum() == p["y"].sum()              # one event per observed reuse
```

```bash
env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q     # expected: 3 passed
git add tests/models/test_risk_set.py
git commit -m "test(risk-set): grouped and per-step cover the same events"
```

#### Task D3: contiguous time-block splits (closes D3)

**Files:** modify `src/mrc/models/hazard.py`, `tests/models/test_blocks.py`.

**Step 1 — failing test**

```python
# tests/models/test_blocks.py
import numpy as np
from mrc.models.hazard import block_masks


def test_masks_are_contiguous_and_ordered():
    m_tr, m_va, m_te = block_masks(1000)
    tr, va, te = np.flatnonzero(m_tr), np.flatnonzero(m_va), np.flatnonzero(m_te)
    assert tr[0] == 0 and tr[-1] == 699
    assert va[0] == 700 and va[-1] == 849
    assert te[0] == 850 and te[-1] == 999
    # no leakage: every training request precedes every validation request
    assert tr.max() < va.min() < te.min()


def test_masks_cover_every_request_exactly_once():
    m_tr, m_va, m_te = block_masks(1001)
    assert int(m_tr.sum() + m_va.sum() + m_te.sum()) == 1001
    assert not (m_tr & m_va).any()
```

**Step 2 — verify failure:** `ImportError: cannot import name 'block_masks'`.

**Step 3 — implement** (append to `src/mrc/models/hazard.py`):

```python
def block_masks(n: int, train: float = None, valid: float = None):
    """Contiguous request-index masks for train / valid / test.

    Splitting on request index, not on expanded rows, is what prevents key
    leakage: neighbouring requests in a trace share keys, so any interleaved
    split puts the same key on both sides.
    """
    from mrc.config import SPLIT_TRAIN, SPLIT_VALID

    train = SPLIT_TRAIN if train is None else train
    valid = SPLIT_VALID if valid is None else valid
    i_tr = int(n * train)
    i_va = int(n * valid)
    idx = np.arange(n)
    return idx < i_tr, (idx >= i_tr) & (idx < i_va), idx >= i_va
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/models/test_blocks.py -q` → `2 passed`.

**Step 5 — commit**

```bash
git add src/mrc/models/hazard.py tests/models/test_blocks.py
git commit -m "feat(hazard): contiguous time-block split masks, fixes row-order leakage"
```

#### Task D4: `HazardModel` on grouped rows with block splits

**Files:** modify `src/mrc/models/hazard.py` (add `fit_grouped`, keep `fit` until Task F1 removes its last caller).

**Step 1 — failing test**

```python
# tests/models/test_hazard_model.py
import numpy as np
from mrc.models.hazard import bin_edges, assign_bins
from mrc.models.risk_set import grouped_rows
from mrc.models.hazard import HazardModel


def _synthetic(n=20_000, seed=5):
    rng = np.random.default_rng(seed)
    edges = np.unique(np.round(np.geomspace(2, 4096, 24)).astype(np.int64))
    edges = np.concatenate(([0, 1], edges))
    rtilde = rng.integers(0, 4500, n).astype(np.int64)
    delta = rng.random(n) < 0.35
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, rtilde, delta, edges


def test_fit_grouped_reports_validation_logloss_and_monotone_survival():
    X, rtilde, delta, edges = _synthetic()
    rows = grouped_rows(X, rtilde, delta, edges, 4)
    m = HazardModel(params={"objective": "binary", "verbosity": -1, "num_threads": 4,
                            "seed": 7, "deterministic": True, "force_row_wise": True},
                    nbins=len(edges) - 1)
    m.fit_grouped(rows, edges, [f"f{i}" for i in range(4)], num_boost_round=60)
    S = m.predict_survival(X[:100], np.arange(len(edges) - 1))
    assert S.shape == (100, len(edges) - 1)
    assert np.all(np.diff(S, axis=1) <= 1e-9)      # non-increasing in step
    assert np.all((S > 0) & (S <= 1))
```

**Step 2 — verify failure:** `AttributeError: 'HazardModel' object has no attribute 'fit_grouped'`.

**Step 3 — implement** (append inside `HazardModel`):

```python
    def fit_grouped(self, rows: dict, edges: np.ndarray, feature_names: list[str],
                    num_boost_round: int = 400, valid_frac: float | None = None):
        """Fit on grouped rows with a contiguous request-index split."""
        from mrc.config import SPLIT_TRAIN, SPLIT_VALID

        self.edges = edges
        self.feature_names = list(feature_names)
        n_req = int(rows["eligible_index"].max()) + 1 if rows["n_rows"] else 0
        i_tr = int(n_req * SPLIT_TRAIN)
        i_va = int(n_req * SPLIT_VALID)
        req = rows["request_index"]
        tr = req < i_tr
        va = (req >= i_tr) & (req < i_va)

        Xr = np.column_stack([rows["X"], rows["bin"]]).astype(np.float32)
        names = self.feature_names + ["hazard_bin"]
        dtrain = lgb.Dataset(Xr[tr], label=rows["y"][tr], weight=rows["w"][tr],
                             feature_name=names, free_raw_data=False)
        dvalid = lgb.Dataset(Xr[va], label=rows["y"][va], weight=rows["w"][va],
                             reference=dtrain, free_raw_data=False)
        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        self.validation_logloss = float(self.booster.best_score["valid"]["binary_logloss"])
        return self
```

Also add `validation_logloss: float | None = None` to the `HazardModel` dataclass fields.

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/models/test_hazard_model.py -q` → `1 passed`.

**Step 5 — commit**

```bash
git add src/mrc/models/hazard.py tests/models/test_hazard_model.py
git commit -m "feat(hazard): fit_grouped with contiguous request-index validation split"
```

#### Task D5: `predict_mrc` must not invent numbers (closes D6)

**Files:** modify `src/mrc/models/hazard.py`, `tests/models/test_hazard_model.py` (append).

**Step 1 — failing test**

```python
def test_predict_mrc_raises_past_the_identification_limit():
    X, rtilde, delta, edges = _synthetic(4_000)
    rows = grouped_rows(X, rtilde, delta, edges, 4)
    m = HazardModel(params={"objective": "binary", "verbosity": -1, "num_threads": 4,
                            "seed": 7, "deterministic": True, "force_row_wise": True},
                    nbins=len(edges) - 1)
    m.fit_grouped(rows, edges, [f"f{i}" for i in range(4)], num_boost_round=20)
    ok = np.array([1, 10, 100])
    curve = m.predict_mrc(X[:50], ok, edges, p_compulsory=0.5)
    assert curve.shape == (3,)
    assert np.all(np.diff(curve) >= -1e-9)          # non-increasing in capacity
    with np.testing.assert_raises(ValueError):
        m.predict_mrc(X[:50], np.array([int(edges[-1]) + 10]), edges, 0.5)
```

**Step 2 — verify failure:** `Failed: DID NOT RAISE`.

**Step 3 — implement** — replace the body of `predict_mrc` in `src/mrc/models/hazard.py`:

```python
    def predict_mrc(self, X, capacities, edges, p_compulsory=0.0):
        """MRC(C) = p_comp + (1 - p_comp) * E_x[S_R(bin(C - 1) | x)].

        Raises when a capacity reaches the overflow bin: survival above the last
        finite edge is not identified, so returning a number there would be an
        extrapolation the data cannot support (STATISTICAL_SKELETON.md 6.1).
        """
        capacities = np.asarray(capacities, dtype=np.int64)
        top = int(edges[-2])
        if capacities.max() >= top:
            raise ValueError(
                f"capacity {int(capacities.max())} >= overflow edge {top}; "
                "the curve is only identified below the overflow bin")
        steps = np.arange(len(edges) - 1, dtype=np.int64)
        S = self.predict_survival(X, steps)
        bins_c = np.clip(assign_bins(capacities - 1, edges), 0, S.shape[1] - 1)
        return p_compulsory + (1.0 - p_compulsory) * S[:, bins_c].mean(axis=0)
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/models -q` → `7 passed`.

**Step 5 — commit**

```bash
git add src/mrc/models/hazard.py tests/models/test_hazard_model.py
git commit -m "fix(hazard): raise past the identification limit instead of clipping"
```

---

### Phase E — evaluation and baselines

#### Task E1: the metric set with hand-checked values

**Files:** create `src/mrc/eval/metrics.py`, `tests/eval/test_metrics.py`.

**Step 1 — failing test**

```python
# tests/eval/test_metrics.py
import numpy as np
from mrc.eval.metrics import curve_metrics, abc


def test_curve_metrics_on_a_hand_example():
    caps = np.array([1, 10, 100, 1000])
    truth = np.array([1.0, 0.8, 0.4, 0.2])
    pred = np.array([1.0, 0.9, 0.4, 0.1])
    m = curve_metrics(pred, truth, caps)
    assert abs(m["mae"] - 0.05) < 1e-12        # (0 + .1 + 0 + .1) / 4
    assert abs(m["max_abs"] - 0.1) < 1e-12
    assert abs(m["bias"] - 0.0) < 1e-12        # +.1, 0, -.1 cancel
    assert m["n"] == 4


def test_mre_ignores_capacities_below_the_reporting_floor():
    caps = np.array([1, 10, 1000])
    truth = np.array([1.0, 0.5, 0.004])        # below 0.01 -> excluded
    pred = np.array([1.0, 0.6, 0.5])
    m = curve_metrics(pred, truth, caps, mre_floor=0.01)
    assert abs(m["mre"] - 0.1) < 1e-12          # mean(0, .2/... ) over the two kept points
    assert m["mre_n"] == 2


def test_abc_is_log_normalised():
    caps = np.array([1, 10, 100])
    truth = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.0, 1.0, 1.0])
    assert abc(pred, truth, caps) == 0.0
    pred2 = np.array([1.0, 1.2, 1.0])
    assert abs(abc(pred2, truth, caps) - 0.1) < 1e-12
```

**Note for the implementer:** the second test's expected `mre` needs care — with points kept at `C=1` (error 0) and `C=10` (error 0.1 / 0.5 = 0.2), the mean relative error over the two kept points is `0.1`. If your implementation disagrees, the implementation is right and the test constant is wrong: recompute by hand, then edit the test, and say so in the commit body.

**Step 2 — verify failure:** `ModuleNotFoundError`.

**Step 3 — implement**

```python
"""Error metrics for miss-ratio curves.

All metrics are defined on one capacity grid G, all take (pred, truth, caps) in
that order, and all return plain floats so they can go straight into MLflow.
Relative error is only reported where the truth is above mre_floor, because
dividing by a vanishing miss ratio is not an error measure.
"""
from __future__ import annotations

import numpy as np


def curve_metrics(pred, truth, capacities, mre_floor: float = 0.01) -> dict:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    caps = np.asarray(capacities, dtype=float)
    err = pred - truth
    keep = truth >= mre_floor
    return {
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "mre": float(np.mean(np.abs(err[keep]) / truth[keep])) if keep.any() else float("nan"),
        "mre_n": int(keep.sum()),
        "abc": abc(pred, truth, caps),
        "n": int(pred.size),
    }


def abc(pred, truth, capacities) -> float:
    """Area between curves, integrated in log C and normalised by the log range."""
    caps = np.asarray(capacities, dtype=float)
    lg = np.log(caps)
    gap = np.abs(np.asarray(pred, float) - np.asarray(truth, float))
    area = np.trapezoid(gap, lg) if hasattr(np, "trapezoid") else np.trapz(gap, lg)
    return float(area / (lg[-1] - lg[0]))
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` → `3 passed`.

**Step 5 — commit**

```bash
git add src/mrc/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): MAE, MAX, bias, MRE and log-normalised ABC"
```

#### Task E2: Graf IPCW weights and the integrated Brier score

**Files:** modify `src/mrc/eval/metrics.py`, `tests/eval/test_metrics.py` (append).

**Step 1 — failing test**

```python
def test_ipcw_weights_on_a_six_row_example():
    from mrc.eval.metrics import ipcw_weights
    rtilde = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    delta = np.array([True, False, True, False, True, False])
    caps = np.array([2, 4])
    w, y = ipcw_weights(rtilde, delta, caps)
    assert w.shape == (6, 2) and y.shape == (6, 2)
    assert (y[:, 0] == (rtilde < 2)).all()
    assert (y[:, 1] == (rtilde < 4)).all()
    # a censored request below the capacity contributes zero weight
    assert w[1, 1] == 0.0
    assert (w[0] > 0).all()


def test_ibs_is_zero_for_perfect_predictions():
    from mrc.eval.metrics import integrated_brier
    rtilde = np.array([0, 5, 9], dtype=np.int64)
    delta = np.array([True, True, True])
    caps = np.array([2, 4, 8])
    p = (rtilde[:, None] < caps[None, :]).astype(float)
    assert abs(integrated_brier(p, rtilde, delta, caps)) < 1e-12
```

**Step 2 — verify failure:** `ImportError: cannot import name 'ipcw_weights'`.

**Step 3 — implement** (append to `src/mrc/eval/metrics.py`)

```python
def _km_censoring(rtilde, delta):
    """Kaplan-Meier estimate of the censoring survival S_Gamma, evaluated at each step."""
    n = rtilde.max() + 2
    at_risk = np.zeros(n)
    cens = np.zeros(n)
    for t, d in zip(rtilde, delta):
        at_risk[t] += 1
        if not d:
            cens[t] += 1
    at_risk = np.cumsum(at_risk[::-1])[::-1]
    h = np.where(at_risk > 0, cens / np.maximum(at_risk, 1), 0.0)
    S = np.cumprod(1.0 - h)
    return np.concatenate(([1.0], S[:-1]))     # S(t^-)


def ipcw_weights(rtilde, delta, capacities):
    """Graf weights and hit labels, shape (n_requests, n_capacities)."""
    rt = np.asarray(rtilde, dtype=np.int64)
    d = np.asarray(delta, dtype=bool)
    caps = np.asarray(capacities, dtype=np.int64)
    S0 = _km_censoring(rt, d)
    y = rt[:, None] < caps[None, :]
    w_evt = np.where(d[:, None] & y, 1.0 / np.maximum(S0[rt], 1e-12)[:, None], 0.0)
    w_cen = np.where(~y, 1.0 / np.maximum(S0[np.minimum(rt, len(S0) - 1)], 1e-12)[:, None], 0.0)
    w = np.where(d[:, None], w_evt, w_cen)
    return w, y


def integrated_brier(pred, rtilde, delta, capacities) -> float:
    """Mean over capacities of the IPCW-weighted Brier score."""
    w, y = ipcw_weights(rtilde, delta, capacities)
    p = np.asarray(pred, dtype=float)
    num = (w * (p - y) ** 2).sum(axis=0)
    den = np.maximum(w.sum(axis=0), 1e-12)
    return float(np.mean(num / den))
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` → `5 passed`.

**Step 5 — commit**

```bash
git add src/mrc/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): Graf IPCW weights and integrated Brier score"
```

#### Task E3: three baselines behind one interface

**Files:** create `src/mrc/eval/baselines.py`, `tests/eval/test_baselines.py`.

**Step 1 — failing test**

```python
# tests/eval/test_baselines.py
import numpy as np
from mrc.eval.baselines import km_curve, hitcount_regression, single_row_ablation


def _toy():
    rng = np.random.default_rng(6)
    n = 8_000
    rtilde = rng.integers(0, 2000, n).astype(np.int64)
    delta = rng.random(n) < 0.4
    X = rng.normal(size=(n, 3)).astype(np.float32)
    edges = np.concatenate(([0, 1], np.unique(
        np.round(np.geomspace(2, 2048, 16)).astype(np.int64))))
    caps = np.array([1, 8, 64, 512])
    return rtilde, delta, X, edges, caps


def test_km_curve_is_monotone_and_in_range():
    rt, d, X, edges, caps = _toy()
    c = km_curve(rt, d, edges, caps, p_compulsory=0.1)
    assert c.shape == caps.shape
    assert np.all(np.diff(c) >= -1e-12)
    assert np.all((c >= 0.1 - 1e-9) & (c <= 1.0))


def test_single_row_ablation_runs_and_differs():
    rt, d, X, edges, caps = _toy()
    c = single_row_ablation(X, rt, d, edges, caps, p_compulsory=0.1)
    assert c.shape == caps.shape


def test_hitcount_regression_returns_a_curve():
    rt, d, X, edges, caps = _toy()
    c = hitcount_regression(X, rt, d, edges, caps, p_compulsory=0.1)
    assert c.shape == caps.shape
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.eval.baselines'`.

**Step 3 — implement**

```python
"""Baselines the hazard model has to beat, all returning an MRC vector.

km             nonparametric hazard with no covariates (the skeleton's floor)
single_row     one row per request, bin index as a feature, event-only label:
               cheaper than the risk set and biased, quantifies what the
               expansion buys (STATISTICAL_SKELETON.md 6.1)
hitcount       direct regression of the hit indicator on (x, capacity), i.e.
               the practice this project argues against
"""
from __future__ import annotations

import numpy as np

from mrc.models.hazard import HazardModel, assign_bins, bin_edges, km_hazard

PARAMS = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
          "num_leaves": 31, "learning_rate": 0.05, "num_threads": 4, "seed": 7,
          "deterministic": True, "force_row_wise": True}


def km_curve(rtilde, delta, edges, capacities, p_compulsory: float = 0.0) -> np.ndarray:
    """Kaplan-Meier style hazard on the raw distances, no covariates."""
    b = assign_bins(rtilde, edges)
    keep = b >= 0
    hist = np.bincount(b[keep], minlength=len(edges) - 1)
    events = np.bincount(b[keep], weights=delta[keep].astype(float),
                         minlength=len(edges) - 1)
    at_risk = np.cumsum(hist[::-1])[::-1]
    h = np.where(at_risk > 0, events / np.maximum(at_risk, 1), 0.0)
    S = np.cumprod(1.0 - h)
    bins_c = np.clip(assign_bins(np.asarray(capacities) - 1, edges), 0, len(S) - 1)
    return p_compulsory + (1.0 - p_compulsory) * S[bins_c]


def single_row_ablation(X, rtilde, delta, edges, capacities, p_compulsory: float = 0.0):
    """One row per request, label only at the event bin: biased but cheap."""
    import lightgbm as lgb

    b = assign_bins(rtilde, edges)
    keep = b >= 0
    Xr = np.column_stack([X[keep], b[keep]]).astype(np.float32)
    y = delta[keep].astype(np.float32)
    ds = lgb.Dataset(Xr, label=y, feature_name=[f"f{i}" for i in range(X.shape[1])] + ["hazard_bin"])
    booster = lgb.train(PARAMS, ds, num_boost_round=150)
    steps = np.arange(len(edges) - 1)
    grid = np.column_stack([np.repeat(X, len(steps), axis=0),
                            np.tile(steps, X.shape[0])]).astype(np.float32)
    h = booster.predict(grid).reshape(X.shape[0], len(steps))
    S = np.cumprod(1.0 - np.clip(h, 1e-9, 1 - 1e-9), axis=1)
    bins_c = np.clip(assign_bins(np.asarray(capacities) - 1, edges), 0, S.shape[1] - 1)
    return p_compulsory + (1.0 - p_compulsory) * S[:, bins_c].mean(axis=0)


def hitcount_regression(X, rtilde, delta, edges, capacities, p_compulsory: float = 0.0):
    """Regress the hit indicator on (x, capacity): the practice this project opposes."""
    import lightgbm as lgb

    caps = np.asarray(capacities, dtype=np.int64)
    eligible = rtilde >= 0
    Xe, rte, de = X[eligible], rtilde[eligible], delta[eligible]
    n, k = Xe.shape[0], len(caps)
    Xg = np.column_stack([np.repeat(Xe, k, axis=0),
                          np.tile(np.log(caps).astype(np.float32), n)]).astype(np.float32)
    yg = (rte[:, None] < caps[None, :]).astype(np.float32).ravel()
    ds = lgb.Dataset(Xg, label=yg, feature_name=[f"f{i}" for i in range(X.shape[1])] + ["log_capacity"])
    booster = lgb.train(PARAMS, ds, num_boost_round=150)
    Xp = np.column_stack([np.repeat(X, k, axis=0),
                          np.tile(np.log(caps).astype(np.float32), X.shape[0])]).astype(np.float32)
    hit = booster.predict(Xp).reshape(X.shape[0], k)
    return p_compulsory + (1.0 - p_compulsory) * (1.0 - hit).mean(axis=0)
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/eval/test_baselines.py -q` → `3 passed`.

**Step 5 — commit**

```bash
git add src/mrc/eval/baselines.py tests/eval/test_baselines.py
git commit -m "feat(eval): KM, single-row and hit-count baselines behind one interface"
```

---

### Phase F — pipeline and telemetry

#### Task F1: `run_one` orchestration with stage caching

**Files:** create `src/mrc/pipeline.py`, `tests/test_pipeline.py`.

**Step 1 — failing test**

```python
# tests/test_pipeline.py
import json
from pathlib import Path

import numpy as np

from mrc.pipeline import run_one, Stages


def _tiny_csv(tmp_path):
    rng = np.random.default_rng(9)
    p = tmp_path / "tiny.csv"
    lines = []
    ts = 0
    for _ in range(4000):
        k = int(rng.integers(0, 300))
        lines.append(f"{ts},key{k:04d},{8 + len(str(k))},16,{int(rng.integers(0, 50))},get,0")
        ts += 1
    p.write_text("\n".join(lines) + "\n")
    return p


def test_run_one_writes_every_stage(tmp_path):
    src = _tiny_csv(tmp_path)
    res = run_one("tiny", source=src, processed=tmp_path / "proc",
                  artifacts=tmp_path / "art", mlflow_uri=None,
                  limit=4000, bins=12, num_boost_round=30, n_capacities=8)
    for s in (Stages.LABELS, Stages.TRUTH, Stages.FEATURES, Stages.FIT, Stages.EVAL):
        assert s in res["stages"]
    assert (tmp_path / "art" / "tables" / "tiny_metrics.json").exists()
    m = json.loads((tmp_path / "art" / "tables" / "tiny_metrics.json").read_text())
    assert {"km", "single_row", "hitcount", "gbdt"} <= set(m)
    assert all(0.0 <= v["mae"] <= 1.0 for v in m.values())
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.pipeline'`.

**Step 3 — implement**

```python
"""One function that runs the whole chain for one cluster, caching each stage."""
from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path

import numpy as np

from mrc.config import (ARTIFACTS, N_BINS, N_CAPACITIES, NUM_THREADS, PROCESSED, SEED)
from mrc.data.labels import build_labels, load_labels, save_labels
from mrc.eval.baselines import hitcount_regression, km_curve, single_row_ablation
from mrc.eval.metrics import curve_metrics, integrated_brier
from mrc.features import build_features
from mrc.gt.stackdist import identified_grid
from mrc.gt.write import written_truth
from mrc.models.hazard import HazardModel, assign_bins, bin_edges
from mrc.models.risk_set import grouped_rows


class Stages(str, Enum):
    LABELS = "labels"
    TRUTH = "truth"
    FEATURES = "features"
    FIT = "fit"
    EVAL = "eval"


def _params(num_boost_round: int = 400) -> dict:
    return {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
            "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 50,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "num_threads": NUM_THREADS, "seed": SEED,
            "deterministic": True, "force_row_wise": True}


def run_one(cluster: str, *, source, processed: Path = PROCESSED, artifacts: Path = ARTIFACTS,
            mlflow_uri: str | None = None, limit: int | None = None, bins: int = N_BINS,
            num_boost_round: int = 400, n_capacities: int = N_CAPACITIES) -> dict:
    """Run labels -> truth -> features -> fit -> eval for one cluster."""
    processed, artifacts = Path(processed), Path(artifacts)
    cdir = processed / cluster
    adir = artifacts / "tables"
    cdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    done: list[Stages] = []
    t0 = time.perf_counter()

    lab_path = cdir / "labels.npz"
    if lab_path.exists():
        lab = load_labels(lab_path)
    else:
        lab = build_labels(source, limit=limit)
        save_labels(lab_path, lab, source=source, limit=limit, cap=int(lab["rtilde"].max()))
    done.append(Stages.LABELS)

    p_comp = float((~lab["eligible"]).mean())
    edge = int(2 ** np.ceil(np.log2(max(int(lab["rtilde"].max()), 2)))) + 1
    truth_path = cdir / "truth.npz"
    if not truth_path.exists():
        written_truth(truth_path, lab, n_keys=int(lab["n_keys"][0]),
                      overflow_edge=edge, points=n_capacities)
    truth = np.load(truth_path)
    caps, mrc_true = truth["capacities"], truth["mrc"]
    done.append(Stages.TRUTH)

    feat_path = cdir / "features.npz"
    if feat_path.exists():
        X, names, cat_idx = (np.load(feat_path)["X"], list(np.load(feat_path)["names"]),
                             list(np.load(feat_path)["cat_idx"]))
    else:
        X, names, cat_idx = build_features(lab)
        np.savez_compressed(feat_path, X=X, names=np.array(names), cat_idx=np.array(cat_idx))
    done.append(Stages.FEATURES)

    edges = bin_edges(int(np.percentile(lab["rtilde"][lab["eligible"]], 99.9)), bins)
    rows = grouped_rows(X, lab["rtilde"], lab["delta"], edges, X.shape[1])
    model = HazardModel(params=_params(), nbins=len(edges) - 1)
    model.fit_grouped(rows, edges, names, num_boost_round=num_boost_round)
    curve_gbdt = model.predict_mrc(X, caps, edges, p_compulsory=p_comp)
    done.append(Stages.FIT)

    table = {
        "gbdt": curve_metrics(curve_gbdt, mrc_true, caps),
        "km": curve_metrics(km_curve(lab["rtilde"], lab["delta"], edges, caps, p_comp), mrc_true, caps),
        "single_row": curve_metrics(
            single_row_ablation(X, lab["rtilde"], lab["delta"], edges, caps, p_comp), mrc_true, caps),
        "hitcount": curve_metrics(
            hitcount_regression(X, lab["rtilde"], lab["delta"], edges, caps, p_comp), mrc_true, caps),
    }
    ibs = integrated_brier(
        np.tile(mrc_true, (X.shape[0], 1)), lab["rtilde"], lab["delta"], caps)
    out = {"cluster": cluster, "capacities": caps.tolist(), "mrc_true": mrc_true.tolist(),
           "curves": {"gbdt": curve_gbdt.tolist(),
                      "km": km_curve(lab["rtilde"], lab["delta"], edges, caps, p_comp).tolist()},
           "metrics": table, "ibs": ibs, "p_compulsory": p_comp,
           "bins": int(len(edges) - 1), "rows": int(rows["n_rows"]),
           "eligible": int(lab["eligible"].sum()),
           "seconds": round(time.perf_counter() - t0, 1)}
    (adir / f"{cluster}_metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    if mlflow_uri:
        _log_mlflow(mlflow_uri, cluster, model, out, names, cat_idx)
    done.append(Stages.EVAL)
    return {"stages": done, "metrics": table, "artifact": str(adir / f"{cluster}_metrics.json")}


def _log_mlflow(uri: str, cluster: str, model: HazardModel, out: dict,
                names: list[str], cat_idx: list[int]) -> None:
    import mlflow

    from mrc.config import N_BINS

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("mrc")
    with mlflow.start_run(run_name=f"{cluster}-gbdt-{N_BINS}bins"):
        mlflow.log_params({k: v for k, v in model.params.items() if not isinstance(v, bool)})
        mlflow.log_params({"cluster": cluster, "bins": out["bins"], "features": len(names),
                           "rows": out["rows"], "eligible": out["eligible"]})
        mlflow.log_metrics({f"{k}_{m}": v for k, row in out["metrics"].items()
                            for m, v in row.items() if isinstance(v, (int, float))})
        mlflow.log_metric("ibs", out["ibs"])
        mlflow.log_dict(out, "curves.json")
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` → `1 passed`.

Note: `np.trapezoid`/`np.trapz` and `np.percentile` on an empty selection will raise; if `lab["eligible"]` is empty on the toy CSV, raise a clear `ValueError("no reuse-eligible request in this window")` instead.

**Step 5 — commit**

```bash
git add src/mrc/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): run_one with per-stage caching and MLflow logging"
```

#### Task F2: MLflow logging is idempotent and observable

**Files:** `tests/test_pipeline.py` (append).

```python
def test_mlflow_run_has_params_and_metrics(tmp_path):
    import mlflow
    src = _tiny_csv(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_one("tiny", source=src, processed=tmp_path / "p", artifacts=tmp_path / "a",
            mlflow_uri=uri, limit=4000, bins=12, num_boost_round=20, n_capacities=8)
    mlflow.set_tracking_uri(uri)
    df = mlflow.search_runs(experiment_names=["mrc"])
    assert len(df) == 1
    assert "params_cluster" in df.columns and df.iloc[0]["params_cluster"] == "tiny"
    assert any(c.startswith("metrics_gbdt_mae") for c in df.columns)
```

```bash
env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q     # expected: 2 passed
git add tests/test_pipeline.py
git commit -m "test(pipeline): MLflow run carries cluster and MAE"
```

#### Task F3: Optuna on cached artefacts

**Files:** create `src/mrc/tuning/__init__.py`, `src/mrc/tuning/search.py`; delete `src/mrc/models/search.py`; test `tests/tuning/test_search.py`.

**Step 1 — failing test**

```python
# tests/tuning/test_search.py
import optuna
from mrc.tuning.search import objective_factory, make_study


def test_study_runs_and_persists(tmp_path):
    seen = []

    def score(params):
        seen.append(params["num_leaves"])
        return float(params["num_leaves"]) / 1000.0

    study = make_study(storage=f"sqlite:///{tmp_path / 's.db'}", name="unit")
    study.optimize(objective_factory(score), n_trials=3)
    assert len(study.trials) == 3
    assert study.best_value == min(t.best_value for t in study.trials if t.value is not None)
    reloaded = optuna.load_study(study_name="unit", storage=f"sqlite:///{tmp_path / 's.db'}")
    assert len(reloaded.trials) == 3
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.tuning'`.

**Step 3 — implement**

```python
# src/mrc/tuning/search.py
"""Optuna study over hazard hyperparameters, scored on cached artefacts."""
from __future__ import annotations

import optuna

from mrc.config import N_BINS_SEARCH, SEED


def suggest(trial: optuna.Trial) -> dict:
    n_bins = trial.suggest_categorical("n_bins", list(N_BINS_SEARCH))
    return {
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_threads": 12, "seed": SEED, "deterministic": True, "force_row_wise": True,
        "_n_bins": n_bins,
    }


def objective_factory(score_fn):
    """Turn score(params) -> float into an Optuna objective with pruning."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        try:
            value = score_fn(params)
        except MemoryError as exc:
            raise optuna.TrialPruned(str(exc)) from exc
        trial.set_user_attr("n_bins", params["_n_bins"])
        return float(value)
    return objective


def make_study(storage: str, name: str = "hazard") -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=name, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/tuning/test_search.py -q` → `1 passed`.

**Step 5 — remove the superseded module**

```bash
git rm src/mrc/models/search.py
env -u PYTHONPATH uv run pytest -q      # expected: all pass, nothing imports models.search
```

**Step 6 — commit**

```bash
git add src/mrc/tuning/ tests/tuning/ src/mrc/models/search.py
git commit -m "feat(tuning): Optuna study over cached artefacts, replaces models/search.py"
```

#### Task F4: the `tune` command and the make gates

**Files:** modify `src/mrc/cli.py`, `Makefile`.

**Step 1 — CLI command**

```python
@main.command()
@click.argument("cluster")
@click.option("--trials", type=int, default=60)
@click.option("--timeout", type=int, default=5400)
def tune(cluster: str, trials: int, timeout: int) -> None:
    """Tune hazard hyperparameters on one cached cluster; score = validation MAE."""
    from mrc.config import MLFLOW_URI, OPTUNA_URI, PROCESSED
    from mrc.data.labels import load_labels
    from mrc.features import build_features
    from mrc.models.hazard import HazardModel, bin_edges
    from mrc.models.risk_set import grouped_rows
    from mrc.pipeline import _params
    from mrc.tuning.search import make_study, objective_factory

    lab = load_labels(PROCESSED / cluster / "labels.npz")
    X, names, _ = build_features(lab)
    truth = np.load(PROCESSED / cluster / "truth.npz")
    caps, mrc_true = truth["capacities"], truth["mrc"]
    p_comp = float((~lab["eligible"]).mean())
    cache: dict[int, dict] = {}

    def score(params: dict) -> float:
        n_bins = params.pop("_n_bins")
        edges = bin_edges(int(np.percentile(lab["rtilde"][lab["eligible"]], 99.9)), n_bins)
        if n_bins not in cache:
            cache[n_bins] = grouped_rows(X, lab["rtilde"], lab["delta"], edges, X.shape[1])
        model = HazardModel(params=params, nbins=len(edges) - 1)
        model.fit_grouped(cache[n_bins], edges, names, num_boost_round=400)
        curve = model.predict_mrc(X, caps, edges, p_compulsory=p_comp)
        return float(np.mean(np.abs(curve - mrc_true)))

    study = make_study(storage=OPTUNA_URI, name="hazard")
    study.optimize(objective_factory(score), n_trials=trials, timeout=timeout,
                   show_progress_bar=False)
    click.echo(f"best mae {study.best_value:.6f} params {study.best_params}")
```

**Step 2 — make targets** (add to `.PHONY`: `labels truth train tune eval figures tables check`)

```make
train:  ## fit one cluster end to end: make train C=cluster45
	$(PY) python -m mrc.cli train "$(C)"

tune:  ## Optuna sweep on a cached cluster: make tune C=cluster45
	$(PY) python -m mrc.cli tune "$(C)"

check:  ## lint + type + tests, the only gate that matters
	$(PY) ruff check src tests && $(PY) mypy src && $(PY) pytest -q
```

and a `train` command that calls `run_one`:

```python
@main.command()
@click.argument("cluster")
@click.option("--source", type=click.Path(exists=True), default=None)
@click.option("--limit", type=int, default=None)
@click.option("--bins", type=int, default=N_BINS)
def train(cluster: str, source: str | None, limit: int | None, bins: int) -> None:
    """Run labels -> truth -> features -> fit -> eval for one cluster."""
    from mrc.config import MLFLOW_URI, RAW
    from mrc.pipeline import run_one

    src = source or (RAW / f"{cluster}.sort.zst")
    res = run_one(cluster, source=src, mlflow_uri=MLFLOW_URI, limit=limit, bins=bins)
    click.echo(json.dumps(res, indent=2))
```

**Step 3 — verify on the sample**

```bash
env -u PYTHONPATH uv run python -m mrc.cli train sample45 \
  --source data/raw/cache-trace/samples/2020Mar/cluster045 --limit 1000000
```

Expected: JSON with `"stages": ["labels","truth","features","fit","eval"]` and four metric dicts, each `mae` between 0 and 1; wall clock ≤ 4 minutes.

**Step 4 — verify the gate**

```bash
make check
```

Expected: ruff silent, mypy `Success: no issues found`, pytest all pass.

**Step 5 — commit**

```bash
git add src/mrc/cli.py Makefile
git commit -m "feat(cli): train and tune commands, make check gate"
```

---

### Phase G — interpretability (the "explainable results" half)

#### Task G1: permutation importance on the validation block

**Files:** create `src/mrc/interp/__init__.py`, `src/mrc/interp/importance.py`; test `tests/interp/test_importance.py`.

**Step 1 — failing test**

```python
# tests/interp/test_importance.py
import numpy as np
from mrc.interp.importance import permutation_importance


def test_importance_ranks_the_informative_column_first():
    rng = np.random.default_rng(10)
    n = 4000
    signal = rng.normal(size=n).astype(np.float32)
    noise = rng.normal(size=(n, 3)).astype(np.float32)
    X = np.column_stack([signal, noise])
    y = (signal > 0).astype(np.float32)

    def score(Xv):
        # a fixed, monotone-in-signal scoring rule; no model needed for the unit test
        return float(np.mean((Xv[:, 0] > 0).astype(np.float32) == y))

    imp = permutation_importance(score, X, names=["signal", "a", "b", "c"],
                                n_repeats=5, seed=0)
    assert imp[0]["feature"] == "signal"
    assert imp[0]["drop"] > 0.2
    assert len(imp) == 4
    assert all("std" in row for row in imp)
```

**Step 2 — verify failure:** `ModuleNotFoundError: No module named 'mrc.interp'`.

**Step 3 — implement**

```python
"""Permutation importance: how much a scoring rule loses when a column is shuffled.

Shuffling a column destroys its association with the target while preserving its
marginal distribution, so the drop in score is attributable to that column alone.
Computed on the validation block only; the test block is never touched.
"""
from __future__ import annotations

import numpy as np


def permutation_importance(score, X: np.ndarray, names: list[str],
                           n_repeats: int = 5, seed: int = 0) -> list[dict]:
    """Return rows {feature, drop, std} sorted by mean drop, descending."""
    rng = np.random.default_rng(seed)
    base = float(score(X))
    rows = []
    for j, name in enumerate(names):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(X.shape[0]), j]
            drops[r] = base - float(score(Xp))
        rows.append({"feature": name, "drop": float(drops.mean()), "std": float(drops.std())})
    return sorted(rows, key=lambda row: row["drop"], reverse=True)
```

**Step 4 — verify pass:** `env -u PYTHONPATH uv run pytest tests/interp/test_importance.py -q` → `1 passed`.

**Step 5 — commit**

```bash
git add src/mrc/interp/ tests/interp/test_importance.py
git commit -m "feat(interp): permutation importance on the validation block"
```

#### Task G2: SHAP contributions and the hazard-by-feature table

**Files:** modify `src/mrc/interp/importance.py`; test appended.

**Step 1 — failing test**

```python
def test_shap_top_table_is_written(tmp_path):
    import json
    import lightgbm as lgb
    from mrc.interp.importance import shap_summary

    rng = np.random.default_rng(11)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (X[:, 0] + 0.2 * rng.normal(size=3000) > 0).astype(np.float32)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 2},
                        lgb.Dataset(X, label=y), num_boost_round=30)
    out = shap_summary(booster, X, names=["a", "b", "c", "d"], top=2,
                       path=tmp_path / "shap.json")
    meta = json.loads(out.read_text())
    assert meta["top"][0]["feature"] == "a"
    assert len(meta["top"]) == 2
```

**Step 2 — implement**

```python
def shap_summary(booster, X: np.ndarray, names: list[str], top: int = 10, path=None):
    """Mean |contribution| per feature from LightGBM's pred_contrib, plus a json dump."""
    import json
    from pathlib import Path

    contrib = booster.predict(np.asarray(X, dtype=np.float32), pred_contrib=True)
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)     # last column is the bias
    order = np.argsort(mean_abs)[::-1][:top]
    rows = [{"feature": names[j], "mean_abs_contribution": float(mean_abs[j])} for j in order]
    if path is not None:
        Path(path).write_text(json.dumps({"top": rows, "n_rows": int(X.shape[0])}, indent=2) + "\n")
        return Path(path)
    return rows
```

**Step 3 — verify pass:** `env -u PYTHONPATH uv run pytest tests/interp -q` → `2 passed`.

**Step 4 — commit**

```bash
git add src/mrc/interp/importance.py tests/interp/test_importance.py
git commit -m "feat(interp): SHAP mean-absolute contributions with json dump"
```

#### Task G3: feature-group ablation matrix

**Files:** modify `src/mrc/pipeline.py`; test appended to `tests/test_pipeline.py`.

**Objective:** prove each feature family earns its place, per skeleton §8.3.

```python
GROUPS = {
    "all": (),
    "no_namespace": ("ns_code",),
    "no_sizes": ("log_obj_size", "log_key_size", "log_val_size"),
    "no_timing": ("log_ttl", "has_ttl", "log_gap", "log_age"),
    "no_popularity": ("log_freq",),
    "no_client": ("client_bucket",),
}


def ablate(X: np.ndarray, names: list[str], groups=GROUPS) -> dict[str, np.ndarray]:
    """Return {group: X with the dropped columns removed}."""
    idx = {n: i for i, n in enumerate(names)}
    return {g: np.delete(X, [idx[c] for c in drop], axis=1) for g, drop in groups.items()}
```

Test: assert `ablate(X, names)["all"].shape == X.shape`, `["no_sizes"].shape[1] == X.shape[1] - 3`, and that every group's column count equals `len(names) - len(dropped)`.

```bash
env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q     # expected: 3 passed
git add src/mrc/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): feature-group ablation matrix"
```

---

### Phase H — figures, tables, report

#### Task H1: the MRC figure

**Files:** create `src/mrc/plots/curve.py`; test `tests/plots/test_curve.py`.

```python
import numpy as np
from mrc.plots.curve import plot_mrc


def test_plot_mrc_writes_pdf(tmp_path):
    caps = np.array([1, 10, 100, 1000])
    truth = np.array([1.0, 0.8, 0.4, 0.2])
    curves = {"Mattson (truth)": truth, "Hazard GBDT": truth + 0.02}
    path = plot_mrc(caps, curves, tmp_path / "mrc.pdf", ylim=(0.0, 1.05))
    assert path.exists() and path.stat().st_size > 1000
    assert path.suffix == ".pdf"
```

```python
# src/mrc/plots/curve.py
"""MRC comparison figure under the frozen style contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mrc.plots.style import DOUBLE_COL, apply_style, label_mrc


def plot_mrc(capacities, curves: dict[str, np.ndarray], path, ylim=None):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.4))
    for label, curve in curves.items():
        ax.plot(capacities, curve, label=label)
    label_mrc(ax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
```

```bash
env -u PYTHONPATH uv run pytest tests/plots -q     # expected: 1 passed
git add src/mrc/plots/curve.py tests/plots/
git commit -m "feat(plots): MRC comparison figure under the style contract"
```

#### Task H2: reliability diagram

`src/mrc/plots/reliability.py` with `plot_reliability(pred, y, path, n_bins=12)` drawing predicted vs observed hit frequency with the diagonal; test asserts a PDF exists and that binning returns `n_bins` fractions summing to 1. Commit `feat(plots): reliability diagram for the hazard model`.

#### Task H3: tables, markdown and booktabs

**Files:** create `src/mrc/report_tables.py`; test `tests/test_report_tables.py`.

```python
from mrc.report_tables import metrics_table, booktabs


def test_metrics_table_has_one_row_per_model():
    metrics = {"gbdt": {"mae": 0.01, "max_abs": 0.03, "bias": 0.0, "mre": 0.02},
               "km": {"mae": 0.05, "max_abs": 0.11, "bias": 0.01, "mre": 0.09}}
    md = metrics_table("cluster45", metrics)
    assert md.count("\n|") == 3                  # header rule + 2 rows
    tex = booktabs(metrics, caption="T1 error", label="tab:t1")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "gbdt" in tex
```

Implement `metrics_table(cluster, metrics) -> markdown str` (columns: model, MAE, MAX, bias, MRE, and a `n` column) and `booktabs(metrics, caption, label) -> str` writing `report/tables/t1_metrics.tex` when given a path.

```bash
env -u PYTHONPATH uv run pytest tests/test_report_tables.py -q   # expected: 1 passed
git add src/mrc/report_tables.py tests/test_report_tables.py
git commit -m "feat(tables): markdown and booktabs metric tables"
```

#### Task H4: fill the report, slides and wiki from generated artefacts

**Files:** modify `report/sections/05-experiments.tex`, `slides/sections/06-results.tex`, `wiki/docs/results.md`; create `scripts/fill_results.py`.

Rules:
1. Every number in those files must come from `artifacts/tables/<cluster>_metrics.json`. No hand-typed numbers. The script writes the tables and the sentence templates; a human reads the diff.
2. Remove every `PLACEHOLDER` / `\figplaceholder` occurrence.
3. `report/tables/*.tex` is generated; never edit it by hand.

Verification gates, exact:

```bash
grep -rn "PLACEHOLDER\|figplaceholder\|pending" report/sections slides/sections wiki/docs/results.md
# expected: no output

cd report && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && pdfinfo main.pdf | grep Pages
# expected: Pages: between 8 and 10 for the body (appendix after References may extend the PDF;
#           count body pages from main.toc, not the PDF)

cd ../slides && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && pdfinfo main.pdf | grep Pages
# expected: Pages: 18

cd ../wiki && env -u PYTHONPATH uv run --group viz mkdocs build --strict 2>&1 | tail -2
# expected: "Documentation built in ... seconds"
```

Commit: `docs: fill results from artifacts, remove every placeholder`.

---

## 5. Test and validation matrix

| Layer | Gate | Command | Expected |
|---|---|---|---|
| Units | reference equality | `uv run pytest tests/data tests/models tests/eval tests/interp -q` | all pass |
| Exact-first | fast path == slow path | `tests/data/test_labels_bound.py`, `tests/gt/test_stackdist.py` | `array_equal`, `atol=0.0` |
| End-to-end | one cluster runs | `uv run python -m mrc.cli train sample45 --limit 1000000 ...` | 4 metric dicts, ≤4 min |
| Telemetry | run is inspectable | `uv run mlflow runs list -x` | one run, `params_cluster=sample45` |
| Tuning | study persists | `uv run python -m mrc.cli tune sample45 --trials 5` | `best mae <initial`, `study.db` non-empty |
| Report | no placeholders | `grep -rn "PLACEHOLDER" report/sections` | no output |
| Style | contract | `tests/test_smoke.py::test_style_contract` | passes |

Numeric acceptance thresholds (put on the Kanban cards, not in prose):

| Metric | Threshold | Meaning |
|---|---|---|
| `gbdt.mae` on test block | ≤ 0.02 | average miss-ratio displacement under 2 points |
| `gbdt.mae` vs `km.mae` | strictly lower on ≥3 of 4 clusters | the features earn their place |
| `gbdt.mae` vs `hitcount.mae` | lower | the hazard form beats the practice it argues against |
| `max_abs` at any capacity with truth ≥ 0.01 | ≤ 0.10 | no local blow-up where the operator operates |
| `bias` | \|bias\| ≤ 0.01 | not systematically optimistic or pessimistic |
| Reuse recompute | bit-identical to `naive_reuse` | the Fenwick path is not drifting |

---

## 6. Risks, tradeoffs, open questions

| Risk | Mechanism | Mitigation in this plan |
|---|---|---|
| 5 M-request window truncates long-distance reuse | distances above the window are unobservable, so MRC at large C is underestimated | overflow bin + identification guard (D5/D6); report the window length in every caption; `SCALE_REQUESTS = 20 M` run on cluster52 as a sensitivity check |
| Grouped hazard is an approximation | grouping assumes a constant hazard within a bin; bias `O(w·h²)` | per-step ablation on 200 k requests quantifies the difference and is a reported table row, not a footnote |
| Censoring is not independent of reuse | end-of-window requests are systematically the ones whose reuse is far away | `Γ` enters the risk set (D1/D2); state the assumption explicitly; do not claim unbiasedness, claim reduced bias |
| Optuna overfits the validation block | 60 trials on one block | test block touched once; report the chosen params on all four clusters; fixed seed |
| Memory blow-up on cluster52 | 20 M requests × grouped rows | `MAX_GROUPED_ROWS` guard raises with the projected count instead of OOM-killing the box |
| `libCacheSim` API drift | single-request constructor may need different arguments | bounded 10-minute attempt, then documented failure; ground truth does not depend on it |
| Byte-MRC is not exact | objects larger than the cache break the stack-distance argument | object-capacity curves are the primary result; byte curves labelled approximate with the measured gap |
| Figure style drift | ad-hoc `rcParams` in a plotting script | every plot imports `apply_style()`; `test_style_contract` gates it |

Open questions for the planner agent to resolve **before** dispatching Phase H (not blocking Phases A–G):

1. Does the course want the byte-capacity curve as a headline result or an appendix? It changes which figure is primary.
2. Is cluster52's 20 M-request run required, or is the 5 M window on four clusters enough for the robustness claim?
3. Should the report lead with the interpretability finding (which features drive reuse) or with the accuracy result? That decides the figure order and the abstract's first sentence.

---

## 7. Planner-agent charter (how the planning bot drives this to completion)

The planner (`mrc-plan`) does not write pipeline code. It owns decomposition, contract freezes, gate definitions, and the reasoning artefacts. Its job, in order:

### 7.1 Produce and maintain four reasoning artefacts

| Artefact | Path | Update trigger |
|---|---|---|
| Decision log | `logs/DECISIONS.md` | any choice that a future reader would otherwise re-litigate: window length, bin count, grid, metric primacy. One line each: `date · decision · evidence · reversible?` |
| Gate registry | `docs/GATES.md` | one row per acceptance command: name, exact command, expected output, owner profile |
| Card specs | Kanban board `mrc` | one card per task in this plan, body = the task text verbatim plus its gate |
| Open-question list | `docs/OPEN_QUESTIONS.md` | the three questions in §6, each with the default assumed if unanswered |

### 7.2 Reasoning sequence the planner must complete before dispatching anything

1. **Confirm the target quantity.** Re-read `STATISTICAL_SKELETON.md` §4.1 and write, in `logs/DECISIONS.md`, the one-line definition of the primary metric: *validation-block MAE of `MRC(C)` against Mattson stack distance, on the frozen 24-point grid.*
2. **Confirm the identification limit.** Compute `b_{M-1}` for one cluster from its label manifest and record it; every figure caption must state it.
3. **Confirm the cost model.** From the measured 3.73 grouped rows per trace request (§1.2), project rows and memory for 5 M and 20 M requests; if either exceeds `MAX_GROUPED_ROWS`, halve the window and record why.
4. **Freeze the capacity grid** on one cluster and assert it transfers to the other three (all four must satisfy `max G < b_{M-1}`).
5. **Decide the ablation list** (Phase G3 groups) before any tuning run, so the ablation is designed, not retrofitted.
6. **Write the card set** in dependency order and verify with `hermes kanban --board mrc list` that no card depends on an archived parent (this exact defect was found and fixed once already).
7. **Only then** dispatch: `mrc-data` for Phases A–D, `mrc-run` for E–G, `mrc-write` for H.

### 7.3 Dispatch order and gates

```
A1 A2 A3 A4  →  B2 B3  →  C1 C2 C3  →  D1 D2 D3 D4 D5
                                              ├→ E1 E2 E3
                                              ├→ F1 F2 → F3 F4
                                              └→ G1 G2 G3
                                                        └→ H1 H2 H3 H4
B1 (libCacheSim cross-check) is off the critical path: it may run any time and may fail without blocking.
```

Dependency edges to encode on the board: `A4 → B3`, `A4 → C1`, `C1 → D1`, `D1 → D4`, `D4 → F1`, `F1 → F3`, `F1 → G1`, `E1 → F1`, `G3 → H3`, `H1 → H4`.

### 7.4 What the planner refuses to do

- **No numbers in a card body before they exist.** "MAE should be about 0.01" is banned; "MAE ≤ 0.02 or the card is not done" is a gate.
- **No card without a runnable gate.** If the exit criterion cannot be a command, the card is a question, not a task.
- **No re-planning of a frozen contract without a `logs/WORKLOG.md` line** naming what changed and which artefacts are now stale.
- **No silent retries.** A failed worker run is reported with its exact error, then re-dispatched once with the error in context; two failures on the same card escalate to a human.

### 7.5 Definition of done for the whole project

All of these must hold simultaneously:

```bash
make check                                                        # 0
env -u PYTHONPATH uv run python -m mrc.cli train cluster45        # writes artifacts/tables/cluster45_metrics.json
env -u PYTHONPATH uv run python -m mrc.cli tune cluster45 --trials 60   # study.db has 60 trials
grep -rn "PLACEHOLDER" report/sections                            # no output
cd report && latexmk -pdf main.tex && pdfinfo main.pdf | grep Pages     # body 8-10 pp
cd ../wiki && mkdocs build --strict                                # 0
git status --porcelain                                            # clean
```
