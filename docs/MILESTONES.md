# Milestones

Parallel by default. Each workstream advances on its own lane against frozen
file contracts; a milestone is a synchronization point only where a contract
changes or an artifact is consumed downstream. Full rationale in
PROJECT_PLAN.md sections 5 and 7.

## Milestone table

| Milestone | Deliverable | Owner profile | Entry criteria | Exit gate | Blocks others |
|---|---|---|---|---|---|
| M0 | Plan, milestones, risks, canonical parquet schema, Makefile gate targets | `mrc-plan` | repo skeleton, dependencies installed | `make check` passes; `make schema-check` validates the fixture parquet against `docs/contracts/requests_schema.json` | yes: schema freeze gates M1 and M3 |
| M1 | Typed parquet for T1 (cluster10, cluster26, cluster45, cluster3) plus manifest with sha256 and row counts | `mrc-data` | M0 schema freeze; disk check `make disk-check` passes | `make data-t1` exits 0 and `make schema-check` passes on every output; row count matches independent DuckDB count within 0 | yes: M2, M3 |
| M2 | Labels (reuse distance, censoring) and exact stack-distance MRC for T1, validated against oracle and LRU simulator | `mrc-data` | M1; schema version pinned | `make truth-t1` exits 0; `pytest tests/data/test_stackdist_oracle.py` passes; max abs deviation from exact LRU sim <= 1e-6 | partially: M3 needs truth only for evaluation, training can start |
| M3 | Fitted hazard model on T1, Optuna study, MLflow runs, metric table versus baseline and Mattson | `mrc-run` | M1 (labels in progress is enough to start feature join); style freeze for figures | `make train-t1` exits 0; `make eval-t1` writes `report/tables/t1_metrics.csv`; every row maps to an MLflow run id | yes: M4 numbers |
| M4 | All figures, paper 8-10 pages, reproducibility check from frozen artifacts | `mrc-write` | M3 metric table; `docs/paper/refs.bib` freeze; figures pipeline green | `make figures` reproduces byte-identical PDFs from cached inputs (hash match); `make paper` builds and page count in [8,10]; `make refs-check` passes | no |

Numbers not yet measured are absent from this table by design. The exit gates
are commands, not opinions: a lane is done when the command exits 0 and its
output is pasted on the Kanban card.

## Integration gates

Three artifacts are frozen once and versioned afterwards. Freezing is a Kanban
card with the command below as its exit gate.

| Gate | Artifact | Frozen by | Command | Unfreeze procedure |
|---|---|---|---|---|
| G1 schema | `docs/contracts/requests_schema.json` | `mrc-plan`, with `mrc-data` sign-off | `make schema-check` | new version file, new card, both consumers re-run their gate |
| G2 style | `src/mrc/plots/style.py` | `mrc-write` | `make style-check` (hash of rcParams block compared to committed hash) | additive-only change, `mrc-write` approves, `make style-check` re-run |
| G3 refs | `docs/paper/refs.bib` | `mrc-write` | `make refs-check` (bib parse; every `\cite` key resolves; every entry cited) | new entries only after M3, key rename requires touching `main.tex` in the same commit |

G1 lands at M0. G2 and G3 land before M4 and may be opened earlier without
blocking anyone, because style and bibliography do not depend on results.
A milestone card that depends on an open gate card cannot move to Done.

## Make targets

The Makefile is a contract owned by `mrc-plan`. These names are fixed; adding
targets is fine, renaming is a breaking change and needs a card.

| Target | Meaning |
|---|---|
| `make check` | ruff, mypy, pytest on the package |
| `make schema-check` | validate parquet files against the frozen schema |
| `make style-check` | verify the frozen rcParams block hash |
| `make refs-check` | bibliography integrity and citation closure |
| `make disk-check` | assert free space on the data volume above the configured floor |
| `make data-t1` | fetch, decode, validate the four T1 clusters |
| `make truth-t1` | stack-distance labels and Mattson MRC for T1 |
| `make train-t1` | fit hazard model and log to MLflow |
| `make eval-t1` | write the T1 metric table |
| `make figures` | regenerate every paper figure from cached artifacts |
| `make paper` | build the PDF and assert the page count |

## Per-milestone detail

### M0 Plan and contracts

Deliverables: PROJECT_PLAN.md, MILESTONES.md, RISKS.md, schema contract,
Makefile with all targets present (targets may be stubs that fail loudly, never
silently succeed), `configs/t1.yaml` and `configs/t2.yaml` with paths and
capacity grids.

Intent: fix the interfaces before any expensive work starts, so the three other
lanes can run without waiting. The schema is the only artifact whose late
change is expensive, because it invalidates decoded parquet and every label
table.

Exit evidence: `make check` output, `make schema-check` on a 1000-row fixture,
schema version string recorded on card G1.

### M1 Ingest

Deliverables: manifest with per-shard sha256, compressed bytes, decoded row
count, min and max timestamp; parquet with canonical dtypes; a decode report
with null counts and bad-row quarantine counts.

Intent: make the raw corpus addressable without re-reading 3 GB of zstd on every
experiment. The manifest is the reproducibility anchor for M4.

Exit evidence: `make data-t1` log, DuckDB row counts side by side with manifest
counts, `make disk-check` before and after.

### M2 Labels and ground truth

Deliverables: per-request reuse distance in distinct-object units, censoring
indicator, gap statistics, person-period or sampled-at-risk training rows, and
the Mattson MRC per cluster.

Intent: separate "what the trace says" from "what the model believes". The
Mattson curve is the measuring stick; it must be correct before it is used to
judge anything.

Exit evidence: oracle test on 1e5 requests, LRU simulator agreement <= 1e-6,
MRC endpoint checks (MRC(0) = 1, MRC(N) = 0), label row counts reconciled with
request counts plus censored tail.

### M3 Model and tuning

Deliverables: hazard model per cluster, pooled model across T1, Optuna study
with trials table, MLflow runs with params, metrics, model artifact, calibration
table, per-bin hazard diagnostics.

Intent: establish whether a feature-conditional hazard reproduces the curve
shape, and where it fails (short distances, TTL-heavy namespaces, write-heavy
operations).

Exit evidence: `make eval-t1` table containing baseline error, model error, knee
error, 90% hit-ratio capacity error for every cluster; MLflow run ids on the
card; seed and config hash recorded.

### M4 Writing and release

Deliverables: figures generated by one command, tables exported from MLflow or
committed CSVs, paper source and PDF, wiki page with reproduction steps.

Intent: no hand-edited numbers. Every value in the paper points at an artifact
that a command regenerates.

Exit evidence: `make figures` hash match, `make paper` page count within
[8,10], `make refs-check` clean, reproduction run on a clean checkout of the
data manifest.

## Coordination rules

- WIP limit is one card per profile. A blocked card returns to Backlog with the
  blocking gate named on it.
- Review is done by a different profile than the author. The reviewer re-runs
  the exit gate rather than reading the diff for intent.
- Contract changes are never bundled with feature work. One card, one gate
  re-run, both consumers notified via the board.
- If a gate command is flaky, fix the command first; a gate that passes
  intermittently is treated as failing.
