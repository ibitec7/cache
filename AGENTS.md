# Agent rules

Canonical repo rules for `/home/ibrahim/uni/cache`. Read before touching anything.

## 1. Environment

| Fact | Value |
|---|---|
| Interpreter | uv-managed, python 3.13, `.venv` in repo root |
| Run anything | `env -u PYTHONPATH uv run ...` |
| Why unset PYTHONPATH | Hermes injects a python 3.14 site-packages path that shadows pydantic and breaks mlflow |
| Add a dependency | `uv add <pkg>` or `uv add --group dev <pkg>`, never `pip install` |
| Lock file | `uv.lock` is committed, never hand-edited |

## 2. Reproducibility gates

1. **Exact-first.** Every optimisation reproduces its reference bit-for-bit on the same input before it runs at scale. Reuse distance: Fenwick kernel vs naive `O(n * window)`. MRC: curve vs LRU replay. Tolerance `0.0`, not "close enough".
2. **One alternate shape.** Every claim is re-checked on a second workload shape (other cluster, other bin count) before it is reported.
3. **No proxy metrics.** Compare the actual quantity on identical inputs, never a model's error against a different configuration's error.
4. **Numbers come from artifacts**, never typed by hand.

## 3. Figures

`src/mrc/plots/style.py` is the only place that sets rcParams.

| Rule | Value |
|---|---|
| Font | serif, `Nimbus Roman` (metric-identical to Times New Roman, which is not installed here) |
| Maths in figures | STIX, matching `newtxmath` in the report |
| Size | 3.35 in single column, 7.0 in double, PDF at 300 dpi |
| Grid | none, top and right spines off |
| Annotations | at most one per panel, no arrows unless they carry the argument |
| Legend | at most 4 entries, no frame |
| Titles | none inside a figure; the caption states the claim |
| Axis labels | short plain words, unit in parentheses only when needed |
| Colour | fixed palette, same series keeps the same colour everywhere |

## 4. Writing

- Main body 8 to 10 pages, then References, then Appendix (unlimited).
- Report preamble inherited from the course template; `newtxtext,newtxmath` only to match the figures.
- No em-dashes, no AI-isms, no marketing adjectives.
- Every equation is followed by a symbol table defining **every** symbol with domain and units.
- Every metric: what it measures, why it is right here, what failing it means.
- Tables over prose whenever the content is enumerable.
- Hedges stay hedges ("at par" never becomes "better"). Retract measured artifacts in the log instead of quietly dropping them.

## 5. Data

- Raw traces in `data/raw/`, never committed or uploaded.
- Acquisition only through `scripts/fetch_traces.sh`, bounded by tier, resumable.
- Scraping uses Scrapling and records `url`, `fetched_at`, `sha256` in a manifest.

## 6. Code

- Layout: `data/` read + reuse distance + reference, `models/` hazard + search, `eval/` curves + metrics, `plots/` style + figures, `cli.py`.
- Each data-touching module states its input contract in the docstring.
- Exact-first gates are mandatory in tests for anything on the critical path.
- Type hints everywhere, ruff line length 100.
- Append to `logs/WORKLOG.md`: UTC time, commit, what changed, what worked, what failed and why.

## 7. Multi-agent workflow

Board `mrc`, four profiles, file-level ownership.

| Profile | Owns | Never touches |
|---|---|---|
| `mrc-plan` | statistical skeleton, experiment matrix, capacity grid, acceptance criteria | `src/` |
| `mrc-data` | `src/mrc/data/`, `scripts/`, `data/` | `report/`, `slides/` |
| `mrc-run` | `src/mrc/models/`, `src/mrc/eval/`, `artifacts/` | `report/` |
| `mrc-write` | `src/mrc/plots/`, `report/`, `slides/`, `wiki/`, `README.md` | `src/mrc/data/` |

Frozen contracts: npz schema (`docs/DATA.md`), `style.py`, `report/bibliography/refs.bib`.
A worker needing a contract change stops and asks.
