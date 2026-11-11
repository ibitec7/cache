# Phase H: figures, tables and report

**Phase:** H. **Owner profile:** `mrc-write`. **Defects closed:** D5's last part (no figures, no tables, no
filled report).
**Upstream:** `07-phase-g-interpretability.md` (edge 9, `G3 -> H3`), `06-phase-f-pipeline.md` (the metric
json every number comes from), `02-phase-b-truth.md` (the truth curve H1 plots).
**Downstream:** none. This phase produces the delivered artefacts.

**Estimated wall time:** 4 tasks at 2 to 5 minutes each (master §4), plus three document builds
(`latexmk` twice, `mkdocs` once) and the editorial pass to fit the body into 8 to 10 pages.
**Compute:** one process, no GPU. `matplotlib` renders at 300 dpi PDF; the builds are I/O bound.

Baseline measured at audit, before this phase starts: `pdfinfo report/main.pdf` -> `Pages: 13`, and
`grep -rn "PLACEHOLDER\|figplaceholder" report/sections` -> 5 hits. The body target is 8 to 10 pages, so
H4 is an editorial task, not only a find-and-replace.

---

## 1. Input and output contract

### Inputs

| Input | Path | Provenance |
|---|---|---|
| metric table | `artifacts/tables/<cluster>_metrics.json` | F1; schema in `06-phase-f-pipeline.md` §1 |
| ablation table | `artifacts/tables/<cluster>_ablation.json` | G3 |
| importance, SHAP | `artifacts/tables/<cluster>_{importance,shap}.json` | G1, G2 |
| truth curve | `data/processed/<cluster>/truth.npz` | B3 |
| style contract | `src/mrc/plots/style.py` | frozen, `SINGLE_COL=3.35`, `DOUBLE_COL=7.0` |
| report skeleton | `report/main.tex`, `report/sections/*.tex` | exists, 13 pages, 5 placeholders |
| slide skeleton | `slides/main.tex`, `slides/sections/*.tex` | exists |
| wiki | `wiki/docs/results.md` | exists |

### Outputs

| Artefact | Path | Invariant |
|---|---|---|
| MRC figure | `artifacts/figures/mrc_<cluster>.pdf` | PDF, over 1000 bytes, produced through `apply_style()`; no title; at most 4 legend entries |
| reliability figure | `artifacts/figures/reliability_<cluster>.pdf` | PDF; diagonal drawn; bin fractions sum to 1 |
| booktabs tables | `report/tables/*.tex` | generated only by `scripts/fill_results.py`, first line a `% generated` comment |
| filled report | `report/sections/*.tex` | zero placeholders; every number from the metric json |
| filled slides | `slides/sections/06-results.tex` | zero placeholders; `Pages: 18` |
| filled wiki | `wiki/docs/results.md` | zero placeholders; `mkdocs build --strict` exit 0 |

Invariants:

- `I-H1` every figure is created through `mrc.plots.style` helpers; no file outside `style.py` sets an
  `rcParam`. `tests/test_smoke.py::test_style_contract` must stay green.
- `I-H2` `grep -rn "PLACEHOLDER\|figplaceholder\|pending"` over `report/sections`, `slides/sections` and
  `wiki/docs/results.md` returns nothing.
- `I-H3` the only writer of `report/tables/*.tex` is `scripts/fill_results.py`; a hand edit is a defect.
- `I-H4` every numeric literal in the filled sections appears in some `artifacts/tables/*_metrics.json`,
  either directly or as a value the script formatted. The reviewer checks this by diff, and the script
  marks its regions with `% BEGIN GENERATED` / `% END GENERATED` so the diff is reviewable.
- `I-H5` body length is 8 to 10 pages, counted from `report/main.toc` (page of References minus page of
  the first section), not from `pdfinfo`, which counts the appendix.
- `I-H6` every caption that shows an MRC curve states the window length (`limit` from the metric json) and
  `b_{M-1}` (the capacity ceiling of the grid).

---

## 2. Module interface contract

```python
# src/mrc/plots/curve.py
def plot_mrc(capacities, curves: dict[str, np.ndarray], path, ylim=None) -> Path:
    """MRC comparison figure under the frozen style contract; one line per entry of `curves`."""
```

Callers: `tests/plots/test_curve.py` and `scripts/fill_results.py`.

```python
# src/mrc/plots/reliability.py
def plot_reliability(pred, y, path, n_bins: int = 12) -> Path:
    """Predicted vs observed hit frequency with the diagonal, under the frozen style contract."""
```

Callers: `tests/plots/test_reliability.py` and `scripts/fill_results.py`.

```python
# src/mrc/report_tables.py
def metrics_table(cluster: str, metrics: dict) -> str:
    """Markdown table with one row per model: model, MAE, MAX, bias, MRE, n."""

def booktabs(metrics: dict, caption: str, label: str, path=None) -> str:
    """LaTeX booktabs table; writes `path` when given, always returns the string."""
```

Callers: `tests/test_report_tables.py` and `scripts/fill_results.py`.

```python
# scripts/fill_results.py
def main(clusters: list[str], artifacts: Path, report: Path, slides: Path, wiki: Path) -> int:
    """Rewrite the generated regions of the report, slides and wiki from the metric json."""
```

Owner: `mrc-write`. `scripts/` is otherwise `mrc-data`'s glob; this single file is the recorded exception
(`00-index.md` §5).

Non-goals: no hand-typed number anywhere, no edit to generated `report/tables/*.tex`, no edit to
`style.py`, no edit to `report/bibliography/refs.bib`.

---

## 3. Ordered tasks

### H1: the MRC figure

**Purpose:** the one figure that carries the accuracy claim.
**Files:** create `src/mrc/plots/curve.py`; create `tests/plots/test_curve.py` (and `tests/plots/__init__.py`
if the package needs one).
**Depends on:** F4 (a metric json to draw from), B3 (the truth curve).

**TDD steps**

1. Failing test, as master H1 gives it. Add one assertion on the legend size, because the contract caps it
   at four entries:

```python
def test_plot_mrc_writes_pdf_and_respects_the_legend_cap(tmp_path):
    import numpy as np
    from mrc.plots.curve import plot_mrc
    caps = np.array([1, 10, 100, 1000])
    truth = np.array([1.0, 0.8, 0.4, 0.2])
    curves = {"Mattson (truth)": truth, "Hazard GBDT": truth + 0.02}
    path = plot_mrc(caps, curves, tmp_path / "mrc.pdf", ylim=(0.0, 1.05))
    assert path.exists() and path.stat().st_size > 1000
    assert path.suffix == ".pdf"
    with __import__("pytest").raises(ValueError):
        plot_mrc(caps, {f"c{i}": truth for i in range(5)}, tmp_path / "bad.pdf")
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/plots -q` -> `ModuleNotFoundError`.
3. Implement. Use `fig_double(2.4)` rather than `plt.subplots(figsize=(DOUBLE_COL, 2.4))`, so the size
   comes from the same helper every other figure uses, and raise `ValueError` when `len(curves) > 4`
   (`I-H1`). No title; `label_mrc(ax)` sets the axes.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/plots -q` -> `2 passed`.
5. Render the real figure from the `sample45` metric json plus `truth.npz` and confirm the caption numbers
   it will carry (`limit`, `b_M_minus_1`).

**Gate:** `G-H1-mrc` (`2 passed`).
**Commit:** `feat(plots): MRC comparison figure under the style contract`
**Rollback:** `git revert <sha>`; no document embeds it yet.
**Done when:** the PDF exists, the legend cap is enforced, and no `rcParam` is set outside `style.py`.

---

### H2: reliability diagram

**Purpose:** show whether the hazard's probabilities are calibrated, not only whether the curve is close.
**Files:** create `src/mrc/plots/reliability.py`; create `tests/plots/test_reliability.py`.
**Depends on:** H1.

**TDD steps**

1. Failing test:

```python
def test_reliability_bins_and_writes_pdf(tmp_path):
    import numpy as np
    from mrc.plots.reliability import bin_fractions, plot_reliability
    rng = np.random.default_rng(0)
    pred = rng.random(1000)
    y = (rng.random(1000) < pred).astype(np.float32)
    frac = bin_fractions(pred, 12)
    assert len(frac) == 12
    assert abs(float(frac.sum()) - 1.0) < 1e-12
    path = plot_reliability(pred, y, tmp_path / "rel.pdf", n_bins=12)
    assert path.exists() and path.stat().st_size > 1000
```

2. Verify failure: `ModuleNotFoundError: No module named 'mrc.plots.reliability'`.
3. Implement `bin_fractions(pred, n_bins) -> np.ndarray` (public, so the fraction invariant is testable)
   and `plot_reliability` using `fig_single` and `apply_style()`, with the diagonal as a reference line and
   at most one annotation.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/plots -q` -> `3 passed`.

**Gate:** `G-H2-reliability` (`3 passed`).
**Commit:** `feat(plots): reliability diagram for the hazard model`
**Rollback:** `git revert <sha>`; the MRC figure stands alone.
**Done when:** the bin fractions sum to 1 and the figure is produced from real validation predictions.

---

### H3: tables, markdown and booktabs

**Purpose:** every table in the report is generated, so no number is typed.
**Files:** create `src/mrc/report_tables.py`, `tests/test_report_tables.py`.
**Depends on:** G3 (edge 9), E3.

**TDD steps**

1. Failing test, as master H3 gives it. Note the `md.count("\n|") == 3` assertion means the markdown
   string must start with the header row and not with a newline: header, separator, two rows give three
   `\n|` occurrences. Keep that layout.
2. Verify failure: `ModuleNotFoundError: No module named 'mrc.report_tables'`.
3. Implement `metrics_table` (columns `model, MAE, MAX, bias, MRE, n`) and `booktabs` (columns as the
   report needs, `\toprule`/`\midrule`/`\bottomrule`, caption and label, a `% generated` first line, and a
   `path` argument that writes the file).
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/test_report_tables.py -q` -> `1 passed`.
5. Generate the real tables from `sample45` and one other cluster, and diff them against the metric json by
   eye in the card comment.

**Gate:** `G-H3-tables` (`1 passed`).
**Commit:** `feat(tables): markdown and booktabs metric tables`
**Rollback:** `git revert <sha>`; `report/tables/` keeps whatever was generated last.
**Done when:** both formats are produced from the json, and the `.tex` carries the generated header.

---

### H4: fill the report, slides and wiki from generated artefacts

**Purpose:** the delivered documents, with zero placeholders and no hand-typed numbers.
**Files:** create `scripts/fill_results.py`; modify `report/sections/05-experiments.tex`,
`slides/sections/06-results.tex`, `wiki/docs/results.md`.
**Depends on:** H1 (edge 10), H2, H3.

**Rules, from the master and from the frozen contracts**

1. Every number comes from `artifacts/tables/<cluster>_metrics.json`. The script writes the tables and the
   sentence templates; a human reads the diff.
2. Remove every `PLACEHOLDER` and `\figplaceholder`.
3. `report/tables/*.tex` is generated; never edit it by hand.
4. Generated regions are delimited by `% BEGIN GENERATED` and `% END GENERATED`, so the diff of a rerun is
   exactly the numbers that changed (`I-H4`).
5. Every MRC caption states the window length and `b_{M-1}` (`I-H6`). On cluster045 that is
   `b_{M-1} = 837`, so every plotted capacity is at most 836; a caption that omits this is a defect.

**TDD steps**

1. Failing evidence: `grep -rn "PLACEHOLDER\|figplaceholder\|pending" report/sections slides/sections
   wiki/docs/results.md` -> 5 hits, measured at audit.
2. Implement `scripts/fill_results.py`. It reads the metric json for each cluster named on its command
   line, writes the three tables, and replaces the generated regions in the three documents. It refuses to
   run when a metric key it needs is absent, rather than writing a blank.
3. Verify, in this order:

```bash
grep -rn "PLACEHOLDER\|figplaceholder\|pending" report/sections slides/sections wiki/docs/results.md
# expected: no output

cd report && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && pdfinfo main.pdf | grep Pages
# expected: a Pages: line; body 8 to 10 pages, counted from main.toc

cd ../slides && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && pdfinfo main.pdf | grep Pages
# expected: Pages: 18

cd ../wiki && env -u PYTHONPATH uv run --group viz mkdocs build --strict 2>&1 | tail -2
# expected: "Documentation built in ... seconds"
```

4. Editorial pass: the report body currently measures 13 PDF pages with 5 placeholders. After binding the
   generated tables and figures, cut or move material until `I-H5` holds; derivations move to the appendix,
   which is unlimited. Record the measured page counts before and after in the card comment.

**Gate:** `G-H4-placeholders`, `G-H4-report`, `G-H4-slides`, `G-H4-wiki`.
**Commit:** `docs: fill results from artifacts, remove every placeholder`
**Rollback:** `git revert <sha>`; the generated tables can be regenerated, the placeholders come back, and
the report builds again.
**Done when:** all four gates pass and every table in the body traces to a json file.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-STYLE` | `env -u PYTHONPATH uv run pytest tests/test_smoke.py::test_style_contract -q` | `1 passed` | a plotting module set an `rcParam` outside `style.py` |
| `G-H1-mrc` | `env -u PYTHONPATH uv run pytest tests/plots -q` | `2 passed` | the figure is not written, or the legend cap is absent |
| `G-H2-reliability` | `env -u PYTHONPATH uv run pytest tests/plots -q` | `3 passed` | the binning does not partition the predictions |
| `G-H3-tables` | `env -u PYTHONPATH uv run pytest tests/test_report_tables.py -q` | `1 passed` | a table row or column is missing, or the markdown layout changed |
| `G-H4-placeholders` | `grep -rn "PLACEHOLDER\|figplaceholder\|pending" report/sections slides/sections wiki/docs/results.md` | no output | a section still has an unfilled slot |
| `G-H4-report` | `cd report && latexmk -pdf -interaction=nonstopmode main.tex && pdfinfo main.pdf \| grep Pages` | a `Pages:` line; body 8 to 10 from `main.toc` | the build failed, or the body length is outside 8 to 10 |
| `G-H4-slides` | `cd slides && latexmk -pdf -interaction=nonstopmode main.tex && pdfinfo main.pdf \| grep Pages` | `Pages: 18` | the deck length changed without a decision |
| `G-H4-wiki` | `cd wiki && env -u PYTHONPATH uv run --group viz mkdocs build --strict 2>&1 \| tail -2` | `Documentation built in ... seconds` | a broken link or a missing page under `--strict` |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| A hand-typed number surviving into the report | editing the prose is faster than templating | `I-H3`, `I-H4` and the BEGIN/END markers make the generated region the only place numbers can live; the reviewer diffs the rest |
| Body length creeping past 10 pages | tables and figures add pages after the text is written | the page budget is measured from `main.toc` before and after each generated block, recorded in the card comment |
| Style drift | a figure script sets its own `rcParams` | `G-STYLE` gates every plotting commit; `style.py` is frozen |
| Legend over four entries | the temptation to plot four models plus the truth | `plot_mrc` raises above four; the report shows the truth plus the best three, and the rest go to the appendix |
| Caption omitting the window or the identification limit | easy to leave out | `I-H6`; the numbers are in the metric json, so the template can inject them |
| Slide page count drifting from 18 | results slides added ad hoc | `G-H4-slides`; an extra slide needs a decision and a note in the card |
| `mkdocs --strict` failing on a stale link | the wiki page references an old artefact path | the build is a gate, not advice |
| Editing generated `.tex` | it looks like a normal file | first line is a `% generated` comment, and the reviewer checks `git diff report/tables/` is empty unless the script ran |

---

## 6. Reviewer checklist

- [ ] No file outside `src/mrc/plots/style.py` calls `plt.rcParams` or `matplotlib.rc`.
- [ ] Every figure uses `fig_single`, `fig_double` or `apply_style`.
- [ ] No figure has an internal title; the caption carries the claim.
- [ ] At most four legend entries, one annotation per panel, no grid.
- [ ] Every table file begins with `% generated`.
- [ ] Every number in the body appears in `artifacts/tables/*_metrics.json`.
- [ ] Every MRC caption states `limit` and `b_{M-1}`.
- [ ] `report/tables/` diff is empty when the script did not run.
- [ ] Body page count is 8 to 10 from `main.toc`, and the number is stated with the command that produced it.
- [ ] `git status --porcelain` is clean after the commit, and `data/processed/` and `artifacts/` are not
      tracked.

---

## 7. Handoff note

This phase is terminal. It hands off to the whole-project definition of done in `00-index.md` §10, which
requires all of the following at once:

1. `make check` exits 0.
2. `mrc train cluster45` writes `artifacts/tables/cluster45_metrics.json`.
3. `mrc tune cluster45 --trials 60` leaves 60 trials in `artifacts/optuna/study.db`.
4. `grep -rn "PLACEHOLDER" report/sections` is empty.
5. The report body is 8 to 10 pages.
6. `mkdocs build --strict` exits 0.
7. `git status --porcelain` is empty.

If any of items 1 to 6 passes only on `sample45` and not on `cluster45`, the deliverable is a one-cluster
report and the report must say so in the experiments section, with the reason (time, memory, or a failed
criterion) stated.
