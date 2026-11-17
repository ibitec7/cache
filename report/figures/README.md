# Figures

Every figure in the report is produced by one module and written into this
directory. There is no second copy: what the producer writes is what LaTeX reads.

| Item | Value |
|---|---|
| Producer | `src/mrc/plots/report_figures.py` (`make_all`) |
| Command | `env -u PYTHONPATH uv run python -m mrc.cli plots` |
| Output directory | `report/figures/` (`REPORT_FIG_DIR` in the producer) |
| Format | PDF, vector, style set only by `src/mrc/plots/style.py` |
| Inputs | `artifacts/tables/<cluster>_metrics.json`, `artifacts/tables/<cluster>_shap.json` |

## Files and where they are used

| File | Content | Included by |
|---|---|---|
| `mrc-curves.pdf` | exact and estimated miss-ratio curves, one panel per trace | `sections/05-experiments.tex` (`fig:mrc-curves`) |
| `cluster-error.pdf` | absolute curve error against capacity, log axes | `sections/05-experiments.tex` (`fig:cluster-error`) |
| `capacity-ratio.pdf` | capacity ratio (model / truth) at 60, 80 and 95 % of the ceiling | `sections/05-experiments.tex` (`fig:capacity`) |
| `importance.pdf` | mean absolute SHAP contribution, top six features per trace | `sections/05-experiments.tex` (`fig:importance`) |

`report/main.tex` sets `\graphicspath{{figures/}}`, so the sections reference the
files as `figures/<name>.pdf` with no path into `artifacts/`. The four PDFs are
generated artefacts that are committed with the report, because the report must
compile from a fresh checkout without re-running training.

## Regeneration

```sh
cd /home/ibrahim/uni/cache
env -u PYTHONPATH uv run python -m mrc.cli plots
```

The command rewrites all four PDFs in place and prints each path it wrote. It
also refreshes the tables under `report/tables/` from the same metrics files, so
a figure and the table beside it can never come from different runs.

If a metrics file is missing the command fails with
`FileNotFoundError: no metrics artefacts`, rather than emitting a partial set.
The sections include the PDFs with plain `\includegraphics`, so a missing figure
is a LaTeX error at compile time. No placeholder box exists.

## Adding a figure

Add a plotting function to `src/mrc/plots/report_figures.py` that takes
`path: Path | None = None` and defaults to `REPORT_FIG_DIR / "<name>.pdf"`, call
it from `make_all`, and include `figures/<name>.pdf` from the report. Do not
write to a second directory and do not copy PDFs by hand: a copy drifts from the
producer and the report then shows a figure no command regenerates.

`.gitkeep` is kept so the directory exists in a fresh checkout before the first
run of the command above.
