# Figures

Every figure in this report is generated, never drawn or edited by hand.

- Producer: `src/mrc/plots/` writes PDF (vector) output into `artifacts/figures/`.
- The report includes those files by relative path, for example
  `../artifacts/figures/mrc-curves.pdf`, so a regenerated figure is picked up
  without editing any `.tex` file.
- If a file is missing at compile time, `sections/05-experiments.tex` falls back
  to a labelled placeholder box. The placeholder is not a figure; it marks a
  figure that has not been generated yet.
- Do not commit rasterized or annotated copies here. Add new figures by adding
  a plotting function under `src/mrc/plots/` and referencing its output path.
- Regeneration commands are listed in Appendix `sec:appendix-repro`.

`figures/` is otherwise empty; it exists so `\graphicspath` resolves and so
locally generated diagnostic figures can be dropped in without touching the
report tree.
