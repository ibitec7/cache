# Slides

Beamer deck, 16:9, `metropolis` theme, same title and author as the report.

## Compile

```sh
cd /home/ibrahim/uni/cache/slides
latexmk -pdf -interaction=nonstopmode main.tex
```

Output: `main.pdf`.

## Structure

| File | Content |
|------|---------|
| `sections/01-motivation.tex` | gap, over/under-allocation |
| `sections/02-reuse-distance.tex` | reuse distance, LRU identification |
| `sections/03-hazard-model.tex` | discrete hazard, parametrizations |
| `sections/04-ground-truth.tex` | simulation reference, metrics |
| `sections/05-protocol.tex` | time split, capacity grid, search |
| `sections/06-results.tex` | result placeholders |
| `sections/07-limitations.tex` | limitations, falsifier |
| `sections/08-takeaway.tex` | takeaway, closing frame |

`\section{...}` lives inside the first frame file of each group, so
`\tableofcontents` on the outline frame resolves. If `metropolis` is missing on
another machine, swap `\usetheme{metropolis}` for `\usetheme{default}` in
`main.tex`; the frames use no metropolis-specific macros except the `standout`
frame option, which `default` also accepts.

## Clean

```sh
latexmk -C
```
