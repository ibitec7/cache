# Akiba et al. 2019 — Optuna

| Field | Value |
|---|---|
| Authors | Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama |
| Title | Optuna: A Next-generation Hyperparameter Optimization Framework |
| Venue | Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '19), pp. 2623–2631 |
| Year | 2019 |
| DOI | 10.1145/3292500.3330701 |
| URLs fetched | https://arxiv.org/abs/1907.10902 (title, authors, submitted 25 Jul 2019) and https://api.crossref.org/works/10.1145/3292500.3330701 (KDD '19, pages 2623–2631) |
| Verified | 2026-11-10 |
| What I saw | arXiv abstract page with the full title and five authors; Crossref record with the KDD proceedings container and page range |
| BibTeX key | `akiba2019optuna` |

## What it did

Described a hyperparameter search framework built around three design choices:
define-by-run search spaces, so the space is constructed by the code that samples
from it; a lightweight trial-and-error database so trials can be parallelised
across processes; and pruning, which aborts unpromising trials early. The default
sampler is TPE, a Bayesian method that models good and bad configurations
separately and samples where the ratio favours improvement.

## The one idea this project borrows

Pruning plus a persistent trial store. A single GBDT trial on this corpus takes
long enough that an exhaustive grid is impossible, and many trials are clearly bad
by the first few hundred trees. Early pruning plus a SQLite study database means
the search can be stopped and resumed across sessions, and the tuning history
becomes an artifact (`artifacts/optuna/study.db`) rather than a lost session.

## What this project does differently

The search space is domain-constrained rather than generic: `num_leaves` and
`min_data_in_leaf` are bounded using the corpus size, because a leaf-wise tree with
255 leaves on a shard with few thousand episodes per bucket overfits immediately.
The objective is validation MAPE on the curve, a functional, not validation
logloss, so the search is judged on what the project actually reports.

## Where it lands in this code

`src/mrc/search.py` (study definition, TPE sampler, median pruner, objective on
validation MAPE), `configs/search/` (space definition).
