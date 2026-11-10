# Hill & Smith 1989 — Evaluating Associativity in CPU Caches

| Field | Value |
|---|---|
| Authors | Mark D. Hill, Alan Jay Smith |
| Title | Evaluating Associativity in CPU Caches |
| Venue | IEEE Transactions on Computers, vol. 38, no. 12, pp. 1612–1630 |
| Year | 1989 |
| DOI | 10.1109/12.40842 |
| URL fetched | https://api.crossref.org/works/10.1109/12.40842 |
| Verified | 2026-11-10 |
| What I saw | Title, both authors, `issued` 1989, container "IEEE Transactions on Computers", volume 38, issue 12, pages 1612–1630 |
| BibTeX key | `hill1989evaluating` |

## What it did

Asked what it means to evaluate a cache design fairly, and answered with the
three-C miss classification (compulsory, capacity, conflict) plus a careful
treatment of how simulation methodology, trace length and warm-up distort measured
miss ratios. It is the paper that turned cache evaluation into a protocol rather
than a single simulation, and it makes explicit that a miss ratio without a stated
capacity, policy and measurement window is not a result.

## The one idea this project borrows

The compulsory/capacity split, and with it the discipline of reporting against a
named baseline. In this project the split is literal: $\kappa$ is the compulsory
floor, and everything the model predicts is the capacity-miss component on top of
it. The measurement-window warning is why splits are time-ordered and why window
length is reported ([Experiments](../experiments.md)).

## What this project does differently

Hill and Smith evaluate associativity for a fixed capacity, taking the trace as
given. This project treats the whole capacity axis as the output and asks whether
the axis can be predicted rather than swept. It also replaces their empirical
classification with a parametric survival model, so the "capacity miss" component
is a fitted distribution rather than a subtraction of two simulations.

## Where it lands in this code

`src/mrc/mrc.py` ($\kappa$ computation and the decomposition), `src/mrc/splits.py`
(window handling).
