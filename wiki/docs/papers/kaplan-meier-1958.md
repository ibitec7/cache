# Kaplan & Meier 1958 — Nonparametric Estimation from Incomplete Observations

| Field | Value |
|---|---|
| Authors | E. L. Kaplan, Paul Meier |
| Title | Nonparametric Estimation from Incomplete Observations |
| Venue | Journal of the American Statistical Association, vol. 53, no. 282, pp. 457–481 |
| Year | 1958 |
| DOI | 10.1080/01621459.1958.10501452 |
| URL fetched | https://api.crossref.org/works/10.1080/01621459.1958.10501452 |
| Verified | 2026-11-10 |
| What I saw | Title, both authors, `issued` 1958-06, container "Journal of the American Statistical Association", volume 53, issue 282, pages 457–481 |
| BibTeX key | `kaplan1958nonparametric` |

## What it did

Derived the product-limit estimator of a survival function from data with
right-censored observations. The estimator multiplies conditional survival
probabilities, $(1 - d_t/n_t)$, over time buckets, using only the subjects still at
risk in each bucket. The key contribution is that censored subjects contribute
information up to the moment they leave, so they are not discarded, and the
estimator remains consistent under independent censoring.

## The one idea this project borrows

The estimator itself, used in two ways: as the nonparametric baseline curve on
[Hazard model](../hazard-model.md), and as the definition of what the GBDT is
approximating. Because the estimator is a product of conditional probabilities, it
decomposes into per-bucket binary decisions, which is exactly what allows a
classifier to replace it.

## What this project does differently

Kaplan-Meier has no covariates: one curve per population. The reuse distance in a
cache cluster depends strongly on namespace and operation type, so a single curve
mixes populations that behave differently. This project keeps the product-limit
structure but makes each factor conditional on a feature vector, giving a curve per
request profile rather than per cluster.

## Where it lands in this code

`src/mrc/hazard.py` (`kaplan_meier`, risk-set construction).
