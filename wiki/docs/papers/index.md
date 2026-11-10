# Papers

The verified reading list. Every entry was fetched and checked for title, authors,
year and venue before it was written here; the exact URL fetched is on each page.
Sources that could not be verified are excluded and listed in
`docs/LITERATURE.md`.

The canonical BibTeX lives in `report/bibliography/refs.bib` and is maintained by
the paper worker. `wiki/references.bib` is a mirror for this site only; if the two
disagree, the report copy wins.

## MRC and stack distance

| Source | What this project takes |
|---|---|
| [Mattson et al. 1970](mattson-1970.md) | the stack algorithm: one pass, all capacities |
| [Hill & Smith 1989](hill-smith-1989.md) | how to compare curve estimators fairly |
| [Wires et al. 2014](wires-2014-counterstacks.md) | sublinear-space exactness via counters |
| [Waldspurger et al. 2015](waldspurger-2015-shards.md) | sampled estimation at bounded memory |
| [Beckmann & Sanchez 2015](beckmann-2015-talus.md) | why the curve's cliffs matter to a policy |
| [Yang et al. 2023](yang-2023-s3fifo.md) | the policy the reference curves are drawn for |

## Traces and tooling

| Source | What this project takes |
|---|---|
| [Yang et al. 2020](yang-2020-twitter.md) | the dataset, and its TTL and write-traffic caveats |
| [libCacheSim](libcachesim.md) | the ground-truth simulator |

## Survival analysis

| Source | What this project takes |
|---|---|
| [Kaplan & Meier 1958](kaplan-meier-1958.md) | product-limit estimator under censoring |
| [Cox 1972](cox-1972.md) | covariates in the hazard, the idea the GBDT replaces |
| [Discrete-time survival](discrete-time-survival.md) | the bucket formulation that admits any classifier |
| [Right-censoring](right-censoring.md) | why never-reused keys cannot be dropped |

## GBDT and search

| Source | What this project takes |
|---|---|
| [Ke et al. 2017](ke-2017-lightgbm.md) | the hazard classifier implementation |
| [Akiba et al. 2019](akiba-2019-optuna.md) | TPE search over the hyperparameters |

## How to cite

BibTeX keys follow `authorYEARkeyword`, for example `mattson1970evaluation`.
Each paper page gives the key to use.
