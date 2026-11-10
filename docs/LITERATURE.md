# Literature

Every entry below was fetched on 2026-11-10 and checked for title, author list,
year and venue. The `verification method` column records what was actually
fetched, not what was intended. No DOI, page range or author list on this page
comes from memory: each is copied from the fetched record.

Canonical BibTeX: `report/bibliography/refs.bib` (maintained by the paper worker).
`wiki/references.bib` is a mirror for the wiki site only.

## MRC and stack distance

| key | title | venue/year | what it gives us | doi/arxiv | url | verified-on | verification method |
|---|---|---|---|---|---|---|---|
| `mattson1970evaluation` | Evaluation Techniques for Storage Hierarchies | IBM Systems Journal 9(2):78-117, 1970 | The stack algorithm: one pass, all capacities, $m(C)=\frac{1}{N}\sum \mathbf{1}\{d_i \ge C\}$ | 10.1147/sj.92.0078 | https://api.crossref.org/works/10.1147/sj.92.0078 | 2026-11-10 | Fetched the Crossref registration record for the DOI; saw the title, the four authors (Mattson, Gecsei, Slutz, Traiger), issued 1970, container IBM Systems Journal, vol 9, issue 2, pages 78-117 |
| `hill1989evaluating` | Evaluating Associativity in CPU Caches | IEEE Transactions on Computers 38(12):1612-1630, 1989 | Three-C miss decomposition and evaluation protocol; the compulsory floor and the fair-baseline discipline | 10.1109/12.40842 | https://api.crossref.org/works/10.1109/12.40842 | 2026-11-10 | Fetched the Crossref record; saw title, Hill and Smith, issued 1989, IEEE Transactions on Computers, 38(12), pages 1612-1630 |
| `wires2014characterizing` | Characterizing Storage Workloads with Counter Stacks | OSDI '14, 2014 | Counter-stack representation: sublinear-space summaries from which curves at new capacities follow without replay | none (USENIX) | https://www.usenix.org/conference/osdi14/technical-sessions/presentation/wires | 2026-11-10 | Fetched the USENIX OSDI '14 programme page; saw title, five authors (Wires, Ingram, Drudi, Harvey, Warfield, Coho Data), session slot 2014-08-07, abstract. Venue is OSDI '14, not FAST '14 as the reading list said |
| `waldspurger2015efficient` | Efficient MRC Construction with SHARDS | FAST '15, 2015 | Sampled reuse distance at bounded memory; the cost baseline a learned model must beat, plus the bias/variance argument | none (USENIX) | https://www.usenix.org/conference/fast15/technical-sessions/presentation/waldspurger | 2026-11-10 | Fetched the USENIX FAST '15 programme page; saw title, four authors (Waldspurger, Park, Garthwaite, Ahmad, CloudPhysics), abstract describing uniform randomized spatial sampling |
| `beckmann2015talus` | Talus: A Simple Way to Remove Cliffs in Cache Performance | HPCA 2015, pp. 64-75 | Why curve cliffs make average error the wrong metric; motivates MaxErr and per-capacity residuals | 10.1109/HPCA.2015.7056022 | https://api.crossref.org/works/10.1109/hpca.2015.7056022 | 2026-11-10 | Fetched the Crossref record; saw title, Beckmann and Sanchez, issued 2015-02, container "2015 IEEE 21st International Symposium on High Performance Computer Architecture", pages 64-75. Venue is HPCA, not ISCA as the reading list said |
| `yang2023fifo` | FIFO queues are all you need for cache eviction | SOSP '23, pp. 130-149, 2023 | S3-FIFO as the policy reference; comparing estimators at matched metadata cost rather than matched capacity | 10.1145/3600006.3613147 | https://api.crossref.org/works/10.1145/3600006.3613147 | 2026-11-10 | Fetched the Crossref record; saw title, five authors (Yang, Zhang, Qiu, Yue, Vinayak), container "Proceedings of the 29th Symposium on Operating Systems Principles", pages 130-149, issued 2023-10-23. ACM DL returned 403 to a direct fetch, so the registration record is the evidence |

## Survival analysis

| key | title | venue/year | what it gives us | doi/arxiv | url | verified-on | verification method |
|---|---|---|---|---|---|---|---|
| `kaplan1958nonparametric` | Nonparametric Estimation from Incomplete Observations | JASA 53(282):457-481, 1958 | Product-limit estimator; the censoring-aware survival curve, and the per-bucket factorisation the classifier exploits | 10.1080/01621459.1958.10501452 | https://api.crossref.org/works/10.1080/01621459.1958.10501452 | 2026-11-10 | Fetched the Crossref record; saw title, Kaplan and Meier, issued 1958-06, JASA 53(282), pages 457-481 |
| `cox1972regression` | Regression Models and Life-Tables | JRSS B 34(2):187-202, 1972 | Covariates act on the hazard; gives a curve per profile instead of per dataset | 10.1111/j.2517-6161.1972.tb00899.x | https://api.crossref.org/works/10.1111/j.2517-6161.1972.tb00899.x | 2026-11-10 | Fetched the Crossref record; saw title, D. R. Cox, issued 1972, Journal of the Royal Statistical Society Series B, 34(2), pages 187-202 |
| `allison1982discrete` | Discrete-Time Methods for the Analysis of Event Histories | Sociological Methodology 13:61, 1982 | The bucket formulation: survival estimation as a sequence of binary decisions, which is what admits any classifier | 10.2307/270718 | https://api.crossref.org/works/10.2307/270718 | 2026-11-10 | Fetched the Crossref record; saw title, Allison, issued 1982, Sociological Methodology vol 13, first page 61. Only the first page is registered, so no page range is claimed here |

## GBDT and hyperparameter search

| key | title | venue/year | what it gives us | doi/arxiv | url | verified-on | verification method |
|---|---|---|---|---|---|---|---|
| `ke2017lightgbm` | LightGBM: A Highly Efficient Gradient Boosting Decision Tree | NeurIPS 30 (NIPS 2017), 2017 | The hazard classifier: native categoricals, binary objective with per-bucket probabilities, fast enough for hundreds of Optuna trials | none (NeurIPS proceedings) | https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html | 2026-11-10 | Fetched the NeurIPS proceedings abstract page; saw title, eight authors (Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu), "Advances in Neural Information Processing Systems 30 (NIPS 2017)", abstract mentioning GOSS and EFB |
| `akiba2019optuna` | Optuna: A Next-generation Hyperparameter Optimization Framework | KDD '19, pp. 2623-2631, 2019 | TPE search, pruning and a resumable trial store; tuning history as an artifact | 10.1145/3292500.3330701 (arXiv 1907.10902) | https://arxiv.org/abs/1907.10902 + https://api.crossref.org/works/10.1145/3292500.3330701 | 2026-11-10 | Fetched the arXiv abstract page (full title, five authors, submitted 25 Jul 2019) and the Crossref record for the KDD DOI (container "Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining", pages 2623-2631) |

## Trace datasets and tooling

| key | title | venue/year | what it gives us | doi/arxiv | url | verified-on | verification method |
|---|---|---|---|---|---|---|---|
| `yang2020large` | A large scale analysis of hundreds of in-memory cache clusters at Twitter | OSDI 20, 2020 | The dataset and its caveats: TTL-driven expiry, write traffic, cluster taxonomy | none (USENIX) | https://www.usenix.org/conference/osdi20/presentation/yang | 2026-11-10 | Fetched the USENIX OSDI 20 programme page; saw title, three authors with affiliations (Yang, CMU; Yue, Twitter; Rashmi, CMU). Dataset facts (54 clusters, March 2020, one-week traces, 2.8 TB compressed / 14 TB uncompressed, column schema) read from the raw README of https://github.com/twitter/cache-trace |
| `libcachesim` | libCacheSim: a high performance library for building cache simulators | software, current release | The ground-truth simulator and the policy reference curves | none (software) | https://github.com/1a1a11a/libCacheSim + https://libcachesim.readthedocs.io/en/latest/ | 2026-11-10 | Fetched the GitHub repository page (title as quoted) and the Read the Docs landing page ("Welcome to libCacheSim's documentation!"). `libcachesim.com` resolves to the GitHub repo, so no separate project site is cited |

## Venue corrections

Two entries in the reading list named the wrong venue. Both are included, with
the venue the source itself states. Recorded here so the change is auditable
rather than silent.

| claimed | actual | evidence |
|---|---|---|
| Wires et al., FAST 2014 | OSDI '14 | USENIX OSDI '14 programme page, session 2014-08-07 |
| Beckmann & Sanchez, ISCA 2015 | HPCA 2015 | Crossref record for 10.1109/HPCA.2015.7056022 |

## Not verified, excluded

Nothing from the reading list was dropped. Every requested item resolved to a
fetched record. Items considered and rejected on grounds other than fetch
failure are not listed here because none were needed.

Two cautions for the paper worker, both about what the fetched pages do **not**
contain:

| item | caution |
|---|---|
| Wires et al. 2014 | USENIX page gives no page numbers and no DOI assignment usable in BibTeX; `wires2014characterizing` therefore carries neither |
| libCacheSim | no DOI and no formal citation record exists; cite as software with the URL and access date, do not invent a venue |

Counts: 13 verified sources, 0 excluded.
