# Glossary

One line each. Alphabetical. A definition never uses another term from this page
without it being defined here too.

| Term | Definition |
|---|---|
| **Admission policy** | Rule deciding whether a requested object may enter the cache at all. |
| **ARC** | Adaptive replacement policy that balances recency and frequency with four lists. |
| **At risk ($n_t$)** | Number of episodes that have neither had their event nor been censored before bucket $t$. |
| **Bucket** | One discrete interval of reuse distance, indexed $t$, holding lifetime values $t$. |
| **Censoring bucket** | Last bucket in which a censored episode is still counted as at risk. |
| **Client id** | Anonymised identifier of the frontend service that issued a request. |
| **Cold miss** | Miss on the first ever request for a key; unavoidable at any capacity. |
| **Compulsory miss ratio ($\kappa$)** | Fraction of requests that are first accesses; the floor of the MRC. |
| **Counter stack** | Data structure that stores a stack's contents as per-key counters, giving sublinear-space MRCs. |
| **Covariate** | Request metadata used as a predictor in the hazard model. |
| **Cumulative hazard ($\Lambda$)** | Sum (or integral) of the hazard up to $t$; equals $-\log S(t)$. |
| **Discrete-time hazard** | Conditional probability of the event in bucket $t$ given survival to $t$. |
| **Episode** | One request treated as the start of a lifetime running to the next request for the same key. |
| **Event** | An episode whose next same-key request was observed inside the trace window. |
| **Event indicator ($\delta$)** | 1 if the event was observed, 0 if the episode was censored. |
| **Exact stack distance** | Reuse distance computed from the full stack of distinct keys, no sampling. |
| **FIFO** | Eviction policy that removes the oldest inserted object, ignoring reuse. |
| **GBDT** | Gradient boosted decision trees; an additive ensemble fitted stagewise to a loss. |
| **Ground truth MRC** | Miss ratio produced by replaying the trace through a simulator. |
| **Hazard ($h_t$)** | Probability of the event in bucket $t$ among episodes still at risk. |
| **Hit ratio ($H$)** | Fraction of requests served from the cache; $1 - m(C)$. |
| **Inter-arrival gap ($g_i$)** | Time between two consecutive requests for the same key. |
| **Isotonic correction** | Projection of a predicted curve onto the set of non-increasing sequences. |
| **Kaplan-Meier** | Nonparametric product-limit estimator of the survival function under censoring. |
| **Leave-one-cluster-out** | Cross-validation that holds out one entire cluster to test transfer. |
| **libCacheSim** | Open-source cache simulator used here as the ground-truth engine. |
| **Lifetime ($d$)** | Reuse distance of the next access to the same key, the survival time. |
| **LightGBM** | Histogram-based GBDT implementation with leaf-wise growth. |
| **LRU** | Eviction policy that removes the least recently used object; makes reuse distance exact. |
| **MAPE** | Mean over capacities of the absolute relative error of the predicted curve. |
| **MaxErr** | Worst absolute residual of the predicted curve over the capacity grid. |
| **MLflow** | Experiment tracker storing params, metrics and artifacts per run. |
| **MRC** | Miss ratio curve: miss ratio as a function of capacity. |
| **Namespace** | Leading component of an anonymised key identifying the application's key family. |
| **Object size ($s_i$)** | Key size plus value size in bytes. |
| **One-hit wonder** | Key requested exactly once in the window; always a compulsory miss. |
| **Operation** | Cache command type: get, set, delete, incr and similar. |
| **Optuna** | Hyperparameter search framework using TPE sampling and pruning. |
| **Policy** | The eviction rule of the cache; the MRC depends on it. |
| **Product-limit** | Synonym for the Kaplan-Meier step estimator form. |
| **Reuse distance ($d_i$)** | Number of distinct keys requested between two consecutive requests for the same key. |
| **Right-censoring** | The event is known only to occur later than the observation window, not when. |
| **Risk set** | Episodes still under observation at a given bucket. |
| **S3-FIFO** | Three-queue FIFO policy (small, main, ghost) designed for skewed popularity. |
| **SHARDS** | Sampled, hashed approximation of reuse distance with bounded error. |
| **Simulation replay** | Running the trace request by request against a real cache implementation. |
| **Stack algorithm** | One-pass algorithm computing all reuse distances by maintaining an ordered stack. |
| **Stack distance** | Synonym for reuse distance in distinct keys. |
| **Survival function ($S(t)$)** | Probability that the lifetime exceeds $t$. |
| **TPE** | Tree-structured Parzen estimator, the default Optuna sampler. |
| **TTL** | Time-to-live in seconds attached to a write; 0 for non-write requests. |
| **Trace** | Ordered record of cache requests with metadata, one row per request. |
| **Trace window ($T_c$)** | Time span covered by one cluster's collected trace. |
| **Working set** | Set of distinct keys active in a recent interval. |
| **Zipf exponent ($\alpha$)** | Skew parameter of the key popularity distribution. |
