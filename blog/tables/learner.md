| parameter | default | tuned range | what it controls |
|---|---|---|---|
| `objective` | `binary` | frozen | binary logloss, one row per (episode, bucket) |
| `metric` | `binary_logloss` | frozen | early stopping on validation logloss |
| `num_leaves` | `63` | 20 to 63 | capacity of one tree, the main regulariser here |
| `learning_rate` | `0.05` | 0.05 to 0.11 | shrinkage per tree |
| `min_data_in_leaf` | `100` | 50 to 100 | minimum rows per leaf, guards rare buckets |
| `feature_fraction` | `0.9` | frozen | column subsample per tree |
| `bagging_fraction` | `0.8` | frozen | row subsample per iteration |
| `bagging_freq` | `1` | frozen | bagging every iteration |
| `lambda_l2` | `1.0` | 0.003 to 1.0 | L2 penalty on leaf values |
| `deterministic` | `True` | frozen | replicate runs are bit-comparable |
| `boosting` | `gbdt` | not tuned | GOSS is available and **not used** |
