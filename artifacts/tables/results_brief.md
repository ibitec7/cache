# Results brief (generated)

## Clusters

| cluster | requests | keys | mean obj B | eligible | p_c | R p50 | p90 | p99 | p99.9 | bins | grid | b_M-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster45 | 5000000 | 2158200 | 91 | 0.568 | 0.432 | 4 | 208853 | 1008186 | 1725692 | 33 | 24 | 1110405 |
| cluster26 | 5000000 | 394747 | 1046 | 0.921 | 0.079 | 472 | 7938 | 17709 | 51990 | 33 | 24 | 37455 |
| cluster10 | 5000000 | 2500051 | 1289 | 0.500 | 0.500 | 1 | 4 | 8 | 14 | 14 | 12 | 13 |

## Test-block error

| cluster | model | MAE | MAX | bias | MRE | ABC |
|---|---|---|---|---|---|---|
| cluster45 | gbdt | 0.00027 | 0.00107 | -0.00017 | 0.0005 | 0.0003 |
| cluster45 | km | 0.02000 | 0.03429 | -0.01645 | 0.0356 | 0.0202 |
| cluster45 | hitcount | 0.10968 | 0.19838 | -0.07341 | 0.1838 | 0.1139 |
| cluster26 | gbdt | 0.00548 | 0.02057 | -0.00539 | 0.0106 | 0.0053 |
| cluster26 | km | 0.01178 | 0.01537 | -0.01178 | 0.0201 | 0.0119 |
| cluster26 | hitcount | 0.07927 | 0.27053 | +0.04805 | 0.1477 | 0.0799 |
| cluster10 | gbdt | 0.00772 | 0.01902 | -0.00772 | 0.0127 | 0.0092 |
| cluster10 | km | 0.04387 | 0.12042 | -0.04387 | 0.0623 | 0.0713 |
| cluster10 | hitcount | 0.07930 | 0.33206 | -0.06852 | 0.1112 | 0.1418 |

## Provisioning (capacity ratio model/truth)

| cluster | model | 60% | 80% | 95% |
|---|---|---|---|---|
| cluster45 | gbdt | 1.003 | 0.976 | 1.004 |
| cluster45 | km | 0.922 | 0.239 | 0.524 |
| cluster26 | gbdt | 0.977 | 0.981 | 0.999 |
| cluster26 | km | 0.971 | 0.985 | 0.996 |
| cluster10 | gbdt | 0.947 | 0.935 | 0.925 |
| cluster10 | km | 0.635 | 0.681 | 0.744 |

## Interpretability (SHAP mean|contribution|, top 5)

- **cluster45**: log_gap=2.66, hazard_step=1.42, log_age=0.12, op=0.02, log_freq=0.02
- **cluster26**: log_gap=1.32, hazard_step=1.03, log_age=0.10, log_freq=0.03, log_ttl=0.02
- **cluster10**: log_gap=3.29, hazard_step=3.29, log_obj_size=0.08, log_age=0.07, log_val_size=0.02

## Ablation (validation MAE, subsample)

| cluster | all | no_timing | no_sizes | no_popularity | no_client | no_namespace |
|---|---|---|---|---|---|---|
| cluster45 | 0.00051 | 0.01557 | 0.00052 | 0.00040 | 0.00048 | 0.00048 |
| cluster26 | 0.02303 | 0.02098 | 0.02166 | 0.02198 | 0.02138 | 0.02180 |
| cluster10 | 0.00509 | 0.02365 | 0.00572 | 0.00493 | 0.00498 | 0.00490 |
