# Per-cluster accuracy

## cluster45 (test block)

| model | MAE | MAX | bias | MRE | n |
|---|---|---|---|---|---|
| gbdt | 0.0003 | 0.0011 | -0.0002 | 0.0005 | 24 |
| km | 0.0200 | 0.0343 | -0.0164 | 0.0356 | 24 |
| hitcount | 0.1097 | 0.1984 | -0.0734 | 0.1838 | 24 |

*cluster45: `artifacts/tables/cluster45_metrics.json`*

## cluster26 (test block)

| model | MAE | MAX | bias | MRE | n |
|---|---|---|---|---|---|
| gbdt | 0.0055 | 0.0206 | -0.0054 | 0.0106 | 24 |
| km | 0.0118 | 0.0154 | -0.0118 | 0.0201 | 24 |
| hitcount | 0.0793 | 0.2705 | 0.0481 | 0.1477 | 24 |

*cluster26: `artifacts/tables/cluster26_metrics.json`*

## cluster10 (test block)

| model | MAE | MAX | bias | MRE | n |
|---|---|---|---|---|---|
| gbdt | 0.0077 | 0.0190 | -0.0077 | 0.0127 | 12 |
| km | 0.0439 | 0.1204 | -0.0439 | 0.0623 | 12 |
| hitcount | 0.0793 | 0.3321 | -0.0685 | 0.1112 | 12 |

*cluster10: `artifacts/tables/cluster10_metrics.json`*
