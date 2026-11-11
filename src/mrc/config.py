"""Frozen run constants. One source of truth; nothing else may hardcode these."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw" / "traces"
PROCESSED = DATA / "processed"
ARTIFACTS = REPO / "artifacts"
MLFLOW_URI = f"sqlite:///{ARTIFACTS / 'mlflow' / 'mlflow.db'}"
OPTUNA_URI = f"sqlite:///{ARTIFACTS / 'optuna' / 'study.db'}"

LABEL_REQUESTS = 5_000_000      # prefix per cluster for labels / features / truth
SCALE_REQUESTS = 20_000_000     # cluster52 robustness run only
CAP_QUANTILE = 1023             # rtilde cap == b_{M-1} - 1 <= 1023
N_BINS = 32
N_BINS_SEARCH = (8, 16, 24, 32, 48, 64)
N_CAPACITIES = 24

SPLIT_TRAIN = 0.70              # contiguous request blocks
SPLIT_VALID = 0.85              # valid [0.70, 0.85), test [0.85, 1.0)

MAX_GROUPED_ROWS = 60_000_000
PER_STEP_MAX_REQUESTS = 200_000

SEED = 7
NUM_THREADS = 12
