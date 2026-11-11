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
CAP_QUANTILE = 2_097_151        # rtilde ceiling, 2^21 - 1. Measured at a 5M prefix:
                                # reuse-distance p90/p99/p99.9 = 209k/1.01M/1.73M on
                                # cluster45 and 7.9k/17.7k/52k on cluster26, so the
                                # master plan's 1023 put most mass in the overflow bin.
                                # Grouped rows cost one row per bin at risk, at most
                                # N_BINS + 1 per request, so a larger ceiling does not
                                # increase the row count.
N_BINS = 32
N_BINS_SEARCH = (8, 16, 24, 32, 48, 64)
N_CAPACITIES = 24

SPLIT_TRAIN = 0.70              # contiguous request blocks
SPLIT_VALID = 0.85              # valid [0.70, 0.85), test [0.85, 1.0)

MAX_GROUPED_ROWS = 60_000_000
PER_STEP_MAX_REQUESTS = 200_000

SEED = 7
NUM_THREADS = 12
