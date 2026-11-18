"""Optuna search over the hazard model, on cached artefacts.

The study optimises validation-block MAE of the predicted MRC curve. The capacity
grid, the bin edges and the grouped rows are built once by `pipeline.prepare`, so a
trial only refits the booster: 20 trials cost minutes, not an evening.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna

from mrc.config import NUM_THREADS, OPTUNA_URI, SEED
from mrc.pipeline import Prepared, fit, predict_block, score, truth_block


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space: capacity and regularisation, no objective-specific knobs."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "num_threads": NUM_THREADS,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
    }


def make_objective(prep: Prepared, num_boost_round: int = 300):
    """Return an Optuna objective that returns validation MAE."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = fit(prep, params, num_boost_round=num_boost_round)
        pred = predict_block(model, prep, "valid")
        return float(score(pred, truth_block(prep, "valid"), prep.capacities)["mae"])

    return objective


def study_name(cluster: str) -> str:
    return f"mrc-{cluster}"


def storage_url(path: str | Path | None = None) -> str:
    if path is None:
        return OPTUNA_URI
    return f"sqlite:///{Path(path)}"


def run_study(prep: Prepared, n_trials: int, *, cluster: str,
              num_boost_round: int = 300, storage: str | None = None,
              timeout: float | None = None, with_mlflow: bool = True) -> optuna.Study:
    """Create or resume the study, optimise validation MAE, and log the best trial."""
    url = storage or storage_url()
    Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name(cluster), storage=url, direction="minimize",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(make_objective(prep, num_boost_round), n_trials=n_trials,
                   timeout=timeout, show_progress_bar=False)
    if with_mlflow:
        import mlflow

        from mrc.config import MLFLOW_URI

        Path(MLFLOW_URI.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("mrc-optuna")
        with mlflow.start_run(run_name=study.study_name):
            mlflow.set_tags({"study": study.study_name, "cluster": cluster})
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"best_valid_mae": float(study.best_value),
                                "n_trials": float(len(study.trials))})
    return study
