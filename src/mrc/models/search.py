"""Optuna search over LightGBM hazard hyperparameters, logged to MLflow."""
from __future__ import annotations

import mlflow
import optuna


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_threads": 0,
        "seed": 7,
    }


def make_objective(fit_fn):
    """Wrap a fit(payload) -> objective value callable as an Optuna objective."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            value = fit_fn(params)
            mlflow.log_metric("objective", value)
        return value
    return objective


def study(storage: str = "sqlite:///artifacts/optuna/study.db", name: str = "hazard") -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=name, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=7), load_if_exists=True,
    )
