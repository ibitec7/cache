SHELL := /bin/bash
PY := env -u PYTHONPATH uv run
export MLFLOW_TRACKING_URI := sqlite:///$(CURDIR)/artifacts/mlflow/mlflow.db

.PHONY: help sync data-samples data-t1 data-t2 inspect reuse mrc fit plots report slides wiki ui smoke test lint

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

sync:  ## install / refresh the environment
	uv sync --all-groups

data-samples:  ## T0: clone the 54 sample traces
	bash scripts/fetch_traces.sh samples

data-t1:  ## T1: three smallest clusters
	bash scripts/fetch_traces.sh t1

data-t2:  ## T2: cluster52 full shard (87 GB, resumable)
	bash scripts/fetch_traces.sh t2

inspect:  ## summarise a trace file: make inspect F=data/raw/cache-trace/samples/2020Mar/cluster045
	$(PY) python -m mrc.cli inspect "$(F)"

reuse:  ## compute reuse distance for a trace: make reuse F=... OUT=data/interim/x.parquet
	$(PY) python -m mrc.cli reuse "$(F)" --out "$(OUT)"

mrc:  ## empirical MRC curve: make mrc R=data/interim/x.parquet
	$(PY) python -m mrc.cli mrc "$(R)"

fit:  ## fit the hazard model under MLflow: make fit R=data/interim/x.parquet
	$(PY) python -m mrc.cli fit "$(R)"

plots:  ## regenerate every figure from artifacts
	$(PY) python -m mrc.cli plots

report:  ## compile the LaTeX report
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

slides:  ## compile the beamer deck
	cd slides && latexmk -pdf -interaction=nonstopmode main.tex

wiki:  ## build the wiki site
	cd wiki && env -u PYTHONPATH uv run --group viz mkdocs build --strict

ui:  ## MLflow + Optuna dashboards
	$(PY) mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000 &
	$(PY) optuna-dashboard sqlite:///artifacts/optuna/study.db --port 8080

smoke:  ## fast end-to-end check on the smallest sample
	$(PY) pytest -q tests/test_smoke.py

test:  ## full test suite
	$(PY) pytest -q

lint:  ## ruff + mypy
	$(PY) ruff check src tests && $(PY) mypy src
