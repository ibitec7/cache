SHELL := /bin/bash
PY := env -u PYTHONPATH uv run
export MLFLOW_TRACKING_URI := sqlite:///$(CURDIR)/artifacts/mlflow/mlflow.db

.PHONY: help sync data-samples data-t1 data-t2 inspect store shard reuse mrc fit labels features truth train tune interpret analysis plots report slides wiki ui smoke test lint check

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

sync:  ## install / refresh the environment
	uv sync --all-groups

data-samples:  ## T0: clone the 54 sample traces
	bash scripts/fetch_traces.sh samples

data-t1:  ## T1: three smallest clusters
	bash scripts/fetch_traces.sh t1

data-t2:  ## T2: cluster52 full shard (93.09 GB, resumable, 8 parallel ranges)
	bash scripts/fetch_traces.sh t2

inspect:  ## summarise a trace file: make inspect F=data/raw/cache-trace/samples/2020Mar/cluster045
	$(PY) python -m mrc.cli inspect "$(F)"

store:  ## convert a shard to zstd parquet: make store F=... D=data/processed/store52
	$(PY) python -m mrc.cli store "$(F)" --out "$(D)"

shard:  ## run one store window: make shard C=s52_0 S=data/processed/store52 I=0 N=5000000
	$(PY) python -m mrc.cli train "$(C)" --store "$(S)" --start "$(I)" --window "$(N)" --no-keep-cache

reuse:  ## compute reuse distance for a trace: make reuse F=... OUT=data/interim/x.npz
	$(PY) python -m mrc.cli reuse "$(F)" --out "$(OUT)"

mrc:  ## empirical MRC curve: make mrc R=data/interim/x.npz
	$(PY) python -m mrc.cli mrc "$(R)"

fit:  ## fit the hazard model under MLflow: make fit R=data/interim/x.parquet
	$(PY) python -m mrc.cli fit "$(R)"

labels:  ## cache labels: make labels C=cluster45
	$(PY) python -m mrc.cli labels "$(or $(F),$(shell python -c 'from mrc.cli import default_source; print(default_source("$(C)"))'))" --cluster "$(C)"

features:  ## build the design matrix: make features C=cluster45
	$(PY) python -m mrc.cli features "$(C)"

truth:  ## ground-truth curve: make truth C=cluster45
	$(PY) python -m mrc.cli truth "$(C)"

train:  ## full run + MLflow: make train C=cluster45
	$(PY) python -m mrc.cli train "$(C)"

tune:  ## Optuna search: make tune C=cluster45 T=20
	$(PY) python -m mrc.cli tune "$(C)" --trials "$(or $(T),20)"

interpret:  ## permutation, SHAP and ablation: make interpret C=cluster45
	$(PY) python -m mrc.cli interpret "$(C)"

analysis:  ## capacity-provisioning table
	$(PY) python -m mrc.cli analysis

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
	$(PY) ruff check src tests scripts && $(PY) mypy src

check:  ## the full gate: ruff, mypy, pytest
	$(PY) ruff check src tests scripts
	$(PY) mypy src
	$(PY) pytest -q
