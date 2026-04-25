.PHONY: sync test lint fmt check verify-probe verify-edge notebooks-execute paper run-e1 run-e2 run-e3 run-all run-e4 run-e5 run-e6 run-e7 run-e8 run-conway

sync:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

check: lint test
	uv run mypy src tests

verify-probe:
	uv run qf-verify-probe

verify-edge-fano:
	uv run qf-verify-edge-fano

notebooks-execute:
	uv run jupyter nbconvert --execute --inplace papers/sedenion-associator-probe/notebooks/*.ipynb

paper:
	cd papers/sedenion-associator-probe && ./build.sh paper

run-e1:
	uv run qf-run-e1

run-e2:
	uv run qf-run-e2

run-e3:
	uv run qf-run-e3

run-all: run-e1 run-e2 run-e3

run-e4:
	uv run qf-run-e4

run-e5:
	uv run qf-run-e5

run-e6:
	uv run qf-run-e6

run-e7:
	uv run qf-run-e7

run-e8:
	uv run qf-run-e8 --n-workers 4 --seeds 30 --max-n 50

run-conway: run-e4 run-e5 run-e6 run-e7 run-e8
