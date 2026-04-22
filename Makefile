.PHONY: sync test lint fmt check verify-probe verify-edge notebooks-execute paper

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
