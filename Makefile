.PHONY: install test lint typecheck check run

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check rav tests

typecheck:
	python -m mypy rav

check: lint typecheck test

run:
	python -m rav
