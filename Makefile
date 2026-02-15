.PHONY: install dev test lint format typecheck docs build clean

install:
	uv sync --locked

dev:
	uv sync --locked --all-extras

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check rlm_code tests

format:
	uv run ruff format rlm_code tests

typecheck:
	uv run mypy rlm_code --ignore-missing-imports

docs:
	uv run mkdocs build

build:
	uv build

clean:
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache site
