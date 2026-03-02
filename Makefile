.PHONY: help install format lint test

help:
	@echo "Targets:"
	@echo "  install  - install deps via poetry"
	@echo "  format   - format code (black)"
	@echo "  lint     - lint (ruff)"
	@echo "  test     - run pytest"

install:
	poetry install

format:
	poetry run black .

lint:
	poetry run ruff check .

test:
	poetry run pytest -q