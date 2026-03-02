.PHONY: help install format lint test db-upgrade db-revision db-stamp

help:
	@echo "Targets:"
	@echo "  install     - install deps via poetry"
	@echo "  format      - format code (black)"
	@echo "  lint        - lint (ruff)"
	@echo "  test        - run pytest"
	@echo "  db-upgrade  - run alembic upgrade head"
	@echo "  db-revision - create new migration: make db-revision m='message'"
	@echo "  db-stamp    - stamp head (useful if you created tables manually)"

install:
	poetry install

format:
	poetry run black .

lint:
	poetry run ruff check .

test:
	poetry run pytest -q

db-upgrade:
	poetry run alembic upgrade head

db-revision:
	poetry run alembic revision --autogenerate -m "$(m)"

db-stamp:
	poetry run alembic stamp head