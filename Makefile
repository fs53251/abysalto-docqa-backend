.PHONY: install run test lint format check clean redis-start redis-stop

install:
	poetry install

run: redis-start
	poetry run uvicorn app.main:app --reload

test: redis-start
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run black .

check:
	./scripts/precommit_check.sh

clean:
	./scripts/cleanup.sh

redis-start:
	@echo "==> Checking Redis..."
	@if redis-cli ping > /dev/null 2>&1; then \
		echo "Redis already running"; \
	else \
		echo "Starting Redis..."; \
		redis-server --daemonize yes; \
	fi

redis-stop:
	@echo "Stopping Redis..."
	@redis-cli shutdown || true