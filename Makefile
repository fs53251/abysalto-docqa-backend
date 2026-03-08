.PHONY: install run run-ui test lint format check clean redis-start redis-stop docker-up docker-up-build docker-down docker-logs docker-ps

install:
	poetry install

run: redis-start
	poetry run uvicorn app.main:app --reload

run-ui:
	poetry run streamlit run ui/streamlit_app.py

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

docker-up:
	docker compose up -d

docker-up-build:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api ui db redis

docker-ps:
	docker compose ps
