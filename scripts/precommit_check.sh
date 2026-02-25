#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Pre-commit check in: $PROJECT_ROOT"

# 1) Cleanup
if [[ -x "./scripts/cleanup.sh" ]]; then
  echo "==> Running cleanup..."
  ./scripts/cleanup.sh
else
  echo "==> cleanup.sh not found or not executable. Running minimal cleanup inline..."

  find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
  find . -type f -name "*.pyc" -delete || true
  rm -rf .pytest_cache .ruff_cache .mypy_cache || true
  rm -f .coverage coverage.xml || true
  rm -rf htmlcov .tox .nox || true
  rm -rf data/uploads/* data/processed/* || true
  mkdir -p data/uploads data/processed
  touch data/uploads/.gitkeep data/processed/.gitkeep
  rm -f .env || true
fi

# 2) Lint + auto-fix
echo "==> Ruff (with --fix)..."
poetry run ruff check . --fix

# 3) Format
echo "==> Black formatting..."
poetry run black .

# 4) Tests
echo "==> Running tests..."
poetry run pytest -q

echo "==> All checks passed ✅"
echo "==> Git status:"
git status