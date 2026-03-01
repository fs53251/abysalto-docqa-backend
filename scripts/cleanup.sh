#!/usr/bin/env bash
set -euo pipefail

# Cleanup script for DocQA project
# Removes caches, temp files, runtime data (uploads/processed), and local .env
# Safe to run before commit.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Running cleanup in: $PROJECT_ROOT"

echo "==> Removing Python bytecode caches..."
find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
find . -type f -name "*.pyc" -delete || true
find . -type f -name "*.pyo" -delete || true

echo "==> Removing test/lint/tool caches..."
rm -rf .pytest_cache .ruff_cache .mypy_cache || true
rm -f .coverage coverage.xml || true
rm -rf htmlcov .tox .nox || true

echo "==> Removing OS/editor junk (if present)..."
find . -type f -name ".DS_Store" -delete || true
find . -type f -name "Thumbs.db" -delete || true
find . -type f -name "Desktop.ini" -delete || true

echo "==> Cleaning runtime data directories (keeps .gitkeep)..."
rm -rf data/uploads/* data/processed/* || true
mkdir -p data/uploads data/processed
touch data/uploads/.gitkeep data/processed/.gitkeep

#echo "==> Removing local secrets file (.env)..."
#rm -f .env || true

echo "==> Done."
echo "Tip: run 'git status' to verify only intended changes remain."