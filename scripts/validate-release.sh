#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required"
  exit 1
fi

echo "==> Lock"
uv lock

echo "==> Lint"
uv run ruff check .

echo "==> Tests"
uv run pytest

echo "==> Smoke"
uv run python scripts/smoke.py

echo "==> Build"
build_dir="$(mktemp -d)"
uv build --out-dir "$build_dir"

echo "==> Metadata check"
uvx twine check "$build_dir"/*

echo "==> Release validation OK"
