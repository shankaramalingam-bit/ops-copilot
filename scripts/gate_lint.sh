#!/usr/bin/env bash
# Gate: lint (ruff check + ruff format --check).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run ruff check .
uv run ruff format --check .
