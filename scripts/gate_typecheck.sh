#!/usr/bin/env bash
# Gate: typecheck (owner decision: mypy src/, missing imports ignored via pyproject).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run mypy src/
