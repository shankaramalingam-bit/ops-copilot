#!/usr/bin/env bash
# Gate: tests. pytest exit 5 (no tests collected) is a failure by design: set -e keeps it failing.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run pytest -q
