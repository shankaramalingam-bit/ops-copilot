#!/usr/bin/env bash
# Gate: tests. pytest exit 5 (no tests collected) is a failure by design: set -e keeps it failing.
# A fresh per-run bytecode cache and no pytest cache dir: stale bytecode is never tested.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHONPYCACHEPREFIX="$(mktemp -d "${TMPDIR:-/tmp}/pyc.XXXXXX")"
export PYTHONPYCACHEPREFIX
trap 'rm -rf "$PYTHONPYCACHEPREFIX"' EXIT
uv run pytest -q -p no:cacheprovider
