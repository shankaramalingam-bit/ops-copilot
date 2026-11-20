#!/usr/bin/env bash
# Gate: typecheck. Replace the body with this app's real check. Exit non-zero on failure.
# Fails closed until configured, so an unconfigured gate can never read as green.
set -euo pipefail
echo "gate_typecheck: NOT CONFIGURED — edit scripts/gate_typecheck.sh for this app" >&2
exit 1
