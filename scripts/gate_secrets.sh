#!/usr/bin/env bash
# Gate: secrets. Replace the body with this app's real check. Exit non-zero on failure.
# Fails closed until configured, so an unconfigured gate can never read as green.
set -euo pipefail
echo "gate_secrets: NOT CONFIGURED — edit scripts/gate_secrets.sh for this app" >&2
exit 1
