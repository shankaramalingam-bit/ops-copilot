#!/usr/bin/env bash
# Gate: infra_diff. Replace the body with this app's real check. Exit non-zero on failure.
# Fails closed until configured, so an unconfigured gate can never read as green.
set -euo pipefail
echo "gate_infra_diff: NOT CONFIGURED — edit scripts/gate_infra_diff.sh for this app" >&2
exit 1
