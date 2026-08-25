#!/usr/bin/env bash
#
# Stop every FR5 node in this workspace and wait until they are actually gone.
#
#   ./scripts/stop_all.sh
#
# Use before starting a session, or when something is left over from a run that
# ended badly. Safe to run when nothing is up.
#
# The monitoring GUI is deliberately left alone — it holds no hardware, and it
# reconnects on its own when the bridge comes back.
set -uo pipefail

cd "$(dirname "$0")/.."
WORKSPACE="$(pwd)"
export WORKSPACE
# shellcheck source=fr5_nodes.sh
source "$WORKSPACE/scripts/fr5_nodes.sh"

echo "FR5 노드 정리 — $WORKSPACE"
fr5_stop_all
