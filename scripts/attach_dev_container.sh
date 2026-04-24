#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

CID="$(docker compose ps -q dev)"

if [[ -z "$CID" ]]; then
  echo "The dev container is not running."
  echo "Start it first with:"
  echo "  ./scripts/run_dev_container.sh"
  exit 1
fi

RUNNING="$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null || true)"
if [[ "$RUNNING" != "true" ]]; then
  echo "The dev container exists but is not running."
  echo "Start it first with:"
  echo "  ./scripts/run_dev_container.sh"
  exit 1
fi

docker compose exec dev bash -lc '
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate inverse-skills || true
fi
exec bash
'
