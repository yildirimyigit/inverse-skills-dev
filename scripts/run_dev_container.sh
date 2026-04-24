#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"

if [[ "${1:-}" == "--build" ]]; then
  docker compose up -d --build dev
else
  docker compose up -d dev
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Error: docker compose is not available."
  exit 1
fi

export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"

if [[ "${1:-}" == "--build" ]]; then
  docker compose up -d --build dev
else
  docker compose up -d dev
fi

echo
echo "Container is up."
docker compose ps dev
echo
echo "Attach with:"
echo "  ./scripts/attach_dev_container.sh"
