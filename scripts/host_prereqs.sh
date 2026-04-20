#!/usr/bin/env bash
set -euo pipefail

echo "Check Docker:"
docker --version

echo "Check Docker Compose:"
docker compose version

echo "Check NVIDIA runtime:"
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
