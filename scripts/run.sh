#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$PWD/checkpoints/torch"
export YOLO_CONFIG_DIR="$PWD/.yolo"
export MPLCONFIGDIR="$PWD/.matplotlib"
exec .venv/bin/python -m drivescene.cli "$@"
