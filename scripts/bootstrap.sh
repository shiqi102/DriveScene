#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON="${DRIVESCENE_BASE_PYTHON:-/Data1/mambaforge/envs/worldmodel/bin/python}"
mkdir -p data/nuscenes data/downloads checkpoints outputs logs
if [ ! -x .venv/bin/python ]; then
  "$BASE_PYTHON" -m venv --system-site-packages .venv
fi
# Dependencies are installed into the project overlay. Base environment is read-only.
.venv/bin/python -m pip install -e '.[data,models,test]'
.venv/bin/python -m pip freeze > requirements.lock.txt
chmod +x scripts/run.sh
scripts/run.sh doctor
