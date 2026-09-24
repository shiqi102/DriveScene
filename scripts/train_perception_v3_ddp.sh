#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
gpus="${GPUS:-0,1,2,3}"
steps="${STEPS:-10000}"
output="${OUTPUT:-outputs/train-mini-perception-v3-ddp}"
manifest="${MANIFEST:-data/processed/mini-v3-with-occ.jsonl}"
config="${CONFIG:-configs/default.yaml}"
workers="$(awk -F, '{print NF}' <<<"$gpus")"

if [[ ! -f "$manifest" ]]; then
  echo "Missing perception-v3 manifest: $manifest" >&2
  exit 2
fi
if [[ -e "$output/last.pt" || -e "$output/latest.pt" ]]; then
  echo "Refusing to overwrite an existing run: $output" >&2
  echo "Choose a new OUTPUT for from-scratch perception-v3 training." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpus"
exec .venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$workers" \
  -m drivescene.cli \
  --config "$config" \
  train \
  --manifest "$manifest" \
  --output "$output" \
  --steps "$steps" \
  --save-every 250 \
  --device cuda \
  --backbone checkpoints/resnet50-11ad3fa6.pth
