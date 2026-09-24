#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

gpus="${1:-0,1,2,3,4,5,6,7}"
min_free_mib="${MIN_FREE_MIB:-20000}"
poll_seconds="${GPU_POLL_SECONDS:-60}"

IFS=',' read -r -a requested_gpus <<<"$gpus"

while true; do
  declare -A free_by_gpu=()
  while IFS=',' read -r index free_mib; do
    index="${index//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    free_by_gpu["$index"]="$free_mib"
  done < <(nvidia-smi \
    --query-gpu=index,memory.free \
    --format=csv,noheader,nounits)

  ready=true
  summary=()
  for gpu in "${requested_gpus[@]}"; do
    free_mib="${free_by_gpu[$gpu]:-0}"
    summary+=("gpu${gpu}=${free_mib}MiB")
    if (( free_mib < min_free_mib )); then
      ready=false
    fi
  done

  printf '[%s] %s\n' "$(date '+%F %T')" "${summary[*]}"
  if [[ "$ready" == true ]]; then
    echo "All requested GPUs have at least ${min_free_mib} MiB free; starting MapTRv2."
    break
  fi

  echo "Waiting ${poll_seconds}s for all requested GPUs to have at least ${min_free_mib} MiB free."
  sleep "$poll_seconds"
done

exec bash scripts/train_maptrv2_full.sh "$gpus"
