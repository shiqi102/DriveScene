#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root="$PWD"
gpus="${1:-0,1,2,3,4,5,6,7}"
gpu_count="$(awk -F, '{print NF}' <<<"$gpus")"
nusc="$PWD/data/nuscenes"
maptr="$PWD/third_party/MapTR"

if [[ ! -d "$nusc/v1.0-trainval" ]]; then
  echo "Missing full nuScenes metadata: $nusc/v1.0-trainval" >&2
  exit 2
fi
if [[ ! -d "$PWD/data/can_bus" ]]; then
  echo "Missing nuScenes CAN bus expansion: $PWD/data/can_bus" >&2
  exit 2
fi
if [[ ! -x "$PWD/.venv-openmmlab/bin/python" ]]; then
  echo "Missing OpenMMLab runtime: $PWD/.venv-openmmlab" >&2
  exit 2
fi

mkdir -p "$maptr/data" "$maptr/ckpts" "$PWD/outputs/maptrv2-full"
ln -sfn "$nusc" "$maptr/data/nuscenes"
ln -sfn "$PWD/data/can_bus" "$maptr/data/can_bus"
ln -sfn "$PWD/checkpoints/resnet50-11ad3fa6.pth" "$maptr/ckpts/resnet50-19c8e357.pth"

if [[ ! -f "$nusc/nuscenes_map_infos_temporal_train.pkl" ||
      ! -f "$nusc/nuscenes_map_infos_temporal_val.pkl" ]]; then
  cd "$maptr"
  "$root/.venv-openmmlab/bin/python" tools/maptrv2/custom_nusc_map_converter.py \
    --root-path data/nuscenes \
    --out-dir data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus data
  cd "$root"
fi

cd "$maptr"
export CUDA_VISIBLE_DEVICES="$gpus"
export PATH="$root/.venv-openmmlab/bin:$PATH"
export PYTHONPATH="$maptr${PYTHONPATH:+:$PYTHONPATH}"
hash -r
bash ./tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  "$gpu_count" \
  --work-dir "$root/outputs/maptrv2-full"
