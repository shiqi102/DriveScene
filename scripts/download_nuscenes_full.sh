#!/usr/bin/env bash
set -euo pipefail

# Download and extract the nuScenes v1.0 trainval split, map expansion, and
# CAN bus expansion. Downloads are resumable and extraction steps are marked
# so the script can be run again after an interruption.

data_root="${1:-/desay_200T/zy/DriveScene/data}"
download_root="$data_root/downloads"
nuscenes_root="$data_root/nuscenes"
marker_root="$data_root/.extract-markers"
parallel_downloads="${PARALLEL_DOWNLOADS:-4}"

mkdir -p "$download_root" "$nuscenes_root" "$marker_root"

declare -a urls=(
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval04_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval08_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval10_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip"
)

download_one() {
  local url="$1"
  local output="$download_root/${url##*/}"
  echo "[download] ${url##*/}"
  # The host has a localhost HTTPS proxy configured in wget. It is unstable
  # for multi-hour transfers, so bypass it and keep retrying indefinitely.
  # --continue preserves every completed byte across retries and reruns.
  while ! wget \
    --no-proxy \
    --continue \
    --tries=20 \
    --timeout=60 \
    --retry-connrefused \
    --waitretry=5 \
    --progress=dot:giga \
    --output-document="$output" \
    "$url"; do
    echo "[retry] ${url##*/} in 10 seconds" >&2
    sleep 10
  done
}
export -f download_one
export download_root

printf '%s\n' "${urls[@]}" | xargs -n1 -P "$parallel_downloads" bash -c 'download_one "$1"' _

echo "[verify] gzip archives"
find "$download_root" -maxdepth 1 -type f -name '*.tgz' -print0 \
  | xargs -0 -n1 -P "$parallel_downloads" gzip -t

echo "[verify] zip archives"
unzip -tq "$download_root/nuScenes-map-expansion-v1.3.zip"
unzip -tq "$download_root/can_bus.zip"

extract_tgz() {
  local archive="$1"
  local base
  base="$(basename "$archive")"
  local marker="$marker_root/$base.done"
  if [[ -f "$marker" ]]; then
    echo "[extract] already complete: $base"
    return
  fi
  echo "[extract] $base"
  tar --use-compress-program=pigz -xf "$archive" -C "$nuscenes_root"
  touch "$marker"
}

extract_tgz "$download_root/v1.0-trainval_meta.tgz"
for part in $(seq -w 1 10); do
  extract_tgz "$download_root/v1.0-trainval${part}_blobs.tgz"
done

map_marker="$marker_root/nuScenes-map-expansion-v1.3.zip.done"
if [[ ! -f "$map_marker" ]]; then
  echo "[extract] nuScenes-map-expansion-v1.3.zip"
  map_tmp="$data_root/.map-expansion-tmp"
  rm -rf "$map_tmp"
  mkdir -p "$map_tmp" "$nuscenes_root/maps"
  unzip -q -o "$download_root/nuScenes-map-expansion-v1.3.zip" -d "$map_tmp"
  if [[ -d "$map_tmp/maps" ]]; then
    cp -a "$map_tmp/maps/." "$nuscenes_root/maps/"
  elif [[ -d "$map_tmp/basemap" && -d "$map_tmp/expansion" ]]; then
    cp -a "$map_tmp/." "$nuscenes_root/maps/"
  else
    echo "Unexpected map expansion layout" >&2
    find "$map_tmp" -maxdepth 2 -type d -print >&2
    exit 3
  fi
  rm -rf "$map_tmp"
  touch "$map_marker"
fi

can_marker="$marker_root/can_bus.zip.done"
if [[ ! -f "$can_marker" ]]; then
  echo "[extract] can_bus.zip"
  can_tmp="$data_root/.can-bus-tmp"
  rm -rf "$can_tmp"
  mkdir -p "$can_tmp" "$data_root/can_bus"
  unzip -q -o "$download_root/can_bus.zip" -d "$can_tmp"
  if [[ -d "$can_tmp/can_bus" ]]; then
    cp -a "$can_tmp/can_bus/." "$data_root/can_bus/"
  else
    cp -a "$can_tmp/." "$data_root/can_bus/"
  fi
  rm -rf "$can_tmp"
  touch "$can_marker"
fi

echo "[validate] dataset layout"
test -f "$nuscenes_root/v1.0-trainval/scene.json"
test -d "$nuscenes_root/samples/CAM_FRONT"
test -d "$nuscenes_root/sweeps/CAM_FRONT"
test -f "$nuscenes_root/maps/expansion/boston-seaport.json"
test -d "$data_root/can_bus"

echo "[complete] nuScenes trainval is ready at $nuscenes_root"
du -sh "$nuscenes_root" "$data_root/can_bus" "$download_root"
