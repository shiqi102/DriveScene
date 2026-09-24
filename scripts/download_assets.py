"""Repeatable downloads, bounded retries, format validation and SHA256 inventory."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ASSETS = {
    "mini": ("https://www.nuscenes.org/data/v1.0-mini.tgz", "data/downloads/v1.0-mini.tgz", "tar"),
    "maps": ("https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip", "data/downloads/maps-v1.3.zip", "zip"),
    "resnet50": ("https://download.pytorch.org/models/resnet50-11ad3fa6.pth", "checkpoints/resnet50-11ad3fa6.pth", "torch"),
    "yolov8": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", "checkpoints/yolov8n.pt", "torch"),
    "bevformer": ("https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth", "checkpoints/bevformer_tiny_epoch_24.pth", "torch"),
    "occformer": ("https://github.com/zhangyp15/OccFormer/releases/download/assets/occformer_nusc_r50.pth", "checkpoints/occformer_nusc_r50.pth", "torch"),
}


def fetch(name):
    url, filename, kind = ASSETS[name]
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            temporary = path.with_suffix(path.suffix+".part")
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "15", "--max-time", "1800",
                            "--speed-time", "30", "--speed-limit", "1024",
                            "-C", "-", url, "-o", str(temporary)], check=True)
            temporary.replace(path)
        if kind == "zip" and not zipfile.is_zipfile(path):
            raise ValueError("Downloaded content is not a ZIP archive")
        if kind == "tar" and not tarfile.is_tarfile(path):
            raise ValueError("Downloaded content is not a TAR archive")
        if kind == "torch":
            with path.open("rb") as stream:
                header = stream.read(128).lower()
            if path.stat().st_size < 1_000_000 or b"<html" in header or b"<!doctype" in header:
                raise ValueError("Invalid checkpoint download")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(block)
        return name, {"url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "status": "downloaded"}
    except Exception as exc:
        return name, {"url": url, "path": str(path), "status": "failed", "error": str(exc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="*", choices=list(ASSETS))
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result = dict(pool.map(fetch, args.assets or list(ASSETS)))
    Path("logs").mkdir(exist_ok=True)
    Path("logs/assets.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(any(item["status"] == "failed" for item in result.values()))
