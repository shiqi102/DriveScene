#!/usr/bin/env python3
"""Run scene-safe multi-GPU inference and merge the compiled frames.

Each scene stays on one worker so temporal BEV memory and ByteTrack IDs are
never split across GPUs. Independent scenes are distributed round-robin.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description="Scene-parallel DriveScene inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU IDs")
    parser.add_argument("--limit", type=int, help="Optional global frame limit")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lights", action="store_true")
    args = parser.parse_args()

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        parser.error("--gpus must contain at least one GPU ID")
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        parser.error("manifest is empty")

    scene_order, by_scene = [], {}
    for row in rows:
        token = row["scene_token"]
        if token not in by_scene:
            scene_order.append(token)
            by_scene[token] = []
        by_scene[token].append(row)
    active_gpus = gpu_ids[:min(len(gpu_ids), len(scene_order))]
    shards = [[] for _ in active_gpus]
    # Greedy balancing by current frame count keeps whole scenes together.
    for token in sorted(scene_order, key=lambda key: len(by_scene[key]), reverse=True):
        target = min(range(len(shards)), key=lambda index: len(shards[index]))
        shards[target].extend(by_scene[token])
    for shard in shards:
        shard.sort(key=lambda row: (scene_order.index(row["scene_token"]), row["timestamp_us"]))

    output = Path(args.output)
    if not output.is_absolute() and output.parts and output.parts[0] == "outputs":
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        output = Path(config["output"]).joinpath(*output.parts[1:])
    shard_root = output / ".shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes = []
    assignments = []
    for index, (gpu, shard) in enumerate(zip(active_gpus, shards)):
        manifest = shard_root / f"gpu-{gpu}.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in shard),
                            encoding="utf-8")
        shard_output = shard_root / f"gpu-{gpu}"
        command = [sys.executable, "-m", "drivescene.cli", "--config", args.config,
                   "--threads", str(args.threads),
                   "run", "--mode", "native_research", "--manifest", str(manifest),
                   "--checkpoint", args.checkpoint, "--output", str(shard_output),
                   "--limit", str(len(shard)), "--device", "cuda:0"]
        if args.lights:
            command.append("--lights")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        processes.append((gpu, shard_output, subprocess.Popen(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)))
        assignments.append({"gpu": gpu, "frames": len(shard),
                            "scenes": sorted({row["scene_name"] for row in shard})})

    failures = []
    for gpu, _, process in processes:
        stdout, _ = process.communicate()
        if process.returncode:
            failures.append({"gpu": gpu, "returncode": process.returncode, "output": stdout[-4000:]})
        else:
            print(f"GPU {gpu}: {stdout.strip()}", flush=True)
    if failures:
        raise RuntimeError("One or more inference workers failed:\n"+json.dumps(failures, indent=2))

    paths = []
    for _, shard_output, _ in processes:
        report = json.loads((shard_output/"run.json").read_text(encoding="utf-8"))
        for source in shard_output.iterdir():
            if source.name == "run.json" or not source.is_file():
                continue
            destination = output/source.name
            shutil.copy2(source, destination)
        paths.extend(str(output/Path(path).name) for path in report["outputs"])
    report = {"mode": "native_research_multi_gpu", "frames": len(paths),
              "checkpoint": args.checkpoint, "assignments": assignments,
              "outputs": paths}
    (output/"run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "frames", "checkpoint", "assignments")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
