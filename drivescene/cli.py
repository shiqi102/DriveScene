from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import yaml


def _resolve_output(value, output_root: Path, default_name: str | None = None,
                    all_relative: bool = False) -> Path:
    """Map legacy outputs/... paths into the configured output root."""
    if value is None:
        if default_name is None:
            raise ValueError("an output path is required")
        return output_root / default_name
    path = Path(value)
    if not path.is_absolute() and path.parts and path.parts[0] == "outputs":
        return output_root.joinpath(*path.parts[1:])
    if all_relative and not path.is_absolute():
        return output_root / path
    return path


def main():
    parser = argparse.ArgumentParser(description="DriveScene research perception integration")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--root")
    prepare.add_argument("--version")
    prepare.add_argument("--output")
    prepare.add_argument("--limit", type=int)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=["annotations", "native_research"], required=True)
    run.add_argument("--manifest")
    run.add_argument("--checkpoint")
    run.add_argument("--output")
    run.add_argument("--limit", type=int, default=None,
                       help="Max frames; default is the full manifest")
    run.add_argument("--device", default="cpu")
    run.add_argument("--lights", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--manifest")
    train.add_argument("--output")
    train.add_argument("--steps", type=int, default=10)
    train.add_argument("--device", default="cpu")
    train.add_argument("--backbone")
    train.add_argument("--resume")
    train.add_argument("--save-every", type=int, default=250)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output")
    smoke.add_argument("--device", default="cpu")
    commands.add_parser("doctor")
    video = commands.add_parser("video")
    video.add_argument("--manifest")
    video.add_argument("--world", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--scene-name")
    video.add_argument("--fps", type=float, default=2.0)
    video.add_argument("--limit", type=int)
    video.add_argument("--view", choices=["2d", "3d"], default="2d",
                       help="BEV panel: flat cost map or 3D occupancy/boxes")
    mcap = commands.add_parser("mcap")
    mcap.add_argument("--manifest")
    mcap.add_argument("--world", required=True)
    mcap.add_argument("--output", required=True)
    mcap.add_argument("--scene-name")
    mcap.add_argument("--limit", type=int)
    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--manifest")
    evaluate_command.add_argument("--world", required=True)
    evaluate_command.add_argument("--output")
    evaluate_command.add_argument("--split", choices=["train", "val"], default="val")
    evaluate_command.add_argument("--video-output")
    evaluate_command.add_argument("--fps", type=float, default=2.0)
    evaluate_command.add_argument("--no-video", action="store_true")
    evaluate_command.add_argument("--skip-official", action="store_true")
    sparse = commands.add_parser("prepare-sparse-occupancy")
    sparse.add_argument("--manifest")
    sparse.add_argument("--output")
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("frame")
    compile_command.add_argument("--occupancy")
    compile_command.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(config["output"])
    artifact_output_root = Path(
        os.environ.get("DRIVESCENE_ARTIFACT_OUTPUT")
        or config.get("artifact_output")
        or config["output"])
    if args.command in ("run", "train", "smoke", "doctor"):
        import torch
        torch.set_num_threads(args.threads)
    print_json = True
    if args.command == "prepare":
        from .data import prepare_nuscenes
        result = prepare_nuscenes(args.root or config["data_root"], args.version or config["version"],
                                  args.output or config.get("base_manifest", config["manifest"]), args.limit,
                                  config["model"]["map_points"], config["grid"])
    elif args.command == "video":
        from .video import render_video
        if args.fps <= 0:
            parser.error("--fps must be positive")
        result = render_video(args.manifest or config["manifest"], args.world,
                              _resolve_output(args.output, artifact_output_root,
                                              all_relative=True), args.scene_name,
                              args.fps, args.limit, args.view)
    elif args.command == "mcap":
        from .mcap_export import export_mcap
        result = export_mcap(args.manifest or config["manifest"], args.world,
                             _resolve_output(args.output, artifact_output_root,
                                             all_relative=True), args.scene_name, args.limit)
    elif args.command == "evaluate":
        from .evaluate import evaluate, format_terminal_report
        if args.fps <= 0:
            parser.error("--fps must be positive")
        name = Path(args.world).name+"-evaluation"
        evaluation_output = _resolve_output(args.output, output_root, name)
        if args.no_video:
            video_output = None
        elif args.video_output:
            video_output = _resolve_output(args.video_output, artifact_output_root,
                                           all_relative=True)
        else:
            video_output = artifact_output_root/name/"validation-errors.mp4"
        result = evaluate(config, args.manifest or config["manifest"], args.world,
                          evaluation_output, args.split, video_output, args.fps,
                          not args.skip_official)
        report = json.loads(Path(result["metrics_json"]).read_text(encoding="utf-8"))
        print(format_terminal_report(report))
        print(f"\nMetrics JSON: {result['metrics_json']}")
        print(f"Metrics Markdown: {result['metrics_markdown']}")
        if result.get("error_video"):
            print(f"Error video: {result['error_video']}")
        print_json = False
    elif args.command == "prepare-sparse-occupancy":
        from .occupancy import prepare_sparse
        result = prepare_sparse(config["data_root"], config["version"],
                                args.manifest or config.get("base_manifest", config["manifest"]),
                                args.output or config["manifest"], config["grid"])
    elif args.command == "run":
        from .pipeline import run
        result = run(config, args.manifest or config["manifest"],
                     _resolve_output(args.output, output_root, "run"), args.mode, args.checkpoint,
                     args.limit, args.device, args.lights)
    elif args.command == "train":
        from .pipeline import train
        if args.steps < 1:
            parser.error("--steps must be positive")
        if args.save_every < 1:
            parser.error("--save-every must be positive")
        result = train(config, args.manifest or config["manifest"],
                       _resolve_output(args.output, output_root, "train"),
                       args.steps, args.device, args.backbone, args.resume, args.save_every)
    elif args.command == "smoke":
        from .pipeline import smoke
        result = smoke(config, _resolve_output(args.output, output_root, "smoke"), args.device)
    elif args.command == "compile":
        import numpy as np
        from .compiler import compile_scene, save_world
        occupancy = None
        if args.occupancy:
            with np.load(args.occupancy) as archive:
                occupancy = archive["semantic"]
        world, arrays = compile_scene(json.loads(Path(args.frame).read_text()), config, occupancy)
        result = {"path": str(save_world(world, arrays,
                                         _resolve_output(args.output, output_root, "compiled")))}
    else:
        import importlib.metadata
        import shutil
        names = ["torch", "torchvision", "nuscenes-devkit", "ultralytics", "mmcv-full", "mmdet", "mmdet3d"]
        packages = {}
        for name in names:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result = {"packages": packages, "cuda_available": torch.cuda.is_available(),
                  "dataset": Path(config["data_root"], config["version"], "sample.json").is_file(),
                  "map_expansion": Path(config["data_root"], "maps/expansion").is_dir(),
                  "free_disk_gb": round(shutil.disk_usage(".").free/2**30, 1),
                  "backend": config["model"]["backend"], "official_shared_stack_validated": False}
    if print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
