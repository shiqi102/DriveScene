from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import imageio.v2 as imageio
import numpy as np

from .data import read_manifest
from .visualize import render_world


def render_video(manifest, world_directory, output, scene_name=None, fps=2.0, limit=None, view="2d"):
    """Render a chronological six-camera + BEV WorldModel video.

    The default selects the first scene because concatenating unrelated nuScenes
    scenes would introduce discontinuous jumps into one video.
    """
    rows = list(read_manifest(manifest))
    if not rows:
        raise ValueError("Manifest is empty")
    selected_scene = scene_name or rows[0]["scene_name"]
    rows = [row for row in rows if row["scene_name"] == selected_scene]
    rows.sort(key=lambda row: row["timestamp_us"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"Scene not present in manifest: {selected_scene}")

    world_directory = Path(world_directory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = [row["sample_token"] for row in rows
               if not (world_directory / f"{row['sample_token']}.json").is_file()
               or not (world_directory / f"{row['sample_token']}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} compiled frames in {world_directory}; "
            f"run inference for this scene first. First missing token: {missing[0]}")

    # Render and encode on local scratch storage. Shared mounts are much slower
    # for dozens of PNG writes and expose an invalid partial MP4 if interrupted.
    with tempfile.TemporaryDirectory(prefix="drivescene-video-") as temporary:
        temporary = Path(temporary)
        encoded = temporary / output.name
        with imageio.get_writer(
            encoded,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
            ffmpeg_log_level="error",
            # imageio already selects yuv420p for libx264; only request faststart
            # here so FFmpeg does not receive a duplicate output -pix_fmt option.
            ffmpeg_params=["-movflags", "+faststart"],
        ) as writer:
            for index, row in enumerate(rows):
                token = row["sample_token"]
                world = json.loads((world_directory / f"{token}.json").read_text(encoding="utf-8"))
                with np.load(world_directory / f"{token}.npz") as archive:
                    arrays = {key: archive[key] for key in archive.files}
                frame_path = temporary / f"{index:06d}.png"
                render_world(world, arrays, frame_path, row, view=view)
                writer.append_data(imageio.imread(frame_path))

        staged = output.with_name(output.name+".tmp")
        try:
            shutil.copyfile(encoded, staged)
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)

    report = {
        "video": str(output),
        "scene_name": selected_scene,
        "frames": len(rows),
        "fps": fps,
        "duration_seconds": len(rows) / fps,
        "world_directory": str(world_directory),
        "view": view,
    }
    output.with_suffix(".video.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
