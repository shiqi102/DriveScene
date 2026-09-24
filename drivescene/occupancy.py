"""Offline sparse voxel supervision from nuScenes LiDARseg (never inference input).

Unobserved voxels remain 255; this is NOT a dense occupancy ground-truth dataset.
"""
import json
from pathlib import Path
import numpy as np
from .data import read_manifest
from .geometry import transform_matrix, transform_points


def _trace_free_voxels(origin, endpoints, low, high, shape, chunk_size=2048):
    """Return voxel indices traversed before each LiDAR return.

    Sampling at one voxel or finer along the dominant axis is sufficient for the
    configured axis-aligned unit grid. Endpoints are excluded and assigned their
    semantic surface class afterwards, so occupied always wins over free.
    """
    resolution = (high-low)/shape
    result = []
    origin = np.asarray(origin, dtype=np.float64)
    for start in range(0, len(endpoints), chunk_size):
        points = endpoints[start:start+chunk_size]
        delta = points-origin
        steps = np.ceil(np.max(np.abs(delta)/resolution, axis=1)).astype(np.int64)
        steps = np.maximum(steps, 1)
        count = int(steps.max())
        sample_index = np.arange(count)[None]
        valid = sample_index < steps[:, None]
        fractions = sample_index/steps[:, None]
        samples = origin[None, None]+delta[:, None]*fractions[..., None]
        indices = np.floor((samples-low)/resolution).astype(np.int64)
        inside = ((indices >= 0) & (indices < shape)).all(-1)
        result.append(indices[valid & inside])
    if not result:
        return np.empty((0, 3), dtype=np.int64)
    return np.concatenate(result)


def prepare_sparse(root, version, manifest, output_manifest, grid):
    from nuscenes.nuscenes import NuScenes
    root = Path(root).resolve()
    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    if "lidarseg" not in nusc.table_names:
        raise FileNotFoundError("Install the nuScenes lidarseg mini expansion first")
    directory = root.parent / "occupancy_raytraced"
    manifest = Path(manifest)
    output_manifest = Path(output_manifest)
    if manifest.resolve() == output_manifest.resolve():
        raise ValueError("input and output manifest must be different; prepare a base manifest first")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    low = np.array([grid[k][0] for k in ("x", "y", "z")])
    high = np.array([grid[k][1] for k in ("x", "y", "z")])
    shape = np.array(grid["shape"])
    names = nusc.lidarseg_idx2name_mapping
    ignored = [index for index, name in names.items() if name in ("noise", "vehicle.ego")]
    drivable = [index for index, name in names.items() if name == "flat.driveable_surface"]
    counts = {"frames": 0, "labeled_voxels": 0, "free_voxels": 0,
              "label_type": "ray_traced_sparse_lidarseg_voxels", "dense_ground_truth": False}
    with output_manifest.open("w") as stream:
        for row in read_manifest(manifest):
            sample = nusc.get("sample", row["sample_token"])
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            lidar_to_ego = np.linalg.inv(np.asarray(row["ego_to_global"])) @ transform_matrix(pose["translation"], pose["rotation"]) @ transform_matrix(cs["translation"], cs["rotation"])
            points = np.fromfile(root / sd["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
            labels = np.fromfile(root / nusc.get("lidarseg", sd["token"])["filename"], dtype=np.uint8)
            if len(points) != len(labels):
                raise ValueError("Point/label count mismatch")
            xyz = transform_points(points, lidar_to_ego)
            indices = np.floor((xyz-low)/(high-low)*shape).astype(np.int64)
            valid = ((indices >= 0) & (indices < shape)).all(1) & ~np.isin(labels, ignored)
            indices, labels = indices[valid], labels[valid]
            semantic = np.full(grid["shape"], 255, dtype=np.uint8)
            free_indices = _trace_free_voxels(lidar_to_ego[:3, 3], xyz[valid], low, high, shape)
            if len(free_indices):
                semantic[tuple(free_indices.T)] = 0
            road = np.isin(labels, drivable)
            # Occupied wins when a voxel contains both road and object returns.
            semantic[tuple(indices[road].T)] = 2
            semantic[tuple(indices[~road].T)] = 1
            path = directory / row["scene_name"] / row["sample_token"] / "labels.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, drivescene_semantic=semantic,
                                label_type="ray_traced_sparse_lidarseg_voxels", axis_order="XYZ",
                                grid_json=json.dumps(grid), lidar_to_ego=lidar_to_ego)
            row["lidar_to_ego"] = lidar_to_ego.tolist()
            row["targets"]["occupancy_path"] = str(path)
            row["targets"]["occupancy_label_type"] = "ray_traced_sparse_lidarseg_voxels"
            counts["frames"] += 1
            counts["labeled_voxels"] += int((semantic != 255).sum())
            counts["free_voxels"] += int((semantic == 0).sum())
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    output_manifest.with_suffix(".report.json").write_text(json.dumps(counts, indent=2))
    return counts
