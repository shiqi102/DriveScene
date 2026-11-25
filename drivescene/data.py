"""Explicit camera-only inputs; all annotation fields stay under `targets`."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .geometry import transform_matrix, transform_points, resample_line

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]


def read_manifest(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]


def _line_parts(geometry):
    """Split a clipped geometry into lines, re-joining pieces of a cut ring."""
    from shapely.ops import linemerge
    if geometry.is_empty:
        return []
    if geometry.geom_type == "MultiLineString":
        geometry = linemerge(geometry)
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]
    return [part for part in getattr(geometry, "geoms", [])
            if part.geom_type in ("LineString", "LinearRing")]


def extract_map(nmap, ego_to_global, grid, points=20):
    """Extract supervision clipped to the exact model grid in ego coordinates.

    Road boundaries are the outline of the union of road segments and lanes, so
    seams between adjacent segments are not labelled as boundaries. Crossings
    are unioned as well. `closed` marks rings whose start point is arbitrary.
    """
    from shapely.geometry import box, LineString, Polygon
    from shapely.ops import unary_union
    x0, x1 = grid["x"]
    y0, y1 = grid["y"]
    local_corners = np.array([[x0, y0, 0], [x1, y0, 0],
                              [x1, y1, 0], [x0, y1, 0]])
    global_corners = transform_points(local_corners, ego_to_global)[:, :2]
    global_clip = Polygon(global_corners)
    patch = global_clip.bounds
    records = nmap.get_records_in_patch(patch,
        ["lane_divider", "road_divider", "road_segment", "ped_crossing", "stop_line", "lane"], mode="intersect")
    local_clip = box(x0, y0, x1, y1)
    inverse = np.linalg.inv(ego_to_global)
    vectors = []

    def append_geometry(geometry, kind, token):
        for part in _line_parts(geometry.intersection(global_clip)):
            if part.length <= 0.2:
                continue
            global_xy = np.asarray(part.coords, dtype=np.float64)[:, :2]
            global_xyz = np.column_stack([global_xy, np.full(len(global_xy), ego_to_global[2, 3])])
            ego_xy = transform_points(global_xyz, inverse)[:, :2]
            # A second ego-frame clip removes numerical spill and makes the
            # supervision contract explicit even for rotated map patches.
            for local in _line_parts(LineString(ego_xy).intersection(local_clip)):
                append_local(local, kind, token)

    def append_local(geometry, kind, token):
        if geometry.length <= .2:
            return
        xy = resample_line(geometry.coords, points)
        xy[:, 0] = np.clip(xy[:, 0], x0, x1)
        xy[:, 1] = np.clip(xy[:, 1], y0, y1)
        vectors.append({"id": f"{kind}:{token}:{len(vectors)}", "kind": kind,
                        "points": xy.tolist(), "score": 1.0,
                        "closed": bool(geometry.is_closed)})

    def append_union_outline(layers, kind):
        polygons = [nmap.extract_polygon(nmap.get(layer, token)["polygon_token"]).buffer(0)
                    for layer in layers for token in records[layer]]
        polygons = [polygon for polygon in polygons if not polygon.is_empty]
        if not polygons:
            return
        for index, polygon in enumerate(_polygon_parts(unary_union(polygons))):
            for ring in (polygon.exterior, *polygon.interiors):
                append_geometry(LineString(ring.coords), kind, f"union{index}")

    for layer in ("lane_divider", "road_divider"):
        for token in records[layer]:
            append_geometry(nmap.extract_line(nmap.get(layer, token)["line_token"]), "divider", token)
    append_union_outline(("road_segment", "lane"), "boundary")
    append_union_outline(("ped_crossing",), "pedestrian_crossing")
    for token in records["stop_line"]:
        record = nmap.get("stop_line", token)
        if "line_token" in record:
            append_geometry(nmap.extract_line(record["line_token"]), "stop_line", token)
            continue
        polygon = nmap.extract_polygon(record["polygon_token"])
        # Long axis of the painted rectangle is the stop-line segment.
        rect = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
        mids = (rect + np.roll(rect, -1, axis=0))/2
        candidates = [LineString([mids[i], mids[i+2]]) for i in range(2)]
        append_geometry(max(candidates, key=lambda item: item.length), "stop_line", token)
    if records["lane"]:
        for token, pts in nmap.discretize_lanes(records["lane"], 1.0).items():
            if len(pts) > 1:
                append_geometry(LineString(np.asarray(pts)[:, :2]), "centerline", token)
    return vectors


def prepare_nuscenes(root, version, output, limit=None, map_points=20, grid=None):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    if grid is None:
        raise ValueError("prepare_nuscenes requires the model grid for exact map clipping")
    root = Path(root).resolve()
    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    splits = create_splits_scenes()
    train_names = set(splits["mini_train" if version == "v1.0-mini" else "train"])
    maps = {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"samples": 0, "with_map": 0, "map_vectors": 0,
              "max_map_vectors_per_frame": 0, "map_points_outside_grid": 0,
              "detection_boxes": 0, "max_detection_boxes_per_frame": 0,
              "detection_boxes_outside_grid_removed": 0,
              "detection_boxes_without_sensor_points_removed": 0,
              "occupancy_labels": 0, "traffic_state_labels": 0,
              "map_grid": {key: grid[key] for key in ("x", "y")},
              "map_label_revision": "union-outline-v1"}
    with output.open("w") as stream:
        ordered = sorted(nusc.sample, key=lambda sample: (sample["scene_token"], sample["timestamp"]))
        for sample in ordered[:limit]:
            scene = nusc.get("scene", sample["scene_token"])
            log = nusc.get("log", scene["log_token"])
            # Camera reference pose; compensates each sensor's actual capture timestamp.
            ref = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            pose = nusc.get("ego_pose", ref["ego_pose_token"])
            ego_to_global = transform_matrix(pose["translation"], pose["rotation"])
            global_to_ego = np.linalg.inv(ego_to_global)
            cameras = []
            for name in CAMERAS:
                sd = nusc.get("sample_data", sample["data"][name])
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                ep = nusc.get("ego_pose", sd["ego_pose_token"])
                cam_to_ref = global_to_ego @ transform_matrix(ep["translation"], ep["rotation"]) @ transform_matrix(cs["translation"], cs["rotation"])
                intrinsic = np.eye(4)
                intrinsic[:3, :3] = cs["camera_intrinsic"]
                filename = root / sd["filename"]
                if not filename.is_file():
                    raise FileNotFoundError(filename)
                cameras.append({"name": name, "path": str(filename), "timestamp_us": sd["timestamp"],
                                "width": sd["width"], "height": sd["height"],
                                "intrinsic": cs["camera_intrinsic"], "camera_to_ego": cam_to_ref.tolist(),
                                "ego_to_image": (intrinsic @ np.linalg.inv(cam_to_ref)).tolist()})
            boxes = []
            for token in sample["anns"]:
                ann = nusc.get("sample_annotation", token)
                category = category_to_detection_name(ann["category_name"])
                if category is None:
                    continue
                if int(ann.get("num_lidar_pts", 0))+int(ann.get("num_radar_pts", 0)) <= 0:
                    counts["detection_boxes_without_sensor_points_removed"] += 1
                    continue
                rotation = global_to_ego[:3, :3] @ transform_matrix([0, 0, 0], ann["rotation"])[:3, :3]
                velocity = nusc.box_velocity(token)
                vel_valid = bool(np.isfinite(velocity).all())
                velocity = global_to_ego[:3, :3] @ np.nan_to_num(velocity)
                center = transform_points([ann["translation"]], global_to_ego)[0]
                if not all(grid[axis][0] <= center[index] <= grid[axis][1]
                           for index, axis in enumerate(("x", "y", "z"))):
                    counts["detection_boxes_outside_grid_removed"] += 1
                    continue
                boxes.append({"id": ann["instance_token"], "kind": category,
                              "center": center.tolist(),
                              "size": [ann["size"][1], ann["size"][0], ann["size"][2]],
                              "yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                              "velocity": velocity[:2].tolist(), "velocity_valid": vel_valid, "score": 1.0})
            location = log["location"]
            expansion = root / "maps" / "expansion" / f"{location}.json"
            if expansion.exists() and location not in maps:
                maps[location] = NuScenesMap(dataroot=str(root), map_name=location)
            vectors = extract_map(maps[location], ego_to_global, grid, points=map_points) if location in maps else None
            # Dense occupancy and signal states are not part of base nuScenes labels.
            occ_path = root.parent / "occupancy" / scene["name"] / sample["token"] / "labels.npz"
            row = {"schema_version": "1.2", "map_label_revision": "union-outline-v1",
                   "sample_token": sample["token"], "scene_token": sample["scene_token"],
                   "scene_name": scene["name"], "timestamp_us": ref["timestamp"],
                   "split": "train" if scene["name"] in train_names else "val",
                   "ego_to_global": ego_to_global.tolist(), "cameras": cameras,
                   "targets": {"boxes": boxes, "map_vectors": vectors,
                               "occupancy_path": str(occ_path) if occ_path.exists() else None,
                               "traffic_lights": None}}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            counts["samples"] += 1
            counts["with_map"] += vectors is not None
            counts["detection_boxes"] += len(boxes)
            counts["max_detection_boxes_per_frame"] = max(
                counts["max_detection_boxes_per_frame"], len(boxes))
            if vectors is not None:
                counts["map_vectors"] += len(vectors)
                counts["max_map_vectors_per_frame"] = max(counts["max_map_vectors_per_frame"], len(vectors))
                for vector in vectors:
                    xy = np.asarray(vector["points"])
                 
                    
                       
                   counts["map_points_outside_grid"] += int(
                        ((xy[:, 0] < grid["x"][0]-1e-5) | (xy[:, 0] > grid["x"][1]+1e-5) |
                         (xy[:, 1] < grid["y"][0]-1e-5) | (xy[:, 1] > grid["y"][1]+1e-5)).sum())
            counts["occupancy_labels"] += occ_path.exists()
    output.with_suffix(".report.json").write_text(json.dumps(counts, indent=2))
    return counts


def camera_tensor(row, image_size):
    import torch
    from PIL import Image
    height, width = image_size
    images, projections = [], []
    for camera in row["cameras"]:
        with Image.open(camera["path"]) as image:
            rgb = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32).copy()/255
        images.append(torch.from_numpy(rgb).permute(2, 0, 1))
        projection = np.asarray(camera["ego_to_image"], dtype=np.float32).copy()
        projection[0] *= width/camera["width"]
        projection[1] *= height/camera["height"]
        projections.append(projection)
    images = torch.stack(images)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    return (images-mean)/std, torch.tensor(np.stack(projections))
