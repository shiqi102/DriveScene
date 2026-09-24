from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from .geometry import grid_centers
from .detection_postprocess import class_threshold


def validate_frame(frame):
    for field in ("sample_token", "scene_token", "timestamp_us", "ego_to_global", "provenance"):
        if field not in frame:
            raise ValueError(f"Missing frame field: {field}")
    matrix = np.asarray(frame["ego_to_global"])
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid ego pose")
    for box in frame.get("agents", []):
        values = box["center"]+box["size"]+[box["yaw"], box["score"]]+box.get("velocity", [])
        if len(box["center"]) != 3 or len(box["size"]) != 3 or not np.isfinite(values).all() or min(box["size"]) <= 0:
            raise ValueError("Invalid 3D box")
    for vector in frame.get("map_vectors", []):
        points = np.asarray(vector["points"])
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Invalid vector map polyline")


def associate_lights(lights, vectors, max_distance):
    from shapely.geometry import Point, LineString
    stops = [v for v in vectors if v["kind"] == "stop_line"]
    output = []
    for source in lights:
        light = deepcopy(source)
        light["stop_line_id"] = None
        light["association_status"] = "unlocalized"
        if light.get("position_ego") is not None:
            position = np.asarray(light["position_ego"])
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("Invalid light position")
            candidates = sorted([(LineString(s["points"]).distance(Point(position[:2])), s["id"]) for s in stops])
            candidates = [c for c in candidates if c[0] <= max_distance]
            # A geometry-only heuristic must not pretend to know lane topology.
            if len(candidates) == 1:
                light["stop_line_id"] = candidates[0][1]
                light["association_status"] = "geometry_candidate_unverified"
            else:
                light["association_status"] = "ambiguous" if candidates else "no_candidate"
        output.append(light)
    return output


def compile_scene(frame, config, occupancy=None):
    """Conservative compiler: absent/unobserved occupancy stays unknown, never free.

    Occupancy convention: [X,Y,Z], 0 free, 1 occupied, 2 drivable, 255 unknown.
    Agent-covered occupied voxels are separated from static occupancy in 3D.
    """
    validate_frame(frame)
    cfg, grid = config["compiler"], config["grid"]
    shape = tuple(grid["shape"])
    semantic = np.full(shape, 255, dtype=np.uint8) if occupancy is None else np.asarray(occupancy).copy()
    if semantic.shape != shape or not np.isin(semantic, [0, 1, 2, 255]).all():
        raise ValueError("Occupancy shape/class mismatch; convert labels explicitly")
    map_confidence = cfg.get("map_confidence", {})
    default_confidence = float(cfg.get("confidence", .35))
    agents = [deepcopy(a) for a in frame.get("agents", [])
              if a["score"] >= class_threshold(cfg, a["kind"], default_confidence)]
    vectors = [deepcopy(v) for v in frame.get("map_vectors", [])
               if v["score"] >= float(map_confidence.get(v["kind"], default_confidence))]
    centers = grid_centers(grid)
    agent_mask = np.zeros(shape, dtype=bool)
    for agent in agents:
        xyz = centers-np.asarray(agent["center"])
        c, s = np.cos(agent["yaw"]), np.sin(agent["yaw"])
        local = np.stack([c*xyz[..., 0]+s*xyz[..., 1], -s*xyz[..., 0]+c*xyz[..., 1], xyz[..., 2]], -1)
        half = np.asarray(agent["size"])/2+cfg["agent_margin"]
        agent_mask |= (np.abs(local) <= half).all(-1)
    occupied = semantic == 1
    static_occupied = occupied & ~agent_mask
    # Only the vehicle-relevant vertical band contributes to ground navigation.
    zcenters = centers[0, 0, :, 2]
    band = (zcenters >= -1.0) & (zcenters <= 2.0)
    if not band.any():
        raise ValueError("Grid has no vehicle-height voxels")
    unknown = (semantic[:, :, band] == 255).any(-1)
    cost = np.where(unknown, cfg["unknown_cost"], 0).astype(np.float32)
    static_2d = static_occupied[:, :, band].any(-1)
    agent_2d = agent_mask[:, :, band].any(-1)
    cost[static_2d | agent_2d] = 1.0
    lights = associate_lights(frame.get("traffic_lights", []), vectors, cfg["light_max_distance"])
    source = frame["provenance"]
    world = {"schema_version": "1.0", "sample_token": frame["sample_token"], "scene_token": frame["scene_token"],
             "timestamp_us": int(frame["timestamp_us"]), "coordinate_frame": "ego",
             "axes": {"x": "forward", "y": "left", "z": "up"}, "units": {"position": "m", "velocity": "m/s", "yaw": "rad"},
             "ego_to_global": frame["ego_to_global"], "provenance": source,
             "validated_for_planning": False,
             "layers": {"vector_map": vectors, "agents": agents, "traffic_lights": lights},
             "occupancy": {"grid": grid, "axis_order": "XYZ", "classes": {"0": "free", "1": "occupied", "2": "drivable", "255": "unknown"},
                           "available": occupancy is not None, "unknown_fraction": float((semantic == 255).mean()),
                           "deduplicated_voxels": int((occupied & agent_mask).sum())},
             "limitations": ["Traffic-light geometric candidates are not verified lane-control associations.",
                             "Cost grid is a research output; missing occupancy remains unknown."]}
    diagnostics = {}
    if frame.get("map_candidates") is not None:
        diagnostics.update(map_candidates=deepcopy(frame["map_candidates"]),
                           map_candidates_postprocessed=False)
    if frame.get("detection_candidates") is not None:
        diagnostics.update(detection_candidates=deepcopy(frame["detection_candidates"]),
                           detection_candidates_postprocessed=False)
    if diagnostics:
        world["diagnostics"] = diagnostics
    arrays = {"semantic": semantic, "static_occupied": static_occupied, "agent_mask": agent_mask, "cost": cost}
    return world, arrays


def save_world(world, arrays, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    token = world["sample_token"]
    np.savez_compressed(directory / f"{token}.npz", **arrays)
    world = deepcopy(world)
    world["occupancy"]["artifact"] = f"{token}.npz"
    path = directory / f"{token}.json"
    path.write_text(json.dumps(world, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
