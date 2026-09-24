"""Geometry validation and duplicate suppression for 3D detections."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon


def class_threshold(settings, kind, default=.35):
    value = settings.get("detection_confidence", settings.get("confidence", default))
    return float(value.get(kind, default) if isinstance(value, dict) else value)


def _corners(box):
    center = np.asarray(box["center"][:2], dtype=float)
    half = np.asarray(box["size"][:2], dtype=float)/2
    local = np.array([[-half[0], -half[1]], [-half[0], half[1]],
                      [half[0], half[1]], [half[0], -half[1]]])
    cosine, sine = np.cos(box["yaw"]), np.sin(box["yaw"])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local@rotation.T+center


def bev_iou(first, second):
    a, b = Polygon(_corners(first)), Polygon(_corners(second))
    if not a.is_valid or not b.is_valid:
        return 0.0
    union = a.union(b).area
    return float(a.intersection(b).area/union) if union > 0 else 0.0


def postprocess_detections(candidates, config):
    """Apply per-class confidence, physical bounds and same-class BEV NMS."""
    compiler, grid = config["compiler"], config["grid"]
    geometry = compiler.get("detection_geometry", {})
    minimum = np.asarray(geometry.get("min_size", [.1, .1, .1]), dtype=float)
    maximum = np.asarray(geometry.get("max_size", [20., 8., 8.]), dtype=float)
    max_speed = float(geometry.get("max_speed", 45.0))
    iou_threshold = float(geometry.get("nms_iou", .25))
    center_threshold = float(geometry.get("nms_center_distance", 1.0))
    accepted = []
    for source in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if float(source["score"]) < class_threshold(compiler, source["kind"]):
            continue
        item = deepcopy(source)
        center = np.asarray(item["center"], dtype=float)
        size = np.asarray(item["size"], dtype=float)
        velocity = np.asarray(item.get("velocity", [0., 0.]), dtype=float)
        values = np.r_[center, size, velocity, item["yaw"], item["score"]]
        if not np.isfinite(values).all() or (size < minimum).any() or (size > maximum).any():
            continue
        if not all(grid[axis][0] <= center[index] <= grid[axis][1]
                   for index, axis in enumerate(("x", "y", "z"))):
            continue
        if np.linalg.norm(velocity) > max_speed:
            continue
        duplicate = False
        for other in accepted:
            if other["kind"] != item["kind"]:
                continue
            distance = np.linalg.norm(center[:2]-np.asarray(other["center"][:2]))
            if distance < center_threshold or bev_iou(item, other) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            accepted.append(item)
    return accepted
