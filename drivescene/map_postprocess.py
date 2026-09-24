"""Geometry-aware vector-map filtering and ego-motion compensated temporal fusion."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.geometry import LineString

from .geometry import transform_points


def symmetric_chamfer(first, second):
    distances = cdist(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    return float((distances.min(0).mean()+distances.min(1).mean())/2)


def smooth_polyline(points, passes=1):
    points = np.asarray(points, dtype=float).copy()
    for _ in range(max(int(passes), 0)):
        if len(points) > 2:
            updated = points.copy()
            updated[1:-1] = .25*points[:-2]+.5*points[1:-1]+.25*points[2:]
            points = updated
    return points


def _setting(settings, name, kind, default):
    value = settings.get(name, default)
    return float(value.get(kind, default) if isinstance(value, dict) else value)


def _valid_geometry(points, kind, grid, settings):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return False
    tolerance = 1e-4
    if ((points[:, 0] < grid["x"][0]-tolerance).any() or
            (points[:, 0] > grid["x"][1]+tolerance).any() or
            (points[:, 1] < grid["y"][0]-tolerance).any() or
            (points[:, 1] > grid["y"][1]+tolerance).any()):
        return False
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    total = float(lengths.sum())
    if total < _setting(settings, "min_length", kind, 1.0):
        return False
    if total > _setting(settings, "max_length", kind, 85.0):
        return False
    if len(lengths) and lengths.max() > _setting(settings, "max_step", kind, 6.0):
        return False
    valid = lengths > 1e-4
    directions = segments[valid]/lengths[valid, None]
    if len(directions) > 1:
        cosine = np.clip((directions[:-1]*directions[1:]).sum(1), -1, 1)
        turns = np.degrees(np.arccos(cosine))
        if turns.max(initial=0) > _setting(settings, "max_turn_degrees", kind, 150.0):
            return False
    return bool(LineString(points).is_simple)


def postprocess_map(candidates, config):
    """Apply class thresholds, smoothing, geometry checks and same-class NMS."""
    compiler = config["compiler"]
    thresholds = compiler.get("map_confidence", {})
    geometry = compiler.get("map_geometry", {})
    default_threshold = float(compiler.get("confidence", .35))
    accepted = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        threshold = float(thresholds.get(candidate["kind"], default_threshold))
        if candidate["score"] < threshold:
            continue
        vector = deepcopy(candidate)
        points = smooth_polyline(vector["points"], geometry.get("smoothing_passes", 0))
        if not _valid_geometry(points, vector["kind"], config["grid"], geometry):
            continue
        vector["points"] = points.tolist()
        duplicate_threshold = float(geometry.get("duplicate_chamfer", 1.0))
        if any(other["kind"] == vector["kind"] and
               symmetric_chamfer(other["points"], vector["points"]) < duplicate_threshold
               for other in accepted):
            continue
        accepted.append(vector)
    return accepted


class MapTemporalStabilizer:
    """Fuse vector instances in global coordinates, then return current ego coordinates."""
    def __init__(self, config):
        self.config = config
        self.settings = config["compiler"].get("map_temporal", {})
        self.thresholds = config["compiler"].get("map_confidence", {})
        self.default_threshold = float(config["compiler"].get("confidence", .35))
        self.scene = None
        self.tracks = []
        self.next_id = 0

    def reset(self, scene=None):
        self.scene, self.tracks = scene, []

    @staticmethod
    def _to_global(points, transform):
        points = np.asarray(points, dtype=float)
        xyz = np.column_stack([points, np.zeros(len(points))])
        return transform_points(xyz, transform)[:, :2]

    def update(self, vectors, ego_to_global, scene):
        if not self.settings.get("enabled", False):
            return vectors
        if scene != self.scene:
            self.reset(scene)
        ego_to_global = np.asarray(ego_to_global, dtype=float)
        global_to_ego = np.linalg.inv(ego_to_global)
        current = []
        for vector in vectors:
            item = deepcopy(vector)
            item["global_points"] = self._to_global(item["points"], ego_to_global)
            current.append(item)

        costs = np.full((len(current), len(self.tracks)), 1e6, dtype=float)
        for i, item in enumerate(current):
            for j, track in enumerate(self.tracks):
                if item["kind"] == track["kind"]:
                    costs[i, j] = symmetric_chamfer(item["global_points"], track["global_points"])
        matched_current, matched_tracks = set(), set()
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols):
                if costs[row, col] > float(self.settings.get("match_chamfer", 2.0)):
                    continue
                item, track = current[row], self.tracks[col]
                new_points = item["global_points"]
                old_points = track["global_points"]
                if np.linalg.norm(new_points-old_points[::-1], axis=1).mean() < np.linalg.norm(new_points-old_points, axis=1).mean():
                    new_points = new_points[::-1]
                alpha = float(self.settings.get("ema", .65))
                track["global_points"] = alpha*new_points+(1-alpha)*old_points
                track["score"] = alpha*float(item["score"])+(1-alpha)*float(track["score"])
                track["age"] = 0
                track["hits"] += 1
                matched_current.add(row); matched_tracks.add(col)

        for index, item in enumerate(current):
            if index in matched_current:
                continue
            self.tracks.append({"id": f"map:{self.next_id}", "kind": item["kind"],
                                "score": float(item["score"]), "global_points": item["global_points"],
                                "age": 0, "hits": 1})
            self.next_id += 1
        decay = float(self.settings.get("score_decay", .85))
        max_age = int(self.settings.get("max_age", 2))
        original_count = len(self.tracks)-sum(index not in matched_current for index in range(len(current)))
        for index, track in enumerate(self.tracks[:original_count]):
            if index not in matched_tracks:
                track["age"] += 1
                track["score"] *= decay
        self.tracks = [track for track in self.tracks if track["age"] <= max_age]

        output = []
        minimum_hits = int(self.settings.get("min_hits", 1))
        for track in self.tracks:
            threshold = float(self.thresholds.get(track["kind"], self.default_threshold))
            if track["score"] < threshold or track["hits"] < minimum_hits:
                continue
            points = self._to_global(track["global_points"], global_to_ego)
            output.append({"id": track["id"], "kind": track["kind"],
                           "score": float(track["score"]), "points": points.tolist(),
                           "temporal_age": int(track["age"]), "temporal_hits": int(track["hits"])})
        # Ego-motion can make multiple historical tracks converge onto the same
        # current polyline. Re-run geometry checks and NMS after temporal fusion.
        output = postprocess_map(output, self.config)
        maximum = int(self.settings.get("max_vectors", 0))
        return output[:maximum] if maximum > 0 else output
