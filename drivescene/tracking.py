from __future__ import annotations
from copy import deepcopy
import numpy as np
from .geometry import transform_points

# CenterPoint nuScenes association radii (m) at 2 Hz keyframes.
DEFAULT_MAX_DISTANCE = {"car": 4.0, "truck": 4.0, "construction_vehicle": 4.0, "bus": 5.5,
                        "trailer": 3.0, "barrier": 1.0, "motorcycle": 13.0, "bicycle": 3.0,
                        "pedestrian": 1.0, "traffic_cone": 1.0}


class CenterTracker:
    """Greedy per-class 3D tracking on global box centers (CenterPoint style).

    Tracks are propagated with a constant-velocity model, so small fast agents
    such as pedestrians remain associated even when consecutive 2 Hz boxes do
    not overlap. Velocity blends tracked global displacement with the previous
    estimate; the first observation keeps the detection-head velocity.
    """
    def __init__(self, config):
        self.max_distance = {**DEFAULT_MAX_DISTANCE, **config.get("max_center_distance", {})}
        self.max_age = int(config.get("max_age", 3))
        self.min_score = float(config.get("min_score", 0.1))
        self.new_track_thresh = float(config.get("new_track_thresh", 0.5))
        self.velocity_ema = float(config.get("velocity_ema", 0.5))
        self.reset()

    def reset(self):
        self.tracks, self.next_id = [], 0
        self.scene, self.timestamp = None, None

    def update(self, detections, timestamp_us, ego_to_global, scene):
        if self.scene != scene:
            self.reset()
            self.scene = scene
        if self.timestamp is not None and timestamp_us <= self.timestamp:
            raise ValueError("Tracker requires strictly increasing timestamps within a scene")
        dt = 0.0 if self.timestamp is None else (timestamp_us-self.timestamp)/1e6
        self.timestamp = timestamp_us
        transform = np.asarray(ego_to_global, dtype=float)
        rotation = transform[:3, :3]
        for track in self.tracks:
            track["elapsed"] += dt
            track["predicted"] = track["position"][:2]+track["velocity"]*track["elapsed"]

        candidates = sorted((d for d in detections if float(d["score"]) >= self.min_score),
                            key=lambda d: float(d["score"]), reverse=True)
        matched, output = set(), []
        for detection in candidates:
            position = transform_points([detection["center"]], transform)[0]
            limit = float(self.max_distance.get(detection["kind"], 2.0))
            best, best_distance = None, limit
            for index, track in enumerate(self.tracks):
                if index in matched or track["kind"] != detection["kind"]:
                    continue
                distance = float(np.linalg.norm(position[:2]-track["predicted"]))
                if distance <= best_distance:
                    best, best_distance = index, distance
            item = deepcopy(detection)
            if best is not None:
                track = self.tracks[best]
                matched.add(best)
                measured = (position[:2]-track["position"][:2])/max(track["elapsed"], 1e-6)
                track["velocity"] = (self.velocity_ema*measured+
                                     (1-self.velocity_ema)*track["velocity"])
                track.update(position=position, age=0, elapsed=0.0, hits=track["hits"]+1)
                item["velocity_source"] = "tracked_global_displacement"
            elif float(detection["score"]) >= self.new_track_thresh:
                head = np.asarray(detection.get("velocity", [0.0, 0.0])[:2], dtype=float)
                track = {"id": f"{scene}:{detection['kind']}:{self.next_id}",
                         "kind": detection["kind"], "position": position, "age": 0,
                         "elapsed": 0.0, "hits": 1,
                         "velocity": (rotation@np.r_[head, 0.0])[:2]}
                self.next_id += 1
                self.tracks.append(track)
                matched.add(len(self.tracks)-1)
                item["velocity_source"] = "detection_head" if "velocity" in detection else "unavailable"
            else:
                continue
            item["id"] = track["id"]
            item["velocity"] = (rotation.T@np.r_[track["velocity"], 0.0])[:2].tolist()
            item["track_hits"] = int(track["hits"])
            output.append(item)

        for index, track in enumerate(self.tracks):
            if index not in matched:
                track["age"] += 1
        self.tracks = [track for track in self.tracks if track["age"] <= self.max_age]
        return output


BEVByteTracker = CenterTracker
