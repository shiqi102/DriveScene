from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from .geometry import transform_points, box_corners_xy


class BEVByteTracker:
    """Official Ultralytics ByteTrack on per-class global XY bounding rectangles.

    Not native 3D ByteTrack: yaw/height remain from the matched 3D detection.
    Global coordinates remove ego motion; physical velocity uses real timestamps.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.trackers, self.history = {}, {}
        self.scene, self.timestamp = None, None

    def update(self, detections, timestamp_us, ego_to_global, scene):
        from ultralytics.trackers.byte_tracker import BYTETracker
        if self.scene != scene:
            self.reset()
            self.scene = scene
        if self.timestamp is not None and timestamp_us <= self.timestamp:
            raise ValueError("Tracker requires strictly increasing timestamps within a scene")
        self.timestamp = timestamp_us
        transform = np.asarray(ego_to_global)
        kinds = sorted(set(self.trackers) | {d["kind"] for d in detections})
        output = []
        for kind in kinds:
            if kind not in self.trackers:
                self.trackers[kind] = BYTETracker(SimpleNamespace(**self.config), frame_rate=2)
            items = [d for d in detections if d["kind"] == kind]
            boxes = []
            for item in items:
                corners = box_corners_xy(item["center"], item["size"], item["yaw"])
                corners = transform_points(np.column_stack([corners, np.full(4, item["center"][2])]), transform)[:, :2]
                lo, hi = corners.min(0), corners.max(0)
                boxes.append(np.r_[(lo+hi)/2, hi-lo])
            results = SimpleNamespace(xywh=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                conf=np.asarray([d["score"] for d in items]), cls=np.zeros(len(items)))
            tracks = self.trackers[kind].update(results)
            for track in tracks:
                index = int(track[-1])
                item = deepcopy(items[index])
                key = f"{scene}:{kind}:{int(track[4])}"
                global_position = transform_points([item["center"]], transform)[0]
                if key in self.history:
                    prev_time, prev_position = self.history[key]
                    global_velocity = (global_position-prev_position)/((timestamp_us-prev_time)/1e6)
                    item["velocity"] = (transform[:3, :3].T @ global_velocity)[:2].tolist()
                    item["velocity_source"] = "tracked_global_displacement"
                else:
                    item["velocity_source"] = "detection_head" if "velocity" in item else "unavailable"
                    item.setdefault("velocity", [0.0, 0.0])
                self.history[key] = (timestamp_us, global_position)
                item["id"] = key
                output.append(item)
        # Bound retained velocity history, independently of ByteTrack's track buffer.
        self.history = {k: v for k, v in self.history.items() if timestamp_us-v[0] < 30_000_000}
        return output
