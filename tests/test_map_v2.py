from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from shapely.geometry import LineString

from drivescene.data import extract_map
from drivescene.geometry import transform_matrix
from drivescene.map_postprocess import MapTemporalStabilizer, postprocess_map
from drivescene.detection_postprocess import postprocess_detections
from drivescene.models import supervised_loss


class _FakeMap:
    def get_records_in_patch(self, patch, layers, mode="intersect"):
        result = {layer: [] for layer in layers}
        result["lane_divider"] = ["line"]
        return result

    def get(self, layer, token):
        return {"line_token": token}

    def extract_line(self, token):
        return LineString([[-100, 0], [100, 0]])


def _config():
    return yaml.safe_load(Path("configs/default.yaml").read_text())


def test_rotated_map_is_clipped_in_ego_grid():
    transform = transform_matrix([20, -10, 0], [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)])
    vectors = extract_map(_FakeMap(), transform,
                          {"x": [-30, 30], "y": [-30, 30]}, points=20)
    assert vectors
    points = np.concatenate([np.asarray(vector["points"]) for vector in vectors])
    assert (points[:, 0] >= -30-1e-6).all() and (points[:, 0] <= 30+1e-6).all()
    assert (points[:, 1] >= -30-1e-6).all() and (points[:, 1] <= 30+1e-6).all()


def test_geometry_filter_nms_and_temporal_ids():
    config = _config()
    points = np.column_stack([np.linspace(-10, 10, 20), np.zeros(20)])
    candidates = [
        {"id": "a", "kind": "divider", "score": .9, "points": points.tolist()},
        {"id": "b", "kind": "divider", "score": .8, "points": (points+[0, .1]).tolist()},
    ]
    filtered = postprocess_map(candidates, config)
    assert len(filtered) == 1
    stabilizer = MapTemporalStabilizer(config)
    first = stabilizer.update(filtered, np.eye(4), "scene")
    assert first == []  # tentative tracks need two observations
    moved = deepcopy(filtered)
    moved[0]["points"] = (np.asarray(moved[0]["points"])+[.2, 0]).tolist()
    second = stabilizer.update(moved, np.eye(4), "scene")
    assert len(second) == 1
    assert second[0]["temporal_hits"] == 2
    stable_id = second[0]["id"]
    third = stabilizer.update(moved, np.eye(4), "scene")
    assert third[0]["id"] == stable_id


def test_structured_map_losses_are_finite_and_differentiable():
    config = _config()
    query_count, point_count = 4, config["model"]["map_points"]
    logits = torch.randn(1, query_count, len(config["model"]["map_classes"])+1,
                         requires_grad=True)
    base = torch.stack([torch.linspace(-10, 10, point_count), torch.zeros(point_count)], -1)
    points = (base[None, None].repeat(1, query_count, 1, 1)+
              torch.randn(1, query_count, point_count, 2)*.05).requires_grad_()
    prediction = {"map": {"logits": logits, "points": points},
                  "detection": {"logits": torch.empty(1, 0, 1), "boxes": torch.empty(1, 0, 10)},
                  "occupancy": torch.empty(1, 3, 1, 1, 1)}
    connected = base+torch.tensor([20.0, 0.0])
    target = {"map_vectors": [{"kind": "divider", "points": base.tolist()},
                              {"kind": "divider", "points": connected.tolist()}],
              "boxes": None, "occupancy": None}
    loss = supervised_loss(prediction, [target], config)["map"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(points.grad).all()


def test_detection_nms_and_structured_loss():
    config = _config()
    config["model"]["detection_queries"] = 4
    candidates = [
        {"id": "a", "kind": "car", "score": .9, "center": [5., 0., 0.],
         "size": [4., 2., 1.5], "yaw": 0., "velocity": [0., 0.]},
        {"id": "b", "kind": "car", "score": .8, "center": [5.1, 0., 0.],
         "size": [4., 2., 1.5], "yaw": 0., "velocity": [0., 0.]},
    ]
    assert len(postprocess_detections(candidates, config)) == 1
    class_count = len(config["model"]["detection_classes"])+1
    logits = torch.randn(1, 4, class_count, requires_grad=True)
    boxes = torch.tensor([[[5., 0., 0., 4., 2., 1.5, 0., 1., 0., 0.],
                           [0., 5., 0., 1., 1., 2., 0., 1., 0., 0.],
                           [-5., 0., 0., 1., 1., 2., 0., 1., 0., 0.],
                           [0., -5., 0., 1., 1., 2., 0., 1., 0., 0.]]],
                         requires_grad=True)
    prediction = {"map": {"logits": torch.empty(1, 0, 1),
                            "points": torch.empty(1, 0, config["model"]["map_points"], 2)},
                  "detection": {"logits": logits, "boxes": boxes},
                  "occupancy": torch.empty(1, 3, 1, 1, 1)}
    target = {"map_vectors": None, "boxes": [{"kind": "car", "center": [5., 0., 0.],
              "size": [4., 2., 1.5], "yaw": 0., "velocity": [0., 0.],
              "velocity_valid": True}], "occupancy": None}
    loss = supervised_loss(prediction, [target], config)["detection"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(boxes.grad).all()
