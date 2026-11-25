from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from drivescene.geometry import transform_matrix, transform_points, project
from drivescene.compiler import compile_scene, associate_lights
from drivescene.tracking import CenterTracker
from drivescene.occupancy import _trace_free_voxels


@pytest.fixture
def config():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    cfg["grid"] = {"x": [-4, 4], "y": [-4, 4], "z": [-2, 6], "shape": [8, 8, 8]}
    return cfg


def frame():
    return {"sample_token": "test", "scene_token": "scene", "timestamp_us": 1_000_000,
            "ego_to_global": np.eye(4).tolist(), "provenance": {"mode": "fixture"}, "agents": [], "map_vectors": []}


def detection(x=0, score=.9):
    return {"center": [x, 0, 0], "size": [4, 2, 2], "yaw": 0., "kind": "car", "score": score, "velocity": [0., 0.]}


def test_transform_roundtrip_and_projection():
    transform = transform_matrix([10, -4, 2], [np.sqrt(.5), 0, 0, np.sqrt(.5)])
    points = np.array([[1., 2., 3.], [4., 5., 6.]])
    np.testing.assert_allclose(transform_points(transform_points(points, transform), np.linalg.inv(transform)), points, atol=1e-9)
    uv, valid = project([[2, 4, 2], [1, 1, -1]], np.eye(4))
    np.testing.assert_allclose(uv[0], [1, 2])
    assert valid.tolist() == [True, False]


def test_missing_occupancy_never_free(config):
    world, arrays = compile_scene(frame(), config)
    assert world["occupancy"]["unknown_fraction"] == 1
    assert np.allclose(arrays["cost"], .7)
    assert not world["validated_for_planning"]


def test_agent_dedup_is_three_dimensional(config):
    sample = frame()
    sample["agents"] = [detection()]
    semantic = np.ones(config["grid"]["shape"], dtype=np.uint8)
    world, arrays = compile_scene(sample, config, semantic)
    assert world["occupancy"]["deduplicated_voxels"] > 0
    # Overhead voxels above the agent must remain static occupied.
    assert arrays["static_occupied"][4, 4, 6]
    assert not arrays["static_occupied"][4, 4, 2]
    assert arrays["cost"][4, 4] == 1


def test_ambiguous_lights_do_not_get_arbitrary_stopline():
    vectors = [{"id": "a", "kind": "stop_line", "points": [[3, -2], [3, 2]]},
               {"id": "b", "kind": "stop_line", "points": [[4, -2], [4, 2]]}]
    light = {"position_ego": [6, 0, 4], "state": "red"}
    result = associate_lights([light], vectors, 10)[0]
    assert result["stop_line_id"] is None
    assert result["association_status"] == "ambiguous"


def test_tracker_compensates_ego_motion(config):
    tracker = CenterTracker(config["tracking"])
    first = tracker.update([detection(10)], 1_000_000, np.eye(4), "a")
    moved_ego = np.eye(4)
    moved_ego[0, 3] = 1
    second = tracker.update([detection(9)], 1_500_000, moved_ego, "a")
    assert first[0]["id"] == second[0]["id"]
    np.testing.assert_allclose(second[0]["velocity"], [0, 0], atol=1e-4)
    with pytest.raises(ValueError, match="timestamp"):
        tracker.update([], 1_500_000, moved_ego, "a")
    third = tracker.update([detection(2)], 100, np.eye(4), "b")
    assert third[0]["id"] != first[0]["id"]


def test_invalid_voxels_rejected(config):
    with pytest.raises(ValueError, match="class mismatch"):
        compile_scene(frame(), config, np.full(config["grid"]["shape"], 17))


def test_lidar_ray_marks_free_before_surface_only():
    indices = _trace_free_voxels(
        np.array([.5, .5, .5]), np.array([[4.5, .5, .5]]),
        np.array([0., 0., 0.]), np.array([5., 1., 1.]), np.array([5, 1, 1]))
    assert indices.tolist() == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]


def test_unknown_cost_scales_with_observed_fraction(config):
    semantic = np.full(config["grid"]["shape"], 255, dtype=np.uint8)
    semantic[4, 4, 1] = 0
    _, arrays = compile_scene(frame(), config, semantic)
    assert 0 < float(arrays["cost"][4, 4]) < 0.7


def test_center_tracker_keeps_non_overlapping_pedestrian(config):
    tracker = CenterTracker(config["tracking"])
    pedestrian = {"center": [0.0, 0.0, 0.0], "size": [0.6, 0.6, 1.7], "yaw": 0.0,
                  "kind": "pedestrian", "score": 0.9, "velocity": [1.6, 0.0]}
    first = tracker.update([pedestrian], 0, np.eye(4), "scene")
    moved = dict(pedestrian, center=[0.8, 0.0, 0.0])
    second = tracker.update([moved], 500_000, np.eye(4), "scene")
    assert first[0]["id"] == second[0]["id"]
