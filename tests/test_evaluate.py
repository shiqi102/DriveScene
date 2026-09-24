import numpy as np

from drivescene.evaluate import (_average_precision, _class_detections,
                                 _clipped_lines, _resample, chamfer_distance)


def test_polyline_resampling_and_grid_clip():
    line = _resample([[0, 0], [2, 0]], 5)
    assert line.shape == (5, 2)
    assert chamfer_distance(line, line) == 0
    clipped = _clipped_lines([[-40, 0], [40, 0]],
                             {"x": [-30, 30], "y": [-30, 30]})
    assert len(clipped) == 1
    assert np.allclose(clipped[0][[0, -1], 0], [-30, 30])


def test_map_matching_counts_duplicate_as_false_positive():
    points = _resample([[0, 0], [10, 0]])
    frames = [{"pred": {"divider": [
        {"points": points, "score": .9}, {"points": points, "score": .8}]},
        "gt": {"divider": [points]}}]
    tp, fp, count, _ = _class_detections(frames, "divider", 1.0)
    assert count == 1
    assert tp.tolist() == [1.0, 0.0]
    assert fp.tolist() == [0.0, 1.0]
    assert _average_precision(tp, fp, count) == 1.0
