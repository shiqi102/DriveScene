"""Validation metrics and qualitative error rendering for compiled WorldModels."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, box

from .data import read_manifest


MAP_CLASSES = ("divider", "centerline", "boundary", "pedestrian_crossing", "stop_line")
MAP_BENCHMARK_CLASSES = ("divider", "pedestrian_crossing", "boundary")
TRACKING_CLASSES = {"bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"}


def _resample(points, count=100):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    points = points[keep]
    if len(points) < 2:
        return None
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0.0, distance[-1], count)
    return np.column_stack([np.interp(samples, distance, points[:, axis]) for axis in range(2)])


def _clipped_lines(points, grid):
    """Clip supervision to the model's ego-frame output range before scoring."""
    geometry = LineString(np.asarray(points, dtype=np.float64)).intersection(
        box(grid["x"][0], grid["y"][0], grid["x"][1], grid["y"][1]))
    if geometry.is_empty:
        return []
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    output = []
    for part in parts:
        if part.geom_type in ("LineString", "LinearRing") and part.length > .2:
            sampled = _resample(part.coords)
            if sampled is not None:
                output.append(sampled)
    return output


def chamfer_distance(first, second):
    distances = cdist(first, second)
    return float((distances.min(axis=0).mean()+distances.min(axis=1).mean())/2)


def _average_precision(tp, fp, ground_truth_count):
    if ground_truth_count == 0 or len(tp) == 0:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp/ground_truth_count
    precision = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[0.0, precision, 0.0]
    for index in range(len(precision)-1, 0, -1):
        precision[index-1] = max(precision[index-1], precision[index])
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed+1]-recall[changed])*precision[changed+1]))


def _class_detections(frames, class_name, distance_threshold):
    records, ground_truth_count = [], 0
    distances_by_frame = {}
    for frame_index, frame in enumerate(frames):
        predictions = frame["pred"].get(class_name, [])
        targets = frame["gt"].get(class_name, [])
        ground_truth_count += len(targets)
        matrix = np.empty((len(predictions), len(targets)), dtype=np.float64)
        for i, prediction in enumerate(predictions):
            for j, target in enumerate(targets):
                matrix[i, j] = chamfer_distance(prediction["points"], target)
        distances_by_frame[frame_index] = matrix
        records.extend((float(prediction["score"]), frame_index, index)
                       for index, prediction in enumerate(predictions))
    records.sort(reverse=True)
    covered = {index: np.zeros(len(frame["gt"].get(class_name, [])), dtype=bool)
               for index, frame in enumerate(frames)}
    tp, fp, matched_distances = [], [], []
    for _, frame_index, prediction_index in records:
        matrix = distances_by_frame[frame_index]
        if matrix.shape[1] == 0:
            tp.append(0.0); fp.append(1.0)
            continue
        target_index = int(np.argmin(matrix[prediction_index]))
        distance = float(matrix[prediction_index, target_index])
        if distance <= distance_threshold and not covered[frame_index][target_index]:
            covered[frame_index][target_index] = True
            tp.append(1.0); fp.append(0.0); matched_distances.append(distance)
        else:
            tp.append(0.0); fp.append(1.0)
    return np.asarray(tp), np.asarray(fp), ground_truth_count, matched_distances


def _map_metrics(frames):
    thresholds = (0.5, 1.0, 1.5)
    classes = {}
    for class_name in MAP_CLASSES:
        aps = {}
        for threshold in thresholds:
            tp, fp, count, distances = _class_detections(frames, class_name, threshold)
            aps[str(threshold)] = _average_precision(tp, fp, count)
        tp, fp, count, distances = _class_detections(frames, class_name, 1.0)
        true_positive = int(tp.sum())
        false_positive = int(fp.sum())
        false_negative = count-true_positive
        precision = true_positive/max(true_positive+false_positive, 1)
        recall = true_positive/max(count, 1)
        classes[class_name] = {
            "ground_truth": count,
            "predictions": int(len(tp)),
            "ap_by_chamfer_m": aps,
            "ap": float(np.mean(list(aps.values()))),
            "f1_at_1m": 2*precision*recall/max(precision+recall, np.finfo(float).eps),
            "precision_at_1m": precision,
            "recall_at_1m": recall,
            "matched_chamfer_mean_m": float(np.mean(distances)) if distances else None,
        }
    official = [classes[name]["ap"] for name in MAP_BENCHMARK_CLASSES]
    limitations = [
        "Ground-truth lines are clipped to the configured ego-grid before evaluation.",
        "centerline and stop_line are project extensions and are excluded from benchmark map_mAP.",
    ]
    if not all(frame.get("unpruned_candidates", False) for frame in frames):
        limitations.insert(0, "Predictions were already pruned during inference; exact full-score AP requires unpruned map candidates.")
    return {
        "protocol": "MapTR-style symmetric Chamfer AP at 0.5/1.0/1.5m",
        "benchmark_classes": list(MAP_BENCHMARK_CLASSES),
        "map_mAP": float(np.mean(official)),
        "classes": classes,
        "limitations": limitations,
    }


def _frame_map_data(row, world, grid):
    predicted, target = defaultdict(list), defaultdict(list)
    diagnostics = world.get("diagnostics", {})
    source = diagnostics.get("map_candidates", world["layers"]["vector_map"])
    for vector in source:
        sampled = _resample(vector["points"])
        if sampled is not None and vector["kind"] in MAP_CLASSES:
            predicted[vector["kind"]].append({"points": sampled, "score": vector["score"],
                                               "raw": np.asarray(vector["points"], dtype=float)})
    for vector in row["targets"].get("map_vectors") or []:
        if vector["kind"] in MAP_CLASSES:
            target[vector["kind"]].extend(_clipped_lines(vector["points"], grid))
    return {"pred": dict(predicted), "gt": dict(target),
            "unpruned_candidates": "map_candidates" in diagnostics}


def _occupancy_metrics(rows, prediction_directory, class_names):
    intersections = np.zeros(len(class_names), dtype=np.int64)
    unions = np.zeros(len(class_names), dtype=np.int64)
    gt_counts = np.zeros(len(class_names), dtype=np.int64)
    pred_counts = np.zeros(len(class_names), dtype=np.int64)
    evaluated = unknown_predictions = frames = 0
    per_frame = {}
    for row in rows:
        target_path = row["targets"].get("occupancy_path")
        prediction_path = prediction_directory/f"{row['sample_token']}.npz"
        if not target_path or not Path(target_path).is_file() or not prediction_path.is_file():
            continue
        with np.load(target_path) as archive:
            target = archive["drivescene_semantic"]
        with np.load(prediction_path) as archive:
            prediction = archive["semantic"]
        if target.shape != prediction.shape:
            raise ValueError(f"Occupancy shape mismatch for {row['sample_token']}")
        mask = target != 255
        evaluated += int(mask.sum())
        unknown_predictions += int(((prediction == 255) & mask).sum())
        frame_ious = []
        for class_id in range(len(class_names)):
            gt = (target == class_id) & mask
            pred = (prediction == class_id) & mask
            intersection = int((gt & pred).sum())
            union = int((gt | pred).sum())
            intersections[class_id] += intersection
            unions[class_id] += union
            gt_counts[class_id] += int(gt.sum())
            pred_counts[class_id] += int(pred.sum())
            if gt.any():
                frame_ious.append(intersection/max(union, 1))
        per_frame[row["sample_token"]] = float(np.mean(frame_ious)) if frame_ious else None
        frames += 1
    classes, valid_ious, valid_f1 = {}, [], []
    for class_id, name in enumerate(class_names):
        tp = int(intersections[class_id])
        fp = int(pred_counts[class_id]-tp)
        fn = int(gt_counts[class_id]-tp)
        supported = gt_counts[class_id] > 0
        iou = tp/max(int(unions[class_id]), 1) if supported else None
        f1 = 2*tp/max(2*tp+fp+fn, 1) if supported else None
        classes[name] = {"support": int(gt_counts[class_id]), "tp": tp, "fp": fp,
                         "fn": fn, "iou": iou, "f1": f1}
        if supported:
            valid_ious.append(iou); valid_f1.append(f1)
    return {
        "protocol": "project sparse observed-voxel evaluation (GT != 255)",
        "official_occ3d": False,
        "frames": frames,
        "evaluated_voxels": evaluated,
        "unknown_prediction_fraction": unknown_predictions/max(evaluated, 1),
        "mIoU": float(np.mean(valid_ious)) if valid_ious else None,
        "macro_f1": float(np.mean(valid_f1)) if valid_f1 else None,
        "classes": classes,
        "per_frame_mIoU": per_frame,
        "limitations": [
            "These labels are sparse LiDARseg-derived voxels, not Occ3D dense semantics or mask_camera.",
            "Free IoU is defined only on observed free cells from ray tracing, not the full volume.",
        ],
    }


def _transform_box(agent, ego_to_global):
    from pyquaternion import Quaternion
    center = np.r_[np.asarray(agent["center"], dtype=float), 1.0]
    translation = (ego_to_global@center)[:3]
    c, s = np.cos(agent["yaw"]), np.sin(agent["yaw"])
    local_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    rotation = Quaternion(matrix=ego_to_global[:3, :3]@local_rotation).elements.tolist()
    velocity = ego_to_global[:3, :3]@np.r_[agent.get("velocity", [0, 0])[:2], 0.0]
    size = agent["size"]
    return translation.tolist(), [size[1], size[0], size[2]], rotation, velocity[:2].tolist()


def _attribute(name, velocity):
    moving = np.linalg.norm(velocity) > .2
    if name in {"car", "bus", "truck", "trailer", "construction_vehicle"}:
        return "vehicle.moving" if moving else "vehicle.parked"
    if name == "pedestrian":
        return "pedestrian.moving" if moving else "pedestrian.standing"
    if name in {"bicycle", "motorcycle"}:
        return "cycle.with_rider"
    return ""


def _nuscenes_submission(rows, worlds, task):
    results = {}
    for row, world in zip(rows, worlds):
        transform = np.asarray(world["ego_to_global"], dtype=float)
        boxes = []
        source = world["layers"]["agents"]
        if task == "detection":
            source = world.get("diagnostics", {}).get("detection_candidates", source)
        for agent in source:
            name = agent["kind"]
            if task == "tracking" and name not in TRACKING_CLASSES:
                continue
            translation, size, rotation, velocity = _transform_box(agent, transform)
            common = {"sample_token": row["sample_token"], "translation": translation,
                      "size": size, "rotation": rotation, "velocity": velocity}
            if task == "detection":
                common.update(detection_name=name, detection_score=float(agent["score"]),
                              attribute_name=_attribute(name, velocity))
            else:
                common.update(tracking_name=name, tracking_score=float(agent["score"]),
                              tracking_id=str(agent["id"]))
            boxes.append(common)
        results[row["sample_token"]] = boxes
    return {"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                     "use_map": False, "use_external": False}, "results": results}


def _official_nuscenes(rows, worlds, data_root, version, split, output):
    if split != "val":
        unavailable = {"available": False,
                       "reason": "Official nuScenes evaluation is run on the validation split."}
        return unavailable, dict(unavailable)
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.tracking.evaluate import TrackingEval
    # nuscenes-devkit 1.1.11 expects the pre-1.4 motmetrics list storage,
    # while Python 3.10 requires motmetrics 1.4, which stores columns in dicts.
    # Adapt the dataframe constructor without changing either installed package.
    from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
    original_event_dataframe = MOTAccumulatorCustom.new_event_dataframe_with_data
    def compatible_event_dataframe(indices, events):
        if isinstance(indices, dict):
            import pandas as pd
            index = pd.MultiIndex.from_arrays(
                [indices["FrameId"], indices["Event"]], names=["FrameId", "Event"])
            return pd.DataFrame(events, index=index, columns=["Type", "OId", "HId", "D"])
        return original_event_dataframe(indices, events)
    MOTAccumulatorCustom.new_event_dataframe_with_data = staticmethod(compatible_event_dataframe)
    eval_set = "mini_val" if version == "v1.0-mini" else "val"
    output.mkdir(parents=True, exist_ok=True)
    detection_path = output/"detection_submission.json"
    tracking_path = output/"tracking_submission.json"
    detection_submission = _nuscenes_submission(rows, worlds, "detection")
    tracking_submission = _nuscenes_submission(rows, worlds, "tracking")
    detection_path.write_text(json.dumps(detection_submission), encoding="utf-8")
    tracking_path.write_text(json.dumps(tracking_submission), encoding="utf-8")
    try:
        nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
        detection_dir = output/"detection"
        detection = NuScenesEval(nusc, config_factory("detection_cvpr_2019"),
                                 str(detection_path), eval_set, str(detection_dir), verbose=False)
        detection_summary = detection.main(plot_examples=0, render_curves=False)
    except Exception as error:
        detection_summary = {"available": False,
                             "reason": f"Official detection evaluation failed: {type(error).__name__}: {error}"}
    if not any(tracking_submission["results"].values()):
        # nuscenes-devkit raises "Invalid box type: None" for a completely
        # empty submission. Its unambiguous degenerate result is zero recall.
        tracking_summary = {"amota": 0.0, "amotp": 2.0, "recall": 0.0,
                            "motar": 0.0, "mota": 0.0, "motp": 2.0,
                            "tp": 0.0, "fp": 0.0, "fn": None,
                            "ids": 0.0, "frag": 0.0,
                            "empty_submission": True,
                            "reason": "No post-tracker agent predictions on the validation split."}
    else:
        try:
            tracking_dir = output/"tracking"
            tracking = TrackingEval(config_factory("tracking_nips_2019"), str(tracking_path), eval_set,
                                    str(tracking_dir), version, str(data_root), verbose=False)
            tracking_summary = tracking.main(render_curves=False)
        except Exception as error:
            tracking_summary = {"available": False,
                                "reason": f"Official tracking evaluation failed: {type(error).__name__}: {error}"}
    return detection_summary, tracking_summary


def _json_safe(value):
    """Replace numpy scalars and non-finite official metric values for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _match_map_frame(frame, threshold=1.0):
    pred_status, gt_status = {}, {}
    tp = fp = fn = 0
    for class_name in MAP_CLASSES:
        predictions = frame["pred"].get(class_name, [])
        targets = frame["gt"].get(class_name, [])
        matrix = np.empty((len(predictions), len(targets)))
        for i, prediction in enumerate(predictions):
            for j, target in enumerate(targets):
                matrix[i, j] = chamfer_distance(prediction["points"], target)
        covered = np.zeros(len(targets), dtype=bool)
        for i in np.argsort([-item["score"] for item in predictions]):
            if len(targets):
                j = int(np.argmin(matrix[i]))
                if matrix[i, j] <= threshold and not covered[j]:
                    covered[j] = True; pred_status[(class_name, i)] = "tp"; tp += 1; continue
            pred_status[(class_name, i)] = "fp"; fp += 1
        for j, matched in enumerate(covered):
            gt_status[(class_name, j)] = "tp" if matched else "fn"
            fn += int(not matched)
    precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
    return pred_status, gt_status, 2*precision*recall/max(precision+recall, np.finfo(float).eps)


def _detection_recall(row, world, threshold=2.0):
    predictions = world["layers"]["agents"]
    targets = row["targets"].get("boxes") or []
    matched = np.zeros(len(targets), dtype=bool)
    true_positive = 0
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        candidates = [(np.linalg.norm(np.asarray(prediction["center"][:2])-np.asarray(target["center"][:2])), index)
                      for index, target in enumerate(targets)
                      if target["kind"] == prediction["kind"] and not matched[index]]
        if candidates:
            distance, index = min(candidates)
            if distance <= threshold:
                matched[index] = True; true_positive += 1
    return true_positive/max(len(targets), 1)


def _render_error_video(rows, worlds, map_frames, occupancy, output, fps):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    status_colors = {"tp": "#3ddc84", "fp": "#ff4d5a", "fn": "#4da3ff"}
    with tempfile.TemporaryDirectory(prefix="drivescene-eval-") as temporary:
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8,
                                macro_block_size=2,
                                ffmpeg_log_level="error",
                                ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for index, (row, world, frame) in enumerate(zip(rows, worlds, map_frames)):
                pred_status, gt_status, map_f1 = _match_map_frame(frame)
                fig = plt.figure(figsize=(16, 9), facecolor="#111318")
                layout = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1], height_ratios=[1, 1])
                ax = fig.add_subplot(layout[:, :2], facecolor="#181b22")
                camera_ax = fig.add_subplot(layout[0, 2]); info_ax = fig.add_subplot(layout[1, 2], facecolor="#181b22")
                for class_name in MAP_CLASSES:
                    for item in frame["gt"].get(class_name, []):
                        ax.plot(item[:, 1], item[:, 0], color="#ffd54a", lw=.7, alpha=.18, ls="--")
                    for item in frame["pred"].get(class_name, []):
                        raw = item["raw"]
                        ax.plot(raw[:, 1], raw[:, 0], color="white", lw=.6, alpha=.16, ls=":")
                    for item_index, item in enumerate(frame["pred"].get(class_name, [])):
                        raw = item["raw"]
                        ax.plot(raw[:, 1], raw[:, 0], color=status_colors[pred_status[(class_name, item_index)]], lw=1.8)
                    for item_index, item in enumerate(frame["gt"].get(class_name, [])):
                        if gt_status[(class_name, item_index)] == "fn":
                            ax.plot(item[:, 1], item[:, 0], color=status_colors["fn"], lw=1.8)
                ax.add_patch(plt.Rectangle((-.95, -2.3), 1.9, 4.6, color="#4169e1", alpha=.9))
                ax.set(xlim=(-30, 30), ylim=(-30, 30), xlabel="ego y (left), m", ylabel="ego x (forward), m")
                ax.grid(color="#38404d", alpha=.45); ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
                ax.set_title(f"{row['scene_name']}  {row['sample_token'][:8]}  |  vector-map error", color="white")
                ax.legend(handles=[Line2D([0], [0], color=status_colors["tp"], label="TP"),
                                   Line2D([0], [0], color=status_colors["fp"], label="FP"),
                                   Line2D([0], [0], color=status_colors["fn"], label="FN"),
                                   Line2D([0], [0], color="#ffd54a", ls="--", label="GT"),
                                   Line2D([0], [0], color="white", ls=":", label="prediction")],
                          loc="upper right", framealpha=.3, labelcolor="white")
                front = next((camera for camera in row["cameras"] if camera["name"] == "CAM_FRONT"), row["cameras"][0])
                with Image.open(front["path"]) as image:
                    camera_ax.imshow(image.convert("RGB"))
                camera_ax.set_title("CAM_FRONT", color="white"); camera_ax.axis("off")
                detection_recall = _detection_recall(row, world)
                occ_iou = occupancy["per_frame_mIoU"].get(row["sample_token"])
                info_ax.axis("off")
                info_ax.text(.06, .9, "Validation diagnostics", color="white", fontsize=18, weight="bold")
                info_ax.text(.06, .68, f"Map F1 @ 1m       {map_f1:.3f}", color="#3ddc84", fontsize=16)
                info_ax.text(.06, .53, f"Detection recall   {detection_recall:.3f}", color="#e8e8e8", fontsize=16)
                info_ax.text(.06, .38, f"Occupancy mIoU     {occ_iou:.3f}" if occ_iou is not None else "Occupancy mIoU     N/A", color="#e8e8e8", fontsize=16)
                info_ax.text(.06, .23, f"Active tracks      {len(world['layers']['agents'])}", color="#e8e8e8", fontsize=16)
                info_ax.text(.06, .08, "Green TP   Red FP   Blue FN", color="#aeb6c3", fontsize=13)
                fig.suptitle("DriveScene validation — diagnostic output, not planning validated", color="white", fontsize=14)
                fig.tight_layout()
                path = Path(temporary)/f"{index:06d}.png"
                fig.savefig(path, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)
                writer.append_data(imageio.imread(path))
    return {"path": str(output), "frames": len(rows), "fps": fps, "duration_seconds": len(rows)/fps}


def _markdown(report):
    lines = ["# DriveScene validation report", "", f"- Split: `{report['split']}` ({report['frames']} frames)",
             f"- Predictions: `{report['predictions']}`", "", "## Vector map", "",
             "| Class | AP | AP@0.5m | AP@1.0m | AP@1.5m | F1@1m |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, values in report["map"]["classes"].items():
        ap = values["ap_by_chamfer_m"]
        lines.append(f"| {name} | {values['ap']:.4f} | {ap['0.5']:.4f} | {ap['1.0']:.4f} | {ap['1.5']:.4f} | {values['f1_at_1m']:.4f} |")
    lines += ["", f"Benchmark three-class map mAP: **{report['map']['map_mAP']:.4f}**", "", "## Detection", ""]
    detection = report["detection"]
    if detection.get("available") is False:
        lines.append(detection["reason"])
    else:
        lines += ["| mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |", "|---:|---:|---:|---:|---:|---:|---:|",
                  f"| {detection['mean_ap']:.4f} | {detection['nd_score']:.4f} | {detection['tp_errors']['trans_err']:.4f} | {detection['tp_errors']['scale_err']:.4f} | {detection['tp_errors']['orient_err']:.4f} | {detection['tp_errors']['vel_err']:.4f} | {detection['tp_errors']['attr_err']:.4f} |",
                  "", detection.get("drivescene_context", ""),
                  "TP errors equal to 1.0 are the nuScenes fallback when a class does not reach 10% recall; they are not measured 1m errors."]
    occ = report["occupancy"]
    lines += ["", "## Occupancy", "", "| Metric | Value |", "|---|---:|",
              f"| Sparse observed mIoU | {occ['mIoU']:.4f} |" if occ["mIoU"] is not None else "| Sparse observed mIoU | N/A |",
              f"| Sparse observed macro F1 | {occ['macro_f1']:.4f} |" if occ["macro_f1"] is not None else "| Sparse observed macro F1 | N/A |",
              "", "This is not official Occ3D mIoU; see `metrics.json` for the label and visibility limitations.",
              "", "## Tracking", ""]
    tracking = report["tracking"]
    if tracking is None or tracking.get("available") is False:
        lines.append("Unavailable.")
    else:
        for key in ("amota", "amotp", "motar", "recall", "ids", "frag"):
            if key in tracking:
                value = tracking[key]
                lines.append(f"- {key.upper()}: `{value:.4f}`" if isinstance(value, (int, float))
                             else f"- {key.upper()}: `N/A`")
        lines += ["", tracking.get("drivescene_context", "")]
    lines += ["", "## Unavailable metrics", "", "- Traffic-light AP/F1: no state ground truth in the manifest.",
              "- FPS/latency: the existing inference run did not record timing; benchmark a new run for these metrics."]
    return "\n".join(lines)+"\n"


def _metric(value, digits=4):
    """Format a metric without hiding unavailable or very small values."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "N/A"
        if value != 0 and abs(value) < 10**(-digits):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _terminal_table(title, headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    headers = [str(cell) for cell in headers]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    separator = "+-"+"-+-".join("-"*width for width in widths)+"-+"
    def line(row):
        return "| "+" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))+" |"
    return "\n".join([title, separator, line(headers), separator,
                       *(line(row) for row in rows), separator])


def format_terminal_report(report):
    """Render the saved evaluation report as compact terminal tables."""
    sections = []
    detection = report.get("detection") or {}
    occupancy = report.get("occupancy") or {}
    tracking = report.get("tracking") or {}
    sections.append(_terminal_table("Evaluation summary", ("Metric", "Value"), [
        ("Split", report.get("split")),
        ("Frames", _metric(report.get("frames"))),
        ("Map mAP", _metric(report.get("map", {}).get("map_mAP"))),
        ("Detection mAP", _metric(detection.get("mean_ap"))),
        ("NDS", _metric(detection.get("nd_score"))),
        ("Occupancy mIoU", _metric(occupancy.get("mIoU"))),
        ("Occupancy macro F1", _metric(occupancy.get("macro_f1"))),
        ("AMOTA", _metric(tracking.get("amota"))),
        ("AMOTP", _metric(tracking.get("amotp"))),
    ]))

    map_rows = []
    for name, values in report.get("map", {}).get("classes", {}).items():
        ap = values.get("ap_by_chamfer_m", {})
        map_rows.append((name, _metric(values.get("ground_truth")),
                         _metric(values.get("predictions")), _metric(ap.get("0.5")),
                         _metric(ap.get("1.0")), _metric(ap.get("1.5")),
                         _metric(values.get("ap")), _metric(values.get("f1_at_1m")),
                         _metric(values.get("matched_chamfer_mean_m"))))
    if map_rows:
        sections.append(_terminal_table(
            "Vector map", ("Class", "GT", "Pred", "AP@0.5", "AP@1.0", "AP@1.5", "AP", "F1@1m", "Chamfer m"),
            map_rows))

    if detection.get("available") is False:
        sections.append(_terminal_table("Detection", ("Status", "Reason"),
                                        [("Unavailable", detection.get("reason", "unknown"))]))
    else:
        errors = detection.get("tp_errors", {})
        sections.append(_terminal_table(
            "nuScenes detection", ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"),
            [(_metric(detection.get("mean_ap")), _metric(detection.get("nd_score")),
              _metric(errors.get("trans_err")), _metric(errors.get("scale_err")),
              _metric(errors.get("orient_err")), _metric(errors.get("vel_err")),
              _metric(errors.get("attr_err")))]))
        class_rows = [(name, _metric(value)) for name, value in detection.get("mean_dist_aps", {}).items()]
        if class_rows:
            sections.append(_terminal_table("Detection AP by class", ("Class", "mAP"), class_rows))

    occ_rows = []
    for name, values in occupancy.get("classes", {}).items():
        occ_rows.append((name, _metric(values.get("support")), _metric(values.get("tp")),
                         _metric(values.get("fp")), _metric(values.get("fn")),
                         _metric(values.get("iou")), _metric(values.get("f1"))))
    if occ_rows:
        sections.append(_terminal_table(
            "Sparse occupancy", ("Class", "Support", "TP", "FP", "FN", "IoU", "F1"), occ_rows))

    if tracking.get("available") is False:
        sections.append(_terminal_table("Tracking", ("Status", "Reason"),
                                        [("Unavailable", tracking.get("reason", "unknown"))]))
    elif tracking:
        sections.append(_terminal_table(
            "nuScenes tracking", ("AMOTA", "AMOTP", "MOTAR", "Recall", "IDS", "FRAG"),
            [tuple(_metric(tracking.get(key)) for key in ("amota", "amotp", "motar", "recall", "ids", "frag"))]))
        label_metrics = tracking.get("label_metrics", {})
        names = list(label_metrics.get("amota", {}))
        track_rows = [(name, *(_metric(label_metrics.get(key, {}).get(name))
                               for key in ("amota", "amotp", "recall", "ids", "frag")))
                      for name in names]
        if track_rows:
            sections.append(_terminal_table(
                "Tracking by class", ("Class", "AMOTA", "AMOTP", "Recall", "IDS", "FRAG"), track_rows))

    unavailable = []
    for key, label in (("traffic_lights", "Traffic-light AP/F1"), ("performance", "FPS/latency")):
        item = report.get(key) or {}
        if item.get("available") is False:
            unavailable.append((label, item.get("reason", "unavailable")))
    if unavailable:
        sections.append(_terminal_table("Unavailable metrics", ("Metric", "Reason"), unavailable))
    return "\n\n".join(sections)


def evaluate(config, manifest, prediction_directory, output, split="val", video_output=None,
             fps=2.0, official=True):
    prediction_directory, output = Path(prediction_directory), Path(output)
    rows = [row for row in read_manifest(manifest) if row.get("split") == split]
    rows.sort(key=lambda row: (row["scene_token"], row["timestamp_us"]))
    if not rows:
        raise ValueError(f"No rows for split: {split}")
    missing = [row["sample_token"] for row in rows if not (prediction_directory/f"{row['sample_token']}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} predictions; first token: {missing[0]}")
    worlds = [json.loads((prediction_directory/f"{row['sample_token']}.json").read_text(encoding="utf-8")) for row in rows]
    map_frames = [_frame_map_data(row, world, config["grid"]) for row, world in zip(rows, worlds)]
    map_report = _map_metrics(map_frames)
    occupancy = _occupancy_metrics(rows, prediction_directory, config["model"]["occupancy_classes"])
    output.mkdir(parents=True, exist_ok=True)
    if official:
        try:
            detection, tracking = _official_nuscenes(rows, worlds, config["data_root"], config["version"], split, output/"official_nuscenes")
        except Exception as error:
            detection = {"available": False, "reason": f"Official nuScenes evaluation failed: {type(error).__name__}: {error}"}
            tracking = {"available": False, "reason": detection["reason"]}
    else:
        detection = {"available": False, "reason": "Skipped by --skip-official."}
        tracking = {"available": False, "reason": "Skipped by --skip-official."}
    grid_range = {"x": list(config["grid"]["x"]), "y": list(config["grid"]["y"])}
    range_note = (
        f"Predictions are limited to the configured ego grid {grid_range}; "
        "the official protocol scores cars/trucks to 50 m and pedestrians to 40 m, "
        "so objects outside the grid are counted as false negatives.")
    if detection.get("available") is not False:
        detection["drivescene_context"] = (
            "Official nuScenes protocol applied to raw detection-head candidates when retained; "
            "CenterTracker identities are used only for the tracking submission. " + range_note)
        detection["range_limitation"] = grid_range
    if tracking.get("available") is not False:
        tracking["drivescene_context"] = (
            "AMOTP=2.0 is the protocol ceiling when recall is zero and is not a 2m successful-match measurement. "
            + range_note)
    video = _render_error_video(rows, worlds, map_frames, occupancy, video_output, fps) if video_output else None
    report = {"schema_version": "1.0", "split": split, "frames": len(rows),
              "manifest": str(manifest), "predictions": str(prediction_directory),
              "map": map_report, "detection": detection, "occupancy": occupancy,
              "tracking": tracking, "traffic_lights": {"available": False, "reason": "No traffic-light state ground truth."},
              "performance": {"available": False, "reason": "Existing inference artifacts do not contain timing measurements."},
              "error_video": video}
    report = _json_safe(report)
    (output/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output/"metrics.md").write_text(_markdown(report), encoding="utf-8")
    return {"output": str(output), "metrics_json": str(output/"metrics.json"),
            "metrics_markdown": str(output/"metrics.md"), "frames": len(rows),
            "map_mAP": map_report["map_mAP"],
            "NDS": detection.get("nd_score") if isinstance(detection, dict) else None,
            "occupancy_mIoU": occupancy["mIoU"], "AMOTA": tracking.get("amota") if isinstance(tracking, dict) else None,
            "error_video": video["path"] if video else None}
