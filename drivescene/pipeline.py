from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .data import camera_tensor, read_manifest
from .models import SharedBEVModel, supervised_loss
from .compiler import compile_scene, save_world
from .tracking import BEVByteTracker
from .visualize import render_world
from .map_postprocess import MapTemporalStabilizer, postprocess_map
from .detection_postprocess import postprocess_detections


def _filter_occupancy(labels, confidence, config):
    """Remove isolated voxels and implausible vertical columns conservatively."""
    from scipy.ndimage import convolve
    labels = np.asarray(labels, dtype=np.uint8).copy()
    confidence = np.asarray(confidence, dtype=float)
    settings = config["compiler"].get("occupancy_geometry", {})
    occupied = labels == 1
    minimum_neighbors = int(settings.get("minimum_neighbors", 0))
    if minimum_neighbors > 1 and occupied.any():
        neighbors = convolve(occupied.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8),
                             mode="constant", cval=0)
        labels[occupied & (neighbors < minimum_neighbors)] = 255
    maximum_vertical = int(settings.get("max_occupied_voxels_per_column", 0))
    if maximum_vertical > 0:
        for x, y in np.argwhere((labels == 1).sum(2) > maximum_vertical):
            indices = np.flatnonzero(labels[x, y] == 1)
            keep = indices[np.argsort(confidence[x, y, indices])[-maximum_vertical:]]
            remove = np.setdiff1d(indices, keep, assume_unique=True)
            labels[x, y, remove] = 255
    drivable_z = settings.get("drivable_z")
    if drivable_z is not None:
        low, high = map(float, drivable_z)
        grid = config["grid"]
        centers = np.linspace(grid["z"][0], grid["z"][1], grid["shape"][2], endpoint=False)
        centers += (grid["z"][1]-grid["z"][0])/(2*grid["shape"][2])
        invalid = (centers < low) | (centers > high)
        for z in np.flatnonzero(invalid):
            layer = labels[:, :, z]
            layer[layer == 2] = 255
    return labels


def _distributed_context(device):
    """Initialize torchrun DDP when WORLD_SIZE > 1; preserve single-GPU usage."""
    import os
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = f"cuda:{local_rank}"
    return device, rank, local_rank, world_size


def decode(prediction, config, trained_tasks):
    cfg = config["model"]
    vectors, boxes, map_candidates, detection_candidates = [], [], [], []
    if "map" in trained_tasks:
        probs = prediction["map"]["logits"][0].softmax(-1)
        scores, labels = probs[:, :-1].max(-1)
        for i in range(len(scores)):
            map_candidates.append({"id": f"query:{i}", "kind": cfg["map_classes"][int(labels[i])],
                                   "score": float(scores[i]),
                                   "points": prediction["map"]["points"][0, i].cpu().tolist()})
        vectors = postprocess_map(map_candidates, config)
    if "detection" in trained_tasks:
        foreground = prediction["detection"]["logits"][0].softmax(-1)[:, :-1]
        scores, labels = foreground.max(-1)
        for i in range(len(scores)):
            values = prediction["detection"]["boxes"][0, i].cpu().numpy()
            detection_candidates.append({"id": f"query:{i}",
                "kind": cfg["detection_classes"][int(labels[i])], "score": float(scores[i]),
                "center": values[:3].tolist(), "size": values[3:6].tolist(),
                "yaw": float(np.arctan2(values[6], values[7])), "velocity": values[8:].tolist()})
        boxes = postprocess_detections(detection_candidates, config)
    occupancy = None
    if "occupancy" in trained_tasks:
        probabilities = prediction["occupancy"][0].softmax(0)
        confidence, labels = probabilities.max(0)
        thresholds = config["compiler"].get("occupancy_confidence", {})
        class_thresholds = confidence.new_tensor([
            thresholds.get(name, 0.0) for name in cfg["occupancy_classes"]])
        accepted = confidence >= class_thresholds[labels]
        occupancy = labels.cpu().numpy().astype(np.uint8)
        occupancy[~accepted.cpu().numpy()] = 255
        occupancy[~prediction["observed"][0].cpu().numpy()] = 255
        occupancy = _filter_occupancy(occupancy, confidence.cpu().numpy(), config)
    return vectors, boxes, occupancy, map_candidates, detection_candidates


def run(config, manifest, output, mode, checkpoint=None, limit=5, device="cpu", lights=False):
    if mode not in ("annotations", "native_research"):
        raise ValueError("Unsupported inference mode")
    tracker = BEVByteTracker(config["tracking"])
    map_stabilizer = MapTemporalStabilizer(config)
    model, trained_tasks = None, []
    if mode == "native_research":
        if not checkpoint:
            raise ValueError("Camera inference needs a trained native checkpoint; use smoke to test random initialization")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("backend") != "native_research":
            raise ValueError("Checkpoint is not a native research checkpoint")
        if state["config"]["grid"] != config["grid"] or state["config"]["model"] != config["model"]:
            raise ValueError("Checkpoint architecture/grid mismatch; use its saved config")
        model = SharedBEVModel(config).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        trained_tasks = state["trained_tasks"]
    detector = None
    if lights:
        from .lights import YOLOv8Lights
        detector = YOLOv8Lights(config["traffic_lights"]["checkpoint"], config["traffic_lights"]["state_classes"], device)
    previous, previous_pose, previous_scene, previous_time = None, None, None, None
    paths = []
    for index, row in enumerate(read_manifest(manifest)):
        if limit is not None and index >= limit:
            break
        frame = {key: row[key] for key in ("sample_token", "scene_token", "timestamp_us", "ego_to_global")}
        occupancy = None
        if model:
            images, projection = camera_tensor(row, config["model"]["image_size"])
            pose = np.asarray(row["ego_to_global"])
            if previous_scene != row["scene_token"] or (previous_time is not None and row["timestamp_us"]-previous_time > 1_500_000):
                previous = None
            transform = None if previous is None else torch.tensor(np.linalg.inv(previous_pose) @ pose, device=device, dtype=torch.float32)[None]
            with torch.no_grad():
                result = model(images[None].to(device), projection[None].to(device), previous, transform)
            vectors, boxes, occupancy, map_candidates, detection_candidates = decode(result, config, trained_tasks)
            vectors = map_stabilizer.update(vectors, row["ego_to_global"], row["scene_token"])
            previous, previous_pose, previous_scene, previous_time = result["bev"].detach(), pose, row["scene_token"], row["timestamp_us"]
            provenance = {"mode": "native_research", "checkpoint": str(checkpoint), "trained_tasks": trained_tasks,
                          "architecture": "custom shared BEV research adaptation; not official paper implementations",
                          "occupancy_confidence": config["compiler"].get("occupancy_confidence", {}),
                          "map_confidence": config["compiler"].get("map_confidence", {}),
                          "map_geometry_filter": config["compiler"].get("map_geometry", {}),
                          "map_temporal": config["compiler"].get("map_temporal", {})}
        else:
            vectors, boxes = row["targets"]["map_vectors"] or [], row["targets"]["boxes"]
            map_candidates = detection_candidates = None
            provenance = {"mode": "annotation_replay", "uses_ground_truth": True,
                          "purpose": "data/geometry/compiler verification, NOT perception inference"}
        frame["map_vectors"] = vectors
        frame["map_candidates"] = map_candidates
        frame["detection_candidates"] = detection_candidates
        frame["agents"] = tracker.update(boxes, row["timestamp_us"], row["ego_to_global"], row["scene_token"])
        frame["traffic_lights"] = detector.predict(row["cameras"]) if detector else []
        frame["provenance"] = provenance
        world, arrays = compile_scene(frame, config, occupancy)
        path = save_world(world, arrays, output)
        if index < 3:
            render_world(world, arrays, path.with_suffix(".png"), row)
        paths.append(str(path))
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(output, "run.json").write_text(json.dumps({"mode": mode, "frames": len(paths), "outputs": paths}, indent=2))
    return {"frames": len(paths), "output": str(output), "mode": mode}


def _prediction_is_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_prediction_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_prediction_is_finite(item) for item in value)
    return True


def _zero_loss_from_prediction(value):
    """Build a finite zero-valued graph so a skipped DDP step completes reduction."""
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        return torch.nan_to_num(value).sum()*0
    if isinstance(value, dict):
        parts = [_zero_loss_from_prediction(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        parts = [_zero_loss_from_prediction(item) for item in value]
    else:
        parts = []
    parts = [part for part in parts if part is not None]
    return sum(parts) if parts else None


def _validate_map_supervision(rows, config):
    """Fail before training if labels exceed either spatial or query capacity."""
    grid = config["grid"]
    query_count = int(config["model"]["map_queries"])
    maximum = 0
    outside = 0
    for row in rows:
        vectors = row["targets"].get("map_vectors") or []
        maximum = max(maximum, len(vectors))
        for vector in vectors:
            points = np.asarray(vector["points"], dtype=float)
            outside += int(((points[:, 0] < grid["x"][0]-1e-5) |
                            (points[:, 0] > grid["x"][1]+1e-5) |
                            (points[:, 1] < grid["y"][0]-1e-5) |
                            (points[:, 1] > grid["y"][1]+1e-5)).sum())
    if outside:
        raise ValueError(f"Map supervision contains {outside} points outside the configured ego grid; regenerate the manifest")
    if maximum > query_count:
        raise ValueError(f"Map supervision needs {maximum} queries but model.map_queries={query_count}")
    return {"maximum_vectors": maximum, "map_queries": query_count, "outside_points": outside}


def _validate_detection_supervision(rows, config):
    grid = config["grid"]
    query_count = int(config["model"]["detection_queries"])
    maximum = outside = 0
    for row in rows:
        boxes = row["targets"].get("boxes") or []
        maximum = max(maximum, len(boxes))
        for box in boxes:
            center = box["center"]
            outside += int(any(center[index] < grid[axis][0]-1e-5 or
                               center[index] > grid[axis][1]+1e-5
                               for index, axis in enumerate(("x", "y", "z"))))
    if outside:
        raise ValueError(f"Detection supervision contains {outside} boxes outside the BEV grid; regenerate the manifest")
    if maximum > query_count:
        raise ValueError(f"Detection supervision needs {maximum} queries but model.detection_queries={query_count}")
    return {"maximum_boxes": maximum, "detection_queries": query_count, "outside_boxes": outside}


def train(config, manifest, output, steps=10, device="cpu", backbone=None,
          resume=None, save_every=250):
    device, rank, local_rank, world_size = _distributed_context(device)
    torch.manual_seed(42+rank)
    model = SharedBEVModel(config).to(device)
    resume_state = None
    if resume:
        resume_state = torch.load(resume, map_location="cpu", weights_only=False)
        if resume_state.get("backend") != "native_research":
            raise ValueError("Resume checkpoint is not a native research checkpoint")
        if (resume_state["config"]["grid"] != config["grid"] or
                resume_state["config"]["model"] != config["model"]):
            raise ValueError("Resume checkpoint architecture/grid mismatch")
        model.load_state_dict(resume_state["model"], strict=True)
    elif backbone:
        weights = torch.load(backbone, map_location="cpu", weights_only=True)
        expected = model.encoder.body.state_dict()
        filtered = {k: v for k, v in weights.items() if k in expected}
        model.encoder.body.load_state_dict(filtered, strict=True)
    model.train()
    # Keep batchnorm running statistics fixed for one sample / six-view training.
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    learning_rate = float(config.get("training", {}).get("learning_rate", 5e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    if resume_state and resume_state.get("optimizer"):
        optimizer.load_state_dict(resume_state["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    rows = [r for r in read_manifest(manifest) if r["split"] == "train"]
    if not rows:
        raise ValueError("Training manifest has no train scenes")
    map_supervision = _validate_map_supervision(rows, config)
    detection_supervision = _validate_detection_supervision(rows, config)
    scene_order = list(dict.fromkeys(row["scene_token"] for row in rows))
    if world_size > len(scene_order):
        raise ValueError(f"DDP world size {world_size} exceeds {len(scene_order)} train scenes")
    local_scenes = set(scene_order[rank::world_size])
    local_rows = [row for row in rows if row["scene_token"] in local_scenes]
    directory = Path(output)
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    start_step = int(resume_state.get("steps", 0)) if resume_state else 0
    if start_step >= steps:
        raise ValueError(f"Resume checkpoint is already at step {start_step}, target is {steps}")
    history_path = directory / "train.json"
    history = (json.loads(history_path.read_text())
               if rank == 0 and resume_state and history_path.is_file() else [])
    trained_tasks = set(resume_state.get("trained_tasks", [])) if resume_state else set()
    previous, previous_pose, previous_scene, previous_time = None, None, None, None
    def save_checkpoint(path, completed_steps):
        if rank != 0:
            return
        state_model = model.module if world_size > 1 else model
        payload = {"backend": "native_research", "model": state_model.state_dict(),
                   "optimizer": optimizer.state_dict(), "config": config,
                   "trained_tasks": sorted(trained_tasks), "steps": completed_steps,
                   "world_size": world_size, "global_batch_size": world_size,
                   "temporal_training": True, "model_revision": "perception-v3",
                   "training_manifest": str(manifest), "validated_for_planning": False}
        payload["map_supervision"] = map_supervision
        payload["detection_supervision"] = detection_supervision
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        history_path.write_text(json.dumps(history, indent=2))

    for step in range(start_step, steps):
        # Whole scenes stay on one rank, allowing each worker to train temporal
        # fusion on consecutive frames while DDP synchronizes task gradients.
        row = local_rows[step % len(local_rows)]
        images, projection = camera_tensor(row, config["model"]["image_size"])
        pose = np.asarray(row["ego_to_global"])
        if (previous_scene != row["scene_token"] or
                (previous_time is not None and row["timestamp_us"]-previous_time > 1_500_000) or
                (previous_time is not None and row["timestamp_us"] <= previous_time)):
            previous = None
        transform = None if previous is None else torch.tensor(
            np.linalg.inv(previous_pose) @ pose, device=device, dtype=torch.float32)[None]
        target = dict(row["targets"])
        # Occupancy from explicit normalized labels only. Original Occ3D must be converted first.
        if target.get("occupancy_path"):
            with np.load(target["occupancy_path"]) as labels:
                if "drivescene_semantic" not in labels:
                    raise ValueError("Run convert-occupancy; raw Occ3D IDs/grid cannot be used directly")
                target["occupancy"] = labels["drivescene_semantic"]
        optimizer.zero_grad(set_to_none=True)
        pred = model(images[None].to(device), projection[None].to(device), previous, transform)
        predictions_finite = torch.tensor(
            1 if _prediction_is_finite(pred) else 0, device=device, dtype=torch.int32)
        if world_size > 1:
            torch.distributed.all_reduce(predictions_finite, op=torch.distributed.ReduceOp.MIN)
        if not int(predictions_finite):
            dummy = _zero_loss_from_prediction(pred)
            if dummy is not None:
                dummy.backward()
            optimizer.zero_grad(set_to_none=True)
            previous = previous_pose = previous_scene = previous_time = None
            if rank == 0:
                print(json.dumps({"step": step+1, "warning": "non-finite prediction; step skipped"}), flush=True)
            continue
        losses = supervised_loss(pred, [target], config)
        total = sum(losses.values())
        loss_finite = torch.isfinite(total).to(torch.int32)
        if world_size > 1:
            torch.distributed.all_reduce(loss_finite, op=torch.distributed.ReduceOp.MIN)
        if not int(loss_finite):
            torch.nan_to_num(total).backward()
            optimizer.zero_grad(set_to_none=True)
            previous = previous_pose = previous_scene = previous_time = None
            if rank == 0:
                print(json.dumps({"step": step+1, "warning": "non-finite loss; step skipped"}), flush=True)
            continue
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        gradients_finite = torch.isfinite(grad_norm).to(torch.int32)
        if world_size > 1:
            torch.distributed.all_reduce(gradients_finite, op=torch.distributed.ReduceOp.MIN)
        if not int(gradients_finite):
            optimizer.zero_grad(set_to_none=True)
            previous = previous_pose = previous_scene = previous_time = None
            for group in optimizer.param_groups:
                group["lr"] *= .5
            if rank == 0:
                print(json.dumps({"step": step+1, "warning": "non-finite gradient; step skipped",
                                  "new_lr": optimizer.param_groups[0]["lr"]}), flush=True)
            continue
        optimizer.step()
        previous, previous_pose = pred["bev"].detach(), pose
        previous_scene, previous_time = row["scene_token"], row["timestamp_us"]
        trained_tasks.update(losses)
        reported = {name: value.detach().clone() for name, value in losses.items()}
        if world_size > 1:
            for value in reported.values():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                value /= world_size
        if rank == 0:
            record = {"step": step+1, **{k: float(v) for k, v in reported.items()},
                      "grad_norm": float(grad_norm)}
            history.append(record)
            print(json.dumps(record), flush=True)
        if (step+1) % save_every == 0:
            save_checkpoint(directory / "latest.pt", step+1)
    checkpoint = directory / "last.pt"
    save_checkpoint(checkpoint, steps)
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()
    return {"checkpoint": str(checkpoint), "steps": steps,
            "trained_tasks": sorted(trained_tasks), "rank": rank,
            "world_size": world_size, "global_batch_size": world_size,
            "map_supervision": map_supervision,
            "detection_supervision": detection_supervision}


def smoke(config, output, device="cpu"):
    """Synthetic tensors exercise differentiable paths, never presented as inference."""
    import copy
    cfg = copy.deepcopy(config)
    cfg["grid"]["shape"] = [12, 12, 4]
    cfg["model"].update(channels=32, map_queries=3, map_points=5, detection_queries=6, layers=1)
    torch.manual_seed(42)
    model = SharedBEVModel(cfg).to(device).eval()
    images = torch.randn(1, 6, 3, 64, 96, device=device)
    projection = torch.tensor([[35., 0., 48., 0.], [0., 35., 32., 0.], [0., 0., 1., 12.], [0., 0., 0., 1.]], device=device)[None, None].repeat(1, 6, 1, 1)
    pred = model(images, projection)
    second = model(images, projection, pred["bev"].detach(), torch.eye(4, device=device)[None])
    targets = [{"boxes": [], "map_vectors": [], "occupancy": np.zeros(cfg["grid"]["shape"], dtype=np.uint8)}]
    losses = supervised_loss(second, targets, cfg)
    total = sum(losses.values())
    total.backward()
    for name, module in [("backbone", model.encoder), ("BEV", model.bev_encoder), ("map", model.map_head),
                         ("detection", model.detection_head), ("occupancy", model.occupancy_head)]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"Missing or invalid gradients: {name}")
    report = {"mode": "synthetic_smoke", "backend": "native_research", "device": device,
              "losses": {k: float(v.detach()) for k, v in losses.items()},
              "bev_shape": list(pred["bev"].shape), "occupancy_shape": list(pred["occupancy"].shape),
              "all_branches_backward": True, "official_models_validated": False}
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(output, "smoke.json").write_text(json.dumps(report, indent=2))
    return report
