"""Training data loading, scene-consistent augmentation and schedules."""
from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .data import camera_tensor
from .geometry import grid_centers, transform_points
from .map_postprocess import clip_polylines
from .models import supervised_loss


def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps=250, min_factor=0.01):
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_factor + (1 - min_factor) * cosine)


def color_jitter_images(images, brightness=0.2, contrast=0.2, saturation=0.15):
    """Apply one set of photometric factors to all six cameras."""
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    rgb = images * std + mean
    brightness_scale = 1 + (random.random() * 2 - 1) * brightness
    contrast_scale = 1 + (random.random() * 2 - 1) * contrast
    saturation_scale = 1 + (random.random() * 2 - 1) * saturation
    rgb = rgb * brightness_scale
    mid = rgb.mean(dim=(-2, -1), keepdim=True)
    rgb = (rgb - mid) * contrast_scale + mid
    gray = rgb.mean(dim=1, keepdim=True)
    rgb = (rgb - gray) * saturation_scale + gray
    return (rgb.clamp(0, 1) - mean) / std


def sample_scene_augment(config, rng=None):
    """One rigid BEV transform reused for every frame of a scene."""
    rng = rng or random
    settings = config.get("training", {}).get("augmentation", {})
    if not settings.get("enabled", True):
        return np.eye(4, dtype=np.float32)
    degrees = float(settings.get("bev_rotation_degrees", 22.5))
    angle = math.radians((rng.random() * 2 - 1) * degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0], transform[0, 1] = cosine, -sine
    transform[1, 0], transform[1, 1] = sine, cosine
    if rng.random() < float(settings.get("bev_flip_prob", 0.5)):
        transform[1] *= -1
    return transform


def _inside_grid(center, grid):
    return all(grid[axis][0] <= center[index] <= grid[axis][1]
               for index, axis in enumerate(("x", "y", "z")))


def _warp_occupancy(semantic, transform, grid):
    centers = grid_centers(grid)
    inverse = np.linalg.inv(transform)
    source = transform_points(centers.reshape(-1, 3), inverse).reshape(centers.shape)
    low = np.array([grid[axis][0] for axis in ("x", "y", "z")])
    high = np.array([grid[axis][1] for axis in ("x", "y", "z")])
    shape = np.asarray(grid["shape"])
    index = np.floor((source - low) / (high - low) * shape).astype(np.int64)
    inside = ((index >= 0) & (index < shape)).all(-1)
    warped = np.full(semantic.shape, 255, dtype=np.uint8)
    dest = index[inside]
    warped[inside] = semantic[dest[:, 0], dest[:, 1], dest[:, 2]]
    return warped


def apply_bev_augment(sample, transform, grid):
    """Rotate/flip ego-frame supervision and the matching camera projections."""
    transform = np.asarray(transform, dtype=np.float32)
    if np.allclose(transform, np.eye(4), atol=1e-6):
        return sample
    inverse = np.linalg.inv(transform)
    sample = dict(sample)
    sample["projection"] = sample["projection"] @ torch.as_tensor(
        inverse, dtype=sample["projection"].dtype)
    sample["pose"] = np.asarray(sample["pose"], dtype=np.float32) @ inverse
    target = deepcopy(sample["target"])
    boxes = []
    for box in target.get("boxes") or []:
        item = deepcopy(box)
        center = transform_points([item["center"]], transform)[0]
        if not _inside_grid(center, grid):
            continue
        heading = transform[:3, :3] @ np.array(
            [math.cos(item["yaw"]), math.sin(item["yaw"]), 0.0])
        velocity = transform[:3, :3] @ np.array(
            [item.get("velocity", [0.0, 0.0])[0], item.get("velocity", [0.0, 0.0])[1], 0.0])
        item["center"] = center.tolist()
        item["yaw"] = float(math.atan2(heading[1], heading[0]))
        item["velocity"] = velocity[:2].tolist()
        boxes.append(item)
    target["boxes"] = boxes
    vectors = []
    for vector in target.get("map_vectors") or []:
        points = np.asarray(vector["points"], dtype=float)
        xyz = np.column_stack([points, np.zeros(len(points))])
        transformed = transform_points(xyz, transform)[:, :2]
        for part in clip_polylines(transformed, grid):
            item = deepcopy(vector)
            item["points"] = part.tolist()
            item["closed"] = bool(np.linalg.norm(part[0] - part[-1]) <= 0.5)
            vectors.append(item)
    target["map_vectors"] = vectors
    if target.get("occupancy") is not None:
        target["occupancy"] = _warp_occupancy(np.asarray(target["occupancy"]), transform, grid)
    sample["target"] = target
    return sample


class CameraFrameDataset(torch.utils.data.Dataset):
    """Load one six-camera frame; photometric jitter is per-sample, BEV is not."""

    def __init__(self, rows, image_size, color_jitter=None):
        self.rows = rows
        self.image_size = image_size
        self.color_jitter = color_jitter or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images, projection = camera_tensor(row, self.image_size)
        if self.color_jitter:
            images = color_jitter_images(images, **self.color_jitter)
        target = dict(row["targets"])
        occupancy_path = target.get("occupancy_path")
        if occupancy_path:
            with np.load(occupancy_path) as labels:
                if "drivescene_semantic" not in labels:
                    raise ValueError("Run convert-occupancy; raw Occ3D IDs/grid cannot be used directly")
                target["occupancy"] = labels["drivescene_semantic"]
        return {
            "images": images,
            "projection": projection,
            "pose": np.asarray(row["ego_to_global"], dtype=np.float32),
            "scene_token": row["scene_token"],
            "timestamp_us": int(row["timestamp_us"]),
            "target": target,
        }


class SceneCycleSampler(torch.utils.data.Sampler):
    """Yield whole scenes in time order; shuffle scene order between cycles."""

    def __init__(self, rows, steps, start_step=0, shuffle_scenes=True, seed=42):
        self.steps = int(steps)
        self.start_step = int(start_step)
        self.shuffle_scenes = shuffle_scenes
        self.seed = int(seed)
        self.scene_order, self.by_scene = [], {}
        for index, row in enumerate(rows):
            token = row["scene_token"]
            if token not in self.by_scene:
                self.scene_order.append(token)
                self.by_scene[token] = []
            self.by_scene[token].append(index)
        if self.start_step >= self.steps:
            raise ValueError("Sampler start_step must be below the target step count")

    def __len__(self):
        return self.steps - self.start_step

    def __iter__(self):
        produced = skipped = epoch = 0
        while produced < len(self):
            order = list(self.scene_order)
            if self.shuffle_scenes:
                random.Random(self.seed + epoch).shuffle(order)
            for token in order:
                for index in self.by_scene[token]:
                    if skipped < self.start_step:
                        skipped += 1
                        continue
                    yield index
                    produced += 1
                    if produced >= len(self):
                        return
            epoch += 1


def identity_collate(batch):
    if len(batch) != 1:
        raise ValueError("Temporal training uses batch size 1")
    return batch[0]


def build_train_loader(rows, config, steps, start_step, device, rank=0):
    settings = config.get("training", {})
    augmentation = settings.get("augmentation", {})
    color = {}
    if augmentation.get("enabled", True):
        color = {key: float(augmentation[key]) for key in
                 ("brightness", "contrast", "saturation") if key in augmentation}
        color.setdefault("brightness", 0.2)
        color.setdefault("contrast", 0.2)
        color.setdefault("saturation", 0.15)
    dataset = CameraFrameDataset(rows, config["model"]["image_size"], color)
    sampler = SceneCycleSampler(
        rows, steps, start_step, settings.get("shuffle_scenes", True), seed=42 + rank)
    workers = int(settings.get("num_workers", 4))
    options = dict(batch_size=1, sampler=sampler, num_workers=workers,
                   collate_fn=identity_collate,
                   pin_memory=str(device).startswith("cuda"))
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(dataset, **options)


@torch.no_grad()
def validation_losses(model, rows, config, device, limit=16):
    if not rows or limit <= 0:
        return {}
    unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    unwrapped.eval()
    totals, count = {}, 0
    for row in rows[:limit]:
        images, projection = camera_tensor(row, config["model"]["image_size"])
        target = dict(row["targets"])
        occupancy_path = target.get("occupancy_path")
        if occupancy_path and Path(occupancy_path).is_file():
            with np.load(occupancy_path) as labels:
                if "drivescene_semantic" in labels:
                    target["occupancy"] = labels["drivescene_semantic"]
        prediction = unwrapped(images[None].to(device), projection[None].to(device))
        for name, value in supervised_loss(prediction, [target], config).items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
    unwrapped.train()
    for module in unwrapped.encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    return {name: value / max(count, 1) for name, value in totals.items()}
