"""Real official BEVFormer-tiny checkpoint inference; no annotation predictions.

Uses sensor calibration only (no LiDAR points). Kept separate from the untrained
joint stack so component accuracy is never attributed to the new shared bridge.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from pyquaternion import Quaternion

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from drivescene.upstream import register
from drivescene.data import read_manifest, camera_tensor
from drivescene.geometry import transform_points

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/processed/mini-v3-with-occ.jsonl")
parser.add_argument("--checkpoint", default="checkpoints/bevformer_tiny_epoch_24.pth")
parser.add_argument("--output", default="outputs/bevformer-official")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
torch.set_num_threads(4)
register(root, include_map=False, include_occ=False)
cfg = Config.fromfile(str(root / "third_party/BEVFormer/projects/configs/bevformer/bevformer_tiny.py"))
cfg.model.pretrained = None
model = build_model(cfg.model).to(args.device).eval()
state = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(state.get("state_dict", state), strict=True)
row = next(read_manifest(args.manifest))
order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
row["cameras"] = sorted(row["cameras"], key=lambda c: order.index(c["name"]))
images, projections = camera_tensor(row, [450, 800])
images = F.pad(images, (0, 0, 0, 30))[None].to(args.device)
lidar_to_ego = np.asarray(row["lidar_to_ego"])
pose = np.asarray(row["ego_to_global"])
can_bus = np.zeros(18)
can_bus[3:7] = Quaternion(matrix=pose[:3, :3]).elements
can_bus[-2] = np.arctan2(pose[1, 0], pose[0, 0]) % (2*np.pi)
metadata = [dict(lidar2img=[p.numpy() @ lidar_to_ego for p in projections], img_shape=[(480, 800, 3)]*6,
                 can_bus=can_bus, scene_token=row["scene_token"], sample_idx=row["sample_token"], box_type_3d=LiDARInstance3DBoxes)]
with torch.no_grad():
    features = model.extract_img_feat(images, metadata)
    result = model.pts_bbox_head(features, metadata)
    boxes, scores, labels = model.pts_bbox_head.get_bboxes(result, metadata)[0]
    centers = transform_points(boxes.gravity_center.cpu().numpy(), lidar_to_ego)
    values = boxes.tensor.cpu().numpy()
agents = []
for index in range(len(scores)):
    if scores[index] < .25:
        continue
    direction = lidar_to_ego[:3, :3] @ np.array([np.cos(values[index, 6]), np.sin(values[index, 6]), 0])
    velocity = lidar_to_ego[:3, :3] @ np.r_[values[index, 7:9], 0]
    agents.append(dict(id="detection:%s" % index, kind=cfg.class_names[int(labels[index])], center=centers[index].tolist(),
                       size=values[index, 3:6].tolist(), yaw=float(np.arctan2(direction[1], direction[0])),
                       velocity=velocity[:2].tolist(), score=float(scores[index])))
frame = {key: row[key] for key in ("sample_token", "scene_token", "timestamp_us", "ego_to_global")}
frame.update(agents=agents, map_vectors=[], traffic_lights=[],
             provenance=dict(mode="official_bevformer_pretrained", checkpoint=args.checkpoint,
                             uses_ground_truth=False, can_bus="pose_only; dynamics unavailable",
                             unavailable_tasks=["vector_map", "occupancy", "traffic_light_state"]))
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
(output / "frame.json").write_text(json.dumps(frame, indent=2))
report = dict(sample=row["sample_token"], detections=len(agents), strict_checkpoint_load=True,
              bev_shape=list(result["bev_embed"].shape), device=args.device)
(output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

