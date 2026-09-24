import json
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from drivescene.official_shared import OfficialSharedStack

torch.set_num_threads(4)
torch.manual_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--manifest")
parser.add_argument("--backward", action="store_true")
args = parser.parse_args()
model = OfficialSharedStack(root, smoke=True).eval()
images = torch.randn(1, 6, 3, 64, 96)
projection = np.array([[35., 0., 48., 0.], [0., 35., 32., 0.], [0., 0., 1., 12.], [0., 0., 0., 1.]])
metadata = [dict(lidar2img=[projection]*6, img_shape=[(64, 96, 3)]*6, can_bus=np.zeros(18),
                 scene_token="synthetic", sample_idx="synthetic", occ_size=np.array([16, 16, 8]))]
if args.manifest:
    from drivescene.data import camera_tensor, read_manifest
    row = next(read_manifest(args.manifest))
    order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    row["cameras"] = sorted(row["cameras"], key=lambda c: order.index(c["name"]))
    images, projections = camera_tensor(row, [128, 224])
    images = images[None]
    if "lidar_to_ego" not in row:
        raise ValueError("Use the manifest produced by prepare-sparse-occupancy, which includes reference calibration")
    lidar_to_ego = np.asarray(row["lidar_to_ego"])
    metadata = [dict(lidar2img=[p.numpy() @ lidar_to_ego for p in projections],
                     img_shape=[(128, 224, 3)]*6, can_bus=np.zeros(18), scene_token=row["scene_token"],
                     sample_idx=row["sample_token"], occ_size=np.array([16, 16, 8]))]
calls = {"image_backbone": 0, "bev_encoder": 0}
def counter(key):
    def hook(*unused):
        calls[key] += 1
    return hook
model.detector.img_backbone.register_forward_hook(counter("image_backbone"))
model.detector.pts_bbox_head.transformer.encoder.register_forward_hook(counter("bev_encoder"))
with torch.set_grad_enabled(args.backward):
    result = model(images, metadata)
if args.backward:
    # Connectivity check, not a supervised loss or an optimizer training step.
    objective = result["map"]["all_pts_preds"].square().mean()+result["detection"]["all_bbox_preds"].square().mean()
    objective = objective + result["occupancy"][0][-1].square().mean()+result["occupancy"][1][-1].square().mean()
    objective.backward()
    for module in [model.detector.img_backbone, model.map_head, model.height_lift, model.occ_backbone, model.occ_head]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
assert calls == {"image_backbone": 1, "bev_encoder": 1}, calls
report = {"mode": "official_shared_real_camera_smoke" if args.manifest else "official_shared_synthetic_smoke", "trained": False,
          "calls": calls, "backward_verified": args.backward,
          "bev": list(result["bev"].shape),
          "map": list(result["map"]["all_pts_preds"].shape),
          "detection": list(result["detection"]["all_bbox_preds"].shape),
          "occupancy_classes": list(result["occupancy"][0][-1].shape),
          "occupancy_masks": list(result["occupancy"][1][-1].shape)}
Path("outputs/official-smoke").mkdir(parents=True, exist_ok=True)
filename = "real-camera.json" if args.manifest else "result.json"
Path("outputs/official-smoke", filename).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
