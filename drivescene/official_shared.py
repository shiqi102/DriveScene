"""Shared BEV integration using actual upstream BEVFormer/MapTRv2/OccFormer.

The BEV-to-volume bridge and replacement of MapTRv2's LSS encoder require joint
training. Individual upstream checkpoints do not make this assembled model trained.
This module uses the separate Python 3.8 / OpenMMLab environment.
"""
import copy
from pathlib import Path
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .upstream import register


def shared_map_features(self, *args, **kwargs):
    shared = kwargs.get("prev_bev")
    if shared is None:
        raise ValueError("Shared MapTRv2 head needs a BEV from the common encoder")
    return {"bev": shared, "depth": None}


class OfficialSharedStack(nn.Module):
    def __init__(self, root, smoke=False):
        super().__init__()
        from mmcv import Config, ConfigDict
        from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
        from mmdet.models import build_neck, build_head
        from mmdet3d.models import build_model, build_backbone
        register(root)

        if "DriveSceneBEVFormerEncoder" not in TRANSFORMER_LAYER_SEQUENCE.module_dict:
            @TRANSFORMER_LAYER_SEQUENCE.register_module()
            class DriveSceneBEVFormerEncoder(nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()

                def forward(self, *args, **kwargs):
                    raise RuntimeError("Use shared_map_features; no second BEV encoder is permitted")

        root = Path(root)
        det_cfg = Config.fromfile(str(root / "third_party/BEVFormer/projects/configs/bevformer/bevformer_tiny.py"))
        map_cfg = Config.fromfile(str(root / "third_party/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py"))
        occ_cfg = Config.fromfile(str(root / "third_party/OccFormer/projects/configs/occformer_nusc/occformer_nusc_r50_256x704.py"))
        det_cfg.model.pretrained = None
        self.source_range = det_cfg.point_cloud_range
        self.map_range = map_cfg.point_cloud_range
        map_head = copy.deepcopy(map_cfg.model.pts_bbox_head)
        map_head.transformer.encoder = ConfigDict(type="DriveSceneBEVFormerEncoder")
        map_head.train_cfg = map_cfg.model.train_cfg.pts
        # Aux image/depth supervision is intentionally disabled for the shared bridge.
        map_head.aux_seg = dict(use_aux_seg=False, bev_seg=False, pv_seg=False)
        self.occ_shape = [128, 128, 16]
        if smoke:
            det_head = det_cfg.model.pts_bbox_head
            det_head.bev_h = det_head.bev_w = 12
            det_head.num_query = 12
            det_head.positional_encoding.row_num_embed = det_head.positional_encoding.col_num_embed = 12
            det_head.transformer.encoder.num_layers = 1
            det_head.transformer.decoder.num_layers = 1
            map_head.bev_h = map_head.bev_w = 12
            map_head.num_vec_one2one = 4
            map_head.num_vec_one2many = 4
            map_head.num_pts_per_vec = map_head.num_pts_per_gt_vec = 5
            map_head.transformer.decoder.num_layers = 1
            map_head.transformer.decoder.transformerlayers.num_vec = 8
            map_head.transformer.decoder.transformerlayers.num_pts_per_vec = 5
            map_head.positional_encoding.row_num_embed = map_head.positional_encoding.col_num_embed = 12
            self.occ_shape = [16, 16, 8]
            occ_cfg.model.img_bev_encoder_backbone.block_numbers = [1, 1, 1, 1]
            occ_cfg.model.img_bev_encoder_neck.encoder.num_layers = 1
            occ_cfg.model.pts_bbox_head.transformer_decoder.num_layers = 1
            occ_cfg.model.pts_bbox_head.num_queries = 8
        self.detector = build_model(det_cfg.model)
        self.map_head = build_head(map_head)
        self.map_head.transformer.get_bev_features = MethodType(shared_map_features, self.map_head.transformer)
        self.height_lift = nn.Conv2d(256, 128*self.occ_shape[2], 1)
        self.occ_backbone = build_backbone(occ_cfg.model.img_bev_encoder_backbone)
        self.occ_neck = build_neck(occ_cfg.model.img_bev_encoder_neck)
        self.occ_head = build_head(occ_cfg.model.pts_bbox_head)
        self.det_cfg = det_cfg
        self.smoke_mode = smoke

    def map_crop(self, feature):
        height, width = self.map_head.bev_h, self.map_head.bev_w
        low, high = self.map_range[:2], self.map_range[3:5]
        x = torch.linspace(low[0], high[0], width+1, device=feature.device)
        y = torch.linspace(low[1], high[1], height+1, device=feature.device)
        yy, xx = torch.meshgrid((y[:-1]+y[1:])/2, (x[:-1]+x[1:])/2, indexing="ij")
        xx = 2*(xx-self.source_range[0])/(self.source_range[3]-self.source_range[0])-1
        yy = 2*(yy-self.source_range[1])/(self.source_range[4]-self.source_range[1])-1
        grid = torch.stack([xx, yy], -1)[None].expand(feature.shape[0], -1, -1, -1)
        return F.grid_sample(feature, grid, align_corners=False)

    def forward(self, images, metadata, previous_bev=None):
        # Exactly one image backbone/FPN and one BEVFormer encoder invocation.
        features = self.detector.extract_img_feat(images, metadata)
        detection = self.detector.pts_bbox_head(features, metadata, prev_bev=previous_bev)
        head = self.detector.pts_bbox_head
        shared = detection["bev_embed"]  # [HW,B,C]
        bev = shared.permute(1, 2, 0).reshape(len(metadata), 256, head.bev_h, head.bev_w)
        cropped = self.map_crop(bev).flatten(2).transpose(1, 2)
        vectors = self.map_head(features, None, metadata, prev_bev=cropped)
        lifted = self.height_lift(F.interpolate(bev, size=self.occ_shape[:2], mode="bilinear", align_corners=False))
        b, _, h, w = lifted.shape
        # BEVFormer is [Y,X]; OccFormer uses [X,Y,Z].
        volume = lifted.reshape(b, 128, self.occ_shape[2], h, w).permute(0, 1, 4, 3, 2).contiguous()
        occ_features = self.occ_neck(self.occ_backbone(volume))
        occupancy = self.occ_head(occ_features, metadata)
        return {"detection": detection, "map": vectors, "occupancy": occupancy, "bev": shared}

    def load_component_weights(self, detection=None, mapping=None, occupancy=None):
        """Initialization report only; does not assert joint checkpoint validity."""
        reports = {}
        for label, path in [("detection", detection), ("map", mapping), ("occupancy", occupancy)]:
            if path is None:
                continue
            state = torch.load(path, map_location="cpu")
            state = state.get("state_dict", state)
            if label == "detection":
                result = self.detector.load_state_dict(state, strict=False)
                reports[label] = {"missing": result.missing_keys, "unexpected": result.unexpected_keys}
            else:
                components = [("pts_bbox_head.", self.map_head)] if label == "map" else [
                    ("img_bev_encoder_backbone.", self.occ_backbone), ("img_bev_encoder_neck.", self.occ_neck),
                    ("pts_bbox_head.", self.occ_head)]
                reports[label] = {}
                for prefix, module in components:
                    subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                    result = module.load_state_dict(subset, strict=False)
                    reports[label][prefix] = {"missing": result.missing_keys, "unexpected": result.unexpected_keys}
        reports["joint_training_required"] = True
        return reports
