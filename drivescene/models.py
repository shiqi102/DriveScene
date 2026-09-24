"""Native research backend, NOT checkpoint-compatible official implementations.

Provides a trainable shared BEV integration prototype. Names intentionally describe
mechanisms rather than claiming paper-equivalent BEVFormer/MapTRv2/OccFormer.
Official sources/configs live in third_party and use separate upstream runtimes.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ResNet50FPN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        from torchvision.models import resnet50
        from torchvision.models.feature_extraction import create_feature_extractor
        from torchvision.ops import FeaturePyramidNetwork
        self.body = create_feature_extractor(resnet50(weights=None),
            return_nodes={"layer2": "c3", "layer3": "c4", "layer4": "c5"})
        self.fpn = FeaturePyramidNetwork([512, 1024, 2048], channels)

    def forward(self, images):
        batch, cameras, channels, height, width = images.shape
        features = self.fpn(self.body(images.reshape(batch*cameras, channels, height, width)))
        return [f.reshape(batch, cameras, *f.shape[1:]) for f in features.values()]


class ProjectedBEVEncoder(nn.Module):
    """Multi-height geometric sampling + learned local offsets + temporal attention.

    This is a portable research adaptation, not the official BEVFormer encoder.
    Temporal memory is warped using full current/previous ego poses before fusion.
    """
    def __init__(self, grid, channels, heads):
        super().__init__()
        self.grid = grid
        nx, ny, self.nz = grid["shape"]
        xs = torch.linspace(grid["x"][0], grid["x"][1], nx+1)
        ys = torch.linspace(grid["y"][0], grid["y"][1], ny+1)
        xx, yy = torch.meshgrid((xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2, indexing="ij")
        self.register_buffer("xy", torch.stack([xx, yy], -1))
        self.register_buffer("heights", torch.linspace(*grid["z"], 4))
        self.queries = nn.Parameter(torch.randn(nx*ny, channels)*0.02)
        self.offsets = nn.Linear(channels, 8)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.sample_weights = nn.Linear(channels, 4)
        self.temporal = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels*2), nn.GELU(), nn.Linear(channels*2, channels))

    def warp(self, previous, current_to_previous):
        batch = previous.shape[0]
        xy = self.xy.to(previous)
        xyz1 = torch.cat([xy, torch.zeros_like(xy[..., :1]), torch.ones_like(xy[..., :1])], -1)
        coords = torch.einsum("bij,xyj->bxyi", current_to_previous, xyz1)
        gx = 2*(coords[..., 0]-self.grid["x"][0])/(self.grid["x"][1]-self.grid["x"][0])-1
        gy = 2*(coords[..., 1]-self.grid["y"][0])/(self.grid["y"][1]-self.grid["y"][0])-1
        # Tensor dimensions are [B,C,X,Y], grid_sample uses horizontal Y then X.
        return F.grid_sample(previous, torch.stack([gy, gx], -1), align_corners=False)

    def forward(self, features, projection, image_shape, previous=None, current_to_previous=None):
        batch, cameras = projection.shape[:2]
        nx, ny, _ = self.grid["shape"]
        count, channels = self.queries.shape
        query = self.queries.unsqueeze(0).expand(batch, -1, -1)
        xyz1 = torch.cat([self.xy[:, :, None].expand(-1, -1, 4, -1),
                         self.heights[None, None, :, None].expand(nx, ny, -1, -1),
                         self.xy.new_ones(nx, ny, 4, 1)], -1).reshape(count, 4, 4)
        projected = torch.einsum("bnij,qkj->bnqki", projection, xyz1)
        depth = projected[..., 2]
        uv = projected[..., :2]/depth.clamp(min=1e-4).unsqueeze(-1)
        height, width = image_shape
        uv = uv/uv.new_tensor([width, height])*2-1
        offsets = self.offsets(query).reshape(batch, count, 4, 2).tanh()*0.02
        uv = uv + offsets[:, None]
        visible = (depth > 1e-4) & (uv.abs() <= 1).all(-1)
        weights = self.sample_weights(query).softmax(-1)[:, None] * visible
        aggregate = query.new_zeros(batch, channels, count)
        denominator = weights.sum((1, 3)).clamp(min=1e-6).unsqueeze(1)
        for feature in features:
            sampled = F.grid_sample(feature.flatten(0, 1), uv.flatten(0, 1), align_corners=False)
            sampled = sampled.reshape(batch, cameras, channels, count, 4)
            aggregate += (sampled*weights[:, :, None]).sum((1, 4))/denominator
        tokens = query + aggregate.transpose(1, 2)/len(features)
        if previous is not None:
            if current_to_previous is None:
                raise ValueError("Temporal BEV requires ego-motion transform")
            warped = self.warp(previous, current_to_previous).flatten(2).transpose(1, 2)
            memory = torch.stack([tokens, warped], dim=2).reshape(batch*count, 2, channels)
            fused, _ = self.temporal(tokens.reshape(batch*count, 1, channels), memory, memory, need_weights=False)
            tokens = tokens + fused.reshape(batch, count, channels)
        tokens = self.norm(tokens + self.ffn(tokens))
        observed_four = visible.any(1).reshape(batch, nx, ny, 4)
        height_indices = torch.linspace(0, 3, self.nz, device=visible.device).round().long()
        observed_volume = observed_four.index_select(-1, height_indices)
        return tokens.transpose(1, 2).reshape(batch, channels, nx, ny), observed_volume


class VectorMapHead(nn.Module):
    """Hierarchical instance/point queries with reversible polyline supervision."""
    def __init__(self, cfg, grid):
        super().__init__()
        c, self.n, self.p = cfg["channels"], cfg["map_queries"], cfg["map_points"]
        self.instances = nn.Parameter(torch.randn(self.n, c)*.02)
        self.points = nn.Parameter(torch.randn(self.p, c)*.02)
        layer = nn.TransformerDecoderLayer(c, cfg["heads"], c*2, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg["layers"])
        self.classifier = nn.Linear(c, len(cfg["map_classes"])+1)
        self.regressor = nn.Linear(c, 2)
        self.register_buffer("minimum", torch.tensor([grid["x"][0], grid["y"][0]]))
        self.register_buffer("extent", torch.tensor([grid["x"][1]-grid["x"][0], grid["y"][1]-grid["y"][0]]))

    def forward(self, bev):
        queries = (self.instances[:, None]+self.points[None]).flatten(0, 1)
        memory = F.adaptive_avg_pool2d(bev, (min(20, bev.shape[2]), min(20, bev.shape[3])))
        output = self.decoder(queries[None].expand(bev.shape[0], -1, -1), memory.flatten(2).transpose(1, 2))
        output = output.reshape(bev.shape[0], self.n, self.p, -1)
        return {"logits": self.classifier(output.mean(2)), "points": self.regressor(output).sigmoid()*self.extent+self.minimum}


class DetectionHead(nn.Module):
    def __init__(self, cfg, grid):
        super().__init__()
        c = cfg["channels"]
        self.queries = nn.Parameter(torch.randn(cfg["detection_queries"], c)*.02)
        layer = nn.TransformerDecoderLayer(c, cfg["heads"], c*2, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg["layers"])
        self.classifier = nn.Linear(c, len(cfg["detection_classes"])+1)
        self.regressor = nn.Linear(c, 10)
        self.register_buffer("minimum", torch.tensor([grid[k][0] for k in ("x", "y", "z")]))
        self.register_buffer("extent", torch.tensor([grid[k][1]-grid[k][0] for k in ("x", "y", "z")]))

    def forward(self, bev):
        memory = F.adaptive_avg_pool2d(bev, (min(20, bev.shape[2]), min(20, bev.shape[3])))
        output = self.decoder(self.queries[None].expand(bev.shape[0], -1, -1), memory.flatten(2).transpose(1, 2))
        raw = self.regressor(output)
        boxes = torch.cat([raw[..., :3].sigmoid()*self.extent+self.minimum,
                           F.softplus(raw[..., 3:6])+.1, raw[..., 6:]], -1)
        return {"logits": self.classifier(output), "boxes": boxes}


class DualPathOccupancyHead(nn.Module):
    """Learned BEV-to-volume lift, local 3D path and global planar attention.

    Not the official OccFormer Mask2Former decoder; this bridge must be trained.
    """
    def __init__(self, cfg, grid):
        super().__init__()
        c = cfg["channels"]
        self.height = nn.Parameter(torch.randn(1, c, 1, 1, grid["shape"][2])*.02)
        def volume_block():
            return nn.Sequential(nn.Conv3d(c, c, 3, padding=1, groups=c),
                                 nn.GroupNorm(8, c), nn.GELU(), nn.Conv3d(c, c, 1))
        self.local_first = volume_block()
        self.local_second = volume_block()
        self.global_path = nn.TransformerEncoderLayer(c, cfg["heads"], c*2, dropout=0, batch_first=True)
        self.classifier = nn.Conv3d(c, len(cfg["occupancy_classes"]), 1)

    def forward(self, bev):
        volume = bev.unsqueeze(-1)+self.height
        planar = F.adaptive_avg_pool2d(bev, (min(10, bev.shape[2]), min(10, bev.shape[3])))
        global_features = self.global_path(planar.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(planar)
        global_features = F.interpolate(global_features, size=bev.shape[2:], mode="bilinear", align_corners=False)
        volume = volume+self.local_first(volume)
        volume = volume+self.local_second(volume)
        return self.classifier(volume+global_features.unsqueeze(-1))


class SharedBEVModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg, grid = config["model"], config["grid"]
        if cfg["backend"] != "native_research":
            raise ValueError("This class implements only the explicitly named native_research backend")
        self.encoder = ResNet50FPN(cfg["channels"])
        self.bev_encoder = ProjectedBEVEncoder(grid, cfg["channels"], cfg["heads"])
        self.map_head = VectorMapHead(cfg, grid)
        self.detection_head = DetectionHead(cfg, grid)
        self.occupancy_head = DualPathOccupancyHead(cfg, grid)

    def forward(self, images, projections, previous=None, current_to_previous=None):
        if images.shape[1] != 6:
            raise ValueError("Exactly six cameras are required")
        features = self.encoder(images)
        bev, observed = self.bev_encoder(features, projections, images.shape[-2:], previous, current_to_previous)
        return {"bev": bev, "observed": observed, "map": self.map_head(bev),
                "detection": self.detection_head(bev), "occupancy": self.occupancy_head(bev)}


def _focal_softmax_loss(logits, labels, gamma=2.0, no_object_weight=.5):
    probabilities = logits.softmax(-1)
    selected = probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-8)
    weights = logits.new_ones(len(labels))
    weights[labels == logits.shape[-1]-1] = float(no_object_weight)
    return (weights*(1-selected).pow(float(gamma))*(-selected.log())).sum()/weights.sum().clamp_min(1)


def _matched_bev_iou(first, second):
    """Differentiable axis-aligned BEV IoU used as a stable coarse box loss."""
    first_half, second_half = first[:, 3:5]/2, second[:, 3:5]/2
    first_min, first_max = first[:, :2]-first_half, first[:, :2]+first_half
    second_min, second_max = second[:, :2]-second_half, second[:, :2]+second_half
    intersection = (torch.minimum(first_max, second_max)-
                    torch.maximum(first_min, second_min)).clamp_min(0).prod(1)
    first_area = first[:, 3]*first[:, 4]
    second_area = second[:, 3]*second[:, 4]
    return intersection/(first_area+second_area-intersection).clamp_min(1e-6)


def supervised_loss(pred, targets, config):
    """Hungarian matching; missing tasks produce no loss, never fake negatives."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    cfg = config["model"]
    losses = {}
    for branch, field, classes in [("map", "map_vectors", cfg["map_classes"]),
                                    ("detection", "boxes", cfg["detection_classes"])]:
        branch_losses = []
        for index, target in enumerate(targets):
            items = target.get(field)
            if items is None:
                continue
            items = [item for item in items if item["kind"] in classes]
            logits = pred[branch]["logits"][index]
            labels = torch.full((len(logits),), len(classes), dtype=torch.long, device=logits.device)
            regression = logits.sum()*0
            map_terms = {}
            if items:
                gt_labels = torch.tensor([classes.index(item["kind"]) for item in items], device=logits.device)
                if branch == "map":
                    gt = logits.new_tensor([item["points"] for item in items])
                    points = pred[branch]["points"][index]
                    forward = (points[:, None]-gt[None]).abs().mean((2, 3))
                    reverse = (points[:, None]-gt.flip(1)[None]).abs().mean((2, 3))
                    costs = torch.minimum(forward, reverse)
                else:
                    gt = logits.new_tensor([item["center"]+item["size"]+[np.sin(item["yaw"]), np.cos(item["yaw"])]+item["velocity"] for item in items])
                    boxes = pred[branch]["boxes"][index]
                    extent = logits.new_tensor([config["grid"][axis][1]-config["grid"][axis][0]
                                                for axis in ("x", "y", "z")])
                    center_cost = torch.cdist(boxes[:, :3]/extent, gt[:, :3]/extent, p=1)
                    size_cost = torch.cdist(boxes[:, 3:6].clamp_min(.1).log(),
                                            gt[:, 3:6].clamp_min(.1).log(), p=1)/3
                    pred_angle = F.normalize(boxes[:, 6:8], dim=-1, eps=1e-6)
                    gt_angle = F.normalize(gt[:, 6:8], dim=-1, eps=1e-6)
                    angle_cost = 1-pred_angle@gt_angle.T
                    costs = 2*center_cost+size_cost+angle_cost
                costs = costs-logits.softmax(-1)[:, gt_labels]
                rows, cols = linear_sum_assignment(costs.detach().cpu().numpy())
                rows = torch.as_tensor(rows, device=logits.device)
                cols = torch.as_tensor(cols, device=logits.device)
                labels[rows] = gt_labels[cols]
                if branch == "map":
                    matched = points[rows]
                    selected = gt[cols]
                    use_reverse = reverse[rows, cols] < forward[rows, cols]
                    selected = torch.where(use_reverse[:, None, None], selected.flip(1), selected)
                    map_terms["point"] = F.smooth_l1_loss(matched, selected)

                    pred_delta = matched[:, 1:]-matched[:, :-1]
                    gt_delta = selected[:, 1:]-selected[:, :-1]
                    pred_direction = F.normalize(pred_delta, dim=-1, eps=1e-6)
                    gt_direction = F.normalize(gt_delta, dim=-1, eps=1e-6)
                    map_terms["direction"] = (1-(pred_direction*gt_direction).sum(-1)).mean()

                    pred_curve = matched[:, 2:]-2*matched[:, 1:-1]+matched[:, :-2]
                    gt_curve = selected[:, 2:]-2*selected[:, 1:-1]+selected[:, :-2]
                    map_terms["smoothness"] = F.smooth_l1_loss(pred_curve, gt_curve)

                    if len(rows) > 1:
                        pred_endpoints = matched[:, [0, -1]]
                        gt_endpoints = selected[:, [0, -1]]
                        count = len(rows)
                        pred_connectivity = torch.cdist(
                            pred_endpoints.flatten(0, 1), pred_endpoints.flatten(0, 1)
                        ).reshape(count, 2, count, 2).amin((1, 3))
                        gt_connectivity = torch.cdist(
                            gt_endpoints.flatten(0, 1), gt_endpoints.flatten(0, 1)
                        ).reshape(count, 2, count, 2).amin((1, 3))
                        radius = float(config.get("training", {}).get("map_loss", {}).get("topology_radius", 2.0))
                        connected = (gt_connectivity < radius) & ~torch.eye(
                            count, dtype=torch.bool, device=logits.device)
                        if connected.any():
                            map_terms["topology"] = F.smooth_l1_loss(
                                pred_connectivity[connected], gt_connectivity[connected])
                else:
                    settings = config.get("training", {}).get("detection_loss", {})
                    matched, selected = boxes[rows], gt[cols]
                    extent = logits.new_tensor([config["grid"][axis][1]-config["grid"][axis][0]
                                                for axis in ("x", "y", "z")])
                    center_loss = F.smooth_l1_loss(matched[:, :3]/extent, selected[:, :3]/extent)
                    size_loss = F.smooth_l1_loss(matched[:, 3:6].clamp_min(.1).log(),
                                                 selected[:, 3:6].clamp_min(.1).log())
                    pred_angle = F.normalize(matched[:, 6:8], dim=-1, eps=1e-6)
                    gt_angle = F.normalize(selected[:, 6:8], dim=-1, eps=1e-6)
                    orientation_loss = (1-(pred_angle*gt_angle).sum(-1)).mean()
                    iou_loss = (1-_matched_bev_iou(matched, selected)).mean()
                    regression = (float(settings.get("center", 5.0))*center_loss+
                                  float(settings.get("size", 2.0))*size_loss+
                                  float(settings.get("orientation", 1.0))*orientation_loss+
                                  float(settings.get("bev_iou", 2.0))*iou_loss)
                    valid_velocity = torch.tensor([items[int(j)].get("velocity_valid", True) for j in cols], device=logits.device)
                    if valid_velocity.any():
                        regression = regression+float(settings.get("velocity", .5))*F.smooth_l1_loss(
                            matched[valid_velocity, 8:], selected[valid_velocity, 8:])
            class_weights = logits.new_ones(len(classes)+1)
            class_weights[-1] = .1
            classification = F.cross_entropy(logits, labels, weight=class_weights)
            if branch == "map":
                settings = config.get("training", {}).get("map_loss", {})
                points = pred[branch]["points"][index]
                if len(points) > 1:
                    direct = (points[:, None]-points[None]).abs().mean((2, 3))
                    reversed_distance = (points[:, None]-points.flip(1)[None]).abs().mean((2, 3))
                    pair_distance = torch.minimum(direct, reversed_distance)
                    foreground = logits.softmax(-1)[:, :-1]
                    same_class_probability = foreground@foreground.transpose(0, 1)
                    unique_pairs = torch.triu(torch.ones_like(pair_distance, dtype=torch.bool), diagonal=1)
                    margin = float(settings.get("duplicate_margin", 1.5))
                    map_terms["duplicate"] = (
                        F.relu(margin-pair_distance)*same_class_probability
                    )[unique_pairs].mean()
                value = float(settings.get("classification", 1.0))*classification
                for name, term in map_terms.items():
                    value = value+float(settings.get(name, 1.0))*term
                branch_losses.append(value)
            else:
                settings = config.get("training", {}).get("detection_loss", {})
                classification = _focal_softmax_loss(
                    logits, labels, settings.get("focal_gamma", 2.0),
                    settings.get("no_object_weight", .5))
                branch_losses.append(float(settings.get("classification", 2.0))*classification+regression)
        if branch_losses:
            losses[branch] = torch.stack(branch_losses).mean()
    occ_losses = []
    for index, target in enumerate(targets):
        if target.get("occupancy") is not None:
            labels = torch.as_tensor(target["occupancy"], device=pred["occupancy"].device, dtype=torch.long)
            if labels.shape != pred["occupancy"].shape[2:]:
                raise ValueError("Occupancy labels must be resampled to the configured ego grid")
            if (labels != 255).any():
                weights = config.get("training", {}).get("occupancy_class_weights")
                weights = None if weights is None else pred["occupancy"].new_tensor(weights)
                occ_losses.append(F.cross_entropy(pred["occupancy"][index:index+1], labels[None],
                                                  weight=weights, ignore_index=255))
    if occ_losses:
        losses["occupancy"] = torch.stack(occ_losses).mean()
    if not losses:
        raise ValueError("No usable supervision for this batch")
    return losses
