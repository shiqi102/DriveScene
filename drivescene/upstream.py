"""Selective upstream registrations, avoiding unused DD3D/GKT/CUDA lift modules.

Run with .venv-openmmlab (legacy MMDetection 2.x). Namespace paths point to
unmodified official sources. Only components actually used by shared BEV load.
"""
from pathlib import Path
import importlib
import sys
import types


def namespace(name, paths):
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(p) for p in paths]
    module.__package__ = name
    sys.modules[name] = module


def register(root, include_map=True, include_occ=True):
    root = Path(root).resolve()
    map_root = root / "third_party/MapTR/projects/mmdet3d_plugin"
    occ_root = root / "third_party/OccFormer/projects/mmdet3d_plugin"
    namespace("projects", [map_root.parent, occ_root.parent])
    namespace("projects.mmdet3d_plugin", [map_root, occ_root])
    for part in ["core", "core.bbox", "core.bbox.assigners", "core.bbox.coders", "core.bbox.match_costs",
                 "bevformer", "bevformer.dense_heads", "bevformer.detectors", "bevformer.modules", "models", "models.utils",
                 "maptr", "maptr.modules", "maptr.dense_heads", "maptr.losses", "maptr.assigners"]:
        namespace("projects.mmdet3d_plugin."+part, [map_root.joinpath(*part.split("."))])
    modules = ["core.bbox.assigners.hungarian_assigner_3d", "core.bbox.coders.nms_free_coder",
               "core.bbox.match_costs.match_cost", "bevformer.modules.temporal_self_attention",
               "bevformer.modules.spatial_cross_attention", "bevformer.modules.decoder",
               "bevformer.modules.encoder", "bevformer.modules.transformer",
               "bevformer.dense_heads.bevformer_head", "bevformer.detectors.bevformer"]
    if include_map:
        modules += ["maptr.modules.decoder", "maptr.modules.transformer", "maptr.dense_heads.maptr_head",
                    "maptr.dense_heads.maptrv2_head", "maptr.losses.map_loss", "maptr.losses.simple_loss", "maptr.assigners.maptr_assigner"]
    if include_occ:
        for part in ["occformer", "occformer.backbones", "occformer.necks", "occformer.mask2former"]:
            namespace("projects.mmdet3d_plugin."+part, [occ_root.joinpath(*part.split("."))])
        modules += ["occformer.mask2former.positional_encodings",
                    "occformer.backbones.occnet", "occformer.necks.multiscale_deformattn_3d",
                    "occformer.mask2former.mask2former_nusc_occ"]
    for name in modules:
        importlib.import_module("projects.mmdet3d_plugin."+name)
    return modules
