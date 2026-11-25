# DriveScene

Training and inference intermediates stay under the local `outputs/` directory.
`video` / `mcap` artifacts also default to `outputs/`. Set
`DRIVESCENE_ARTIFACT_OUTPUT` to write them to a shared filesystem, for example
`/mnt/sh_data/common3share/17_端到端/users/wzy/DriveScene/outputs`.


## 当前状态

- nuScenes mini：404 帧，`mini-v3-with-occ.jsonl` 已将地图真值严格裁剪到 ego `[-30,30]² m`；12,922 条折线、越界点为 0、单帧最多 74 条。404 帧均使用重新生成的射线 LiDARseg 占据监督（2,709,631 个已标注体素，其中 1,956,752 个 free）。
- 官方 BEVFormer-tiny：权重完整，已在真实六相机样本上严格加载并推理成功。
- 官方共享结构：BEVFormer 编码器只运行一次，输出同时送入检测、MapTRv2 和 OccFormer 分支；真实相机前向、反向均已验证。
- 原生联调模型：地图、检测、占据三分支可训练、推理和编译 WorldModel。
- `outputs/train-all-tasks-smoke/last.pt` 仅训练 2 步，只用于接口检查，不代表模型精度。
- 原生 perception-v3 使用 80 个地图查询与 128 个检测查询；检测标注严格裁剪到 BEV，采用 focal、归一化框、方向和 BEV-IoU 损失；地图包含方向、曲率、拓扑、融合后 NMS；占据采用逐高度可见性和空间过滤。旧 checkpoint 与新结构不兼容。
- MapTRv2 官方 nuScenes 配置的模型权重在上游表格中仍标为 WIP；OccFormer 下载未完成，`.part` 文件不可使用。
- 地图标注改为 road_segment/lane 与 ped_crossing 先 union 再取外轮廓；旧 `mini-v3*.jsonl` 含内部拼缝，需重新 `prepare`。闭合折线按起点平移匹配，匈牙利代价已按网格归一化。
- 跟踪改为 CenterPoint 风格的全局中心距离 + 匀速预测，不再使用图像 ByteTrack。
- 训练使用 DataLoader 预取、warmup+cosine 学习率、场景级 BEV 旋转/翻转和颜色抖动，并在保存 checkpoint 时记录 val loss。
- `run --limit` 默认处理整个 manifest。官方检测/跟踪指标仍按 50 m / 40 m 协议计算，但模型输出只有 ±30 m，范围外真值会记为漏检。

## 进入项目和环境检查

```bash
ssh 42
cd /Data1/wzy/DriveScene
scripts/run.sh doctor
nvidia-smi
```

所有命令都应从项目根目录执行。`scripts/run.sh` 使用 Python 3.10 工程环境；官方旧版 OpenMMLab 模块使用 `.venv-openmmlab/bin/python`。

## 1. 直接查看完整流程（无需训练）

标注回放用于验证数据、坐标、跟踪、场景编译和可视化。它使用真值目标和地图，因此不是感知模型效果。

```bash
scripts/run.sh run \
  --mode annotations \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --limit 20 \
  --output outputs/annotation-demo
```

结果在 `outputs/annotation-demo/`：每帧一个 WorldModel JSON、占据 NPZ，前 3 帧还会生成 PNG 总览。

将同一场景的逐帧结果编码为六相机 + BEV 可视化视频：

```bash
scripts/run.sh video \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --world outputs/annotation-demo \
  --output outputs/annotation-demo/scene.mp4 \
  --fps 2
```

The sparse LiDARseg occupancy preparation ray-traces free space before each
return. It is still sparse supervision rather than dense Occ3D ground truth,
but unlike endpoint-only labels it includes the `free` class and avoids the
all-occupied training collapse.

Inference retains low-confidence occupancy cells as unknown. The class-specific
thresholds are recorded in output provenance and configured under
`compiler.occupancy_confidence`; this represents uncertainty from sparse labels
rather than inventing dense occupancy in unsupervised voxels.

Run inference on multiple GPUs by assigning complete scenes to workers (temporal
BEV memory and tracking never cross worker boundaries):

```bash
.venv/bin/python scripts/infer_multi_gpu.py \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --checkpoint outputs/train-mini-perception-v3-ddp/last.pt \
  --output outputs/mini-infer-perception-v3 \
  --gpus 0,1,2,3
```

Train perception-v3 from scratch with PyTorch DistributedDataParallel. The wrapper
refuses to overwrite an existing run and validates grid/query capacity before
the first optimizer step:

```bash
GPUS=0,1,2,3 \
STEPS=10000 \
OUTPUT=outputs/train-mini-perception-v3-ddp \
scripts/train_perception_v3_ddp.sh
```

Training atomically updates `latest.pt` every 250 steps. Resume to a target
total step count with `--resume outputs/train-mini-perception-v3-ddp/latest.pt`; `--steps`
is the final target, not the number of additional steps.

With four GPUs, 2,500 DDP updates process approximately the same number of
training samples as 10,000 single-GPU updates. Use 10,000 DDP updates when the
intent is four times as many sample exposures and a global batch size of four.

Use `--view 3d` to render occupied/drivable voxels, 3D agent boxes and map
vectors in the left panel while keeping the six cameras in a 2x3 wall:

```bash
scripts/run.sh video \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --world outputs/mini-infer-perception-v3 \
  --output outputs/mini-infer-perception-v3/scene-0757-3d.mp4 \
  --scene-name scene-0757 \
  --fps 2 \
  --view 3d
```

Export the synchronized scene as a Foxglove-native MCAP recording:

```bash
scripts/run.sh mcap \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --world outputs/mini-infer-perception-v3 \
  --output outputs/mini-infer-perception-v3/scene-0757.mcap \
  --scene-name scene-0757
```

Open the `.mcap` file in Foxglove Studio. Set the 3D panel fixed frame to
`base_link`, enable `/drivescene/occupancy` and `/drivescene/scene`, then add
six Image panels for the `/drivescene/camera/*` topics. Camera calibration
topics and `base_link` transforms are embedded for optional image projection
in the 3D panel. The scene also contains an `ego_vehicle` model at the origin.

## Validation metrics and error video

Evaluate the compiled validation split with MapTR-style Chamfer AP, official
nuScenes detection/tracking metrics, and sparse observed-voxel occupancy IoU:

```bash
scripts/run.sh evaluate \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --world outputs/mini-infer-perception-v3 \
  --output outputs/mini-infer-perception-v3-evaluation \
  --split val \
  --fps 2
```

The JSON and Markdown reports stay in the local evaluation directory. The
TP/FP/FN MP4 is written to the configured shared `artifact_output` directory.
Map AP uses symmetric Chamfer thresholds 0.5/1.0/1.5 m; the benchmark mAP is
the mean over divider, pedestrian crossing, and boundary. Centerline and stop
line are reported separately. Occupancy is explicitly labeled as the project's
sparse LiDARseg-derived metric because these labels are not Occ3D `mask_camera`
ground truth. Traffic-light metrics and historical latency remain unavailable
until their labels and timing records exist.

默认导出 manifest 中第一个场景。可以用 `--scene-name scene-0061` 指定场景；推理目录必须包含该场景全部帧。

## 2. 官方 BEVFormer 真实检测（无需训练）

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
.venv-openmmlab/bin/python scripts/infer_bevformer.py \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --checkpoint checkpoints/bevformer_tiny_epoch_24.pth \
  --device cuda \
  --output outputs/bevformer-official
```

如果暂时没有空闲 GPU，把 `--device cuda` 改为 `--device cpu`。当前脚本处理一个样本，用来验证官方权重和标定链路。

## 3. 共享 BEV 官方模块连接检查（无需训练，但不是有效预测）

```bash
.venv-openmmlab/bin/python scripts/smoke_official_shared.py \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --backward
```

该命令验证一次 ResNet/FPN + 一次 BEVFormer 编码后，MapTRv2、检测和 OccFormer 分支均可前向和反传。随机新增的 BEV-to-volume 桥接层尚未训练，所以输出不能用于评价。

## 4. 训练可运行的原生共享模型

先重新生成 union-outline 地图 manifest（旧 `mini-v3*.jsonl` 含 road_segment 内部拼缝），再生成带射线占据监督的训练 manifest：

```bash
scripts/run.sh prepare \
  --root data/nuscenes \
  --version v1.0-mini \
  --output data/processed/mini-v3.jsonl

scripts/run.sh prepare-sparse-occupancy \
  --manifest data/processed/mini-v3.jsonl \
  --output data/processed/mini-v3-with-occ.jsonl
```

单卡从头训练可以执行：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> scripts/run.sh train \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --steps 10000 \
  --device cuda \
  --backbone checkpoints/resnet50-11ad3fa6.pth \
  --output outputs/train-mini-perception-v3
```

断开 SSH 前建议放进 `tmux`。训练完成后：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> scripts/run.sh run \
  --mode native_research \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --checkpoint outputs/train-mini-perception-v3/last.pt \
  --lights \
  --limit 404 \
  --device cuda \
  --output outputs/mini-infer-perception-v3
```

生成模型预测视频：

```bash
scripts/run.sh video \
  --manifest data/processed/mini-v3-with-occ.jsonl \
  --world outputs/mini-infer-perception-v3 \
  --output outputs/mini-infer-perception-v3/scene-0061.mp4 \
  --scene-name scene-0061 \
  --fps 2
```

这套原生模型适合验证训练闭环。mini 只有 10 个场景，训练结果只能作为技术演示。

## 是否需要训练

| 目标 | 是否训练 | 说明 |
|---|---:|---|
| 看完整数据和 WorldModel 输出 | 否 | 使用 annotation replay，属于真值回放 |
| 使用官方 BEVFormer 检测车辆/行人 | 否 | 官方权重已验证 |
| 原生共享 BEV 三任务演示 | 是 | 可用上述 `train` 命令 |
| BEVFormer + MapTRv2 + OccFormer 完整共享模型 | 是 | 新桥接层和共享特征分布必须联合训练 |
| 红黄绿灯状态 | 是 | COCO YOLOv8 只能找 traffic light，不能可靠输出状态 |
| 可用于规划的稳定 WorldModel | 是，并且要完整评测 | 需要全量数据、稠密占据标注和闭环验证 |

## 完整训练前还缺什么

1. 全量 nuScenes trainval。当前目录只有 `v1.0-mini`；应把 trainval 放到 `data/nuscenes`，共享盘空间充足时可在共享盘存放后软链接。
2. 稠密占据标注，例如 Occ3D-nuScenes/OpenOccupancy。现有 LiDARseg 只提供激光扫到位置的稀疏监督，未观测体素仍是 unknown。
3. MapTRv2 上游 nuScenes R50 配置提供了正式训练代码，但其 README 模型项仍为 WIP，没有可核验的官方 MapTRv2 checkpoint。全量数据和 CAN bus 准备好后，可运行 `scripts/train_maptrv2_full.sh` 按上游配置从 ResNet-50 预训练骨干开始训练。
4. 完整 OccFormer 权重。`checkpoints/occformer_nusc_r50.pth.part` 是未完成下载，不能加载。
5. 交通灯状态数据和权重。nuScenes 基础数据没有逐帧红黄绿真值，应接入专门交通灯数据或自有标注。

## 关键验证结果

```bash
.venv/bin/python -m pytest -q
```

当前为 13 项测试通过；perception-v3 还完成了真实六相机单卡单步训练、双卡 DDP 单步训练和 2 帧端到端推理检查，详见 `outputs/train-mini-perception-v3-smoke-20260921`、`outputs/train-mini-perception-v3-ddp-smoke-20260921` 与 `outputs/mini-perception-v3-infer-smoke-20260921`。


