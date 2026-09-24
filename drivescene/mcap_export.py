from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import read_manifest
from .geometry import grid_centers


def _timestamp(timestamp_us):
    from google.protobuf.timestamp_pb2 import Timestamp

    seconds, microseconds = divmod(int(timestamp_us), 1_000_000)
    return Timestamp(seconds=seconds, nanos=microseconds*1_000)


def _point_cloud(semantic, grid, timestamp):
    from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
    from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud

    centers = grid_centers(grid)
    mask = (semantic == 1) | (semantic == 2)
    xyz = centers[mask].astype("<f4", copy=False)
    labels = semantic[mask]
    # x/y/z float32 followed by red/green/blue/alpha uint8.
    packed = np.empty(len(xyz), dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
    ], align=False))
    if len(xyz):
        packed["x"], packed["y"], packed["z"] = xyz.T
        occupied = labels == 1
        packed["red"] = np.where(occupied, 255, 46)
        packed["green"] = np.where(occupied, 145, 204)
        packed["blue"] = np.where(occupied, 30, 113)
        packed["alpha"] = np.where(occupied, 235, 120)
    fields = [
        PackedElementField(name="x", offset=0, type=PackedElementField.FLOAT32),
        PackedElementField(name="y", offset=4, type=PackedElementField.FLOAT32),
        PackedElementField(name="z", offset=8, type=PackedElementField.FLOAT32),
        PackedElementField(name="red", offset=12, type=PackedElementField.UINT8),
        PackedElementField(name="green", offset=13, type=PackedElementField.UINT8),
        PackedElementField(name="blue", offset=14, type=PackedElementField.UINT8),
        PackedElementField(name="alpha", offset=15, type=PackedElementField.UINT8),
    ]
    cloud = PointCloud(timestamp=timestamp, frame_id="base_link", point_stride=16,
                       fields=fields, data=packed.tobytes())
    cloud.pose.orientation.w = 1.0
    return cloud


def _scene_update(world, timestamp):
    from foxglove_schemas_protobuf.Color_pb2 import Color
    from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
    from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
    from foxglove_schemas_protobuf.Point3_pb2 import Point3
    from foxglove_schemas_protobuf.Pose_pb2 import Pose
    from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
    from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    from foxglove_schemas_protobuf.Vector3_pb2 import Vector3

    class_colors = {
        "divider": Color(r=1, g=1, b=1, a=.9),
        "centerline": Color(r=.2, g=1, b=.2, a=.9),
        "boundary": Color(r=0, g=1, b=1, a=.9),
        "pedestrian_crossing": Color(r=1, g=.6, b=0, a=.9),
        "stop_line": Color(r=1, g=0, b=0, a=1),
    }
    map_lines = []
    for vector in world["layers"]["vector_map"]:
        points = [Point3(x=float(point[0]), y=float(point[1]), z=-1.9)
                  for point in vector["points"]]
        if len(points) >= 2:
            line = LinePrimitive(type=LinePrimitive.LINE_STRIP, thickness=.12,
                                 points=points,
                                 color=class_colors.get(vector["kind"], class_colors["divider"]))
            line.pose.orientation.w = 1.0
            map_lines.append(line)

    cubes = []
    for agent in world["layers"]["agents"]:
        yaw = float(agent["yaw"])
        pose = Pose(
            position=Vector3(x=float(agent["center"][0]), y=float(agent["center"][1]),
                             z=float(agent["center"][2])),
            orientation=Quaternion(z=float(np.sin(yaw/2)), w=float(np.cos(yaw/2))),
        )
        cubes.append(CubePrimitive(
            pose=pose,
            size=Vector3(x=float(agent["size"][0]), y=float(agent["size"][1]),
                         z=float(agent["size"][2])),
            color=Color(r=.05, g=.65, b=1, a=.45),
        ))

    ego_cubes = [
        CubePrimitive(pose=Pose(position=Vector3(x=0, y=0, z=.52),
                                orientation=Quaternion(w=1)),
                      size=Vector3(x=4.6, y=1.9, z=.68),
                      color=Color(r=.12, g=.33, b=.9, a=.9)),
        CubePrimitive(pose=Pose(position=Vector3(x=.22, y=0, z=1.17),
                                orientation=Quaternion(w=1)),
                      size=Vector3(x=2.35, y=1.64, z=.62),
                      color=Color(r=.25, g=.7, b=1, a=.78)),
        CubePrimitive(pose=Pose(position=Vector3(x=2.22, y=0, z=.58),
                                orientation=Quaternion(w=1)),
                      size=Vector3(x=.08, y=1.55, z=.13),
                      color=Color(r=1, g=.9, b=.3, a=1)),
    ]
    for x in (-1.2, 1.45):
        for y in (-1.02, 1.02):
            ego_cubes.append(CubePrimitive(
                pose=Pose(position=Vector3(x=x, y=y, z=.37),
                          orientation=Quaternion(w=1)),
                size=Vector3(x=.72, y=.18, z=.58),
                color=Color(r=.025, g=.025, b=.025, a=1)))

    entities = [
        SceneEntity(timestamp=timestamp, frame_id="base_link", id="vector_map",
                    frame_locked=True, lines=map_lines),
        SceneEntity(timestamp=timestamp, frame_id="base_link", id="agents",
                    frame_locked=True, cubes=cubes),
        SceneEntity(timestamp=timestamp, frame_id="base_link", id="ego_vehicle",
                    frame_locked=True, cubes=ego_cubes),
    ]
    return SceneUpdate(entities=entities)


def _matrix_quaternion(matrix):
    """Return an x,y,z,w quaternion for a proper 3x3 rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = np.trace(matrix)
    if trace > 0:
        s = np.sqrt(trace+1.0)*2
        return np.array([(matrix[2, 1]-matrix[1, 2])/s,
                         (matrix[0, 2]-matrix[2, 0])/s,
                         (matrix[1, 0]-matrix[0, 1])/s, .25*s])
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        s = np.sqrt(1+matrix[0, 0]-matrix[1, 1]-matrix[2, 2])*2
        return np.array([.25*s, (matrix[0, 1]+matrix[1, 0])/s,
                         (matrix[0, 2]+matrix[2, 0])/s,
                         (matrix[2, 1]-matrix[1, 2])/s])
    if index == 1:
        s = np.sqrt(1+matrix[1, 1]-matrix[0, 0]-matrix[2, 2])*2
        return np.array([(matrix[0, 1]+matrix[1, 0])/s, .25*s,
                         (matrix[1, 2]+matrix[2, 1])/s,
                         (matrix[0, 2]-matrix[2, 0])/s])
    s = np.sqrt(1+matrix[2, 2]-matrix[0, 0]-matrix[1, 1])*2
    return np.array([(matrix[0, 2]+matrix[2, 0])/s,
                     (matrix[1, 2]+matrix[2, 1])/s, .25*s,
                     (matrix[1, 0]-matrix[0, 1])/s])


def _camera_support(camera, timestamp):
    from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
    from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
    from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
    from foxglove_schemas_protobuf.Vector3_pb2 import Vector3

    frame_id = camera["name"].lower()
    intrinsic = np.asarray(camera["intrinsic"], dtype=float)
    projection = np.zeros((3, 4), dtype=float)
    projection[:, :3] = intrinsic
    calibration = CameraCalibration(
        timestamp=timestamp, frame_id=frame_id,
        width=int(camera["width"]), height=int(camera["height"]),
        distortion_model="plumb_bob", D=[0.0]*5, K=intrinsic.ravel().tolist(),
        R=np.eye(3).ravel().tolist(), P=projection.ravel().tolist())
    camera_to_ego = np.asarray(camera["camera_to_ego"], dtype=float)
    quaternion = _matrix_quaternion(camera_to_ego)
    transform = FrameTransform(
        timestamp=timestamp, parent_frame_id="base_link", child_frame_id=frame_id,
        translation=Vector3(x=float(camera_to_ego[0, 3]),
                            y=float(camera_to_ego[1, 3]),
                            z=float(camera_to_ego[2, 3])),
        rotation=Quaternion(x=float(quaternion[0]), y=float(quaternion[1]),
                            z=float(quaternion[2]), w=float(quaternion[3])))
    return calibration, transform


def export_mcap(manifest, world_directory, output, scene_name=None, limit=None):
    """Write synchronized Foxglove-native camera and 3D world topics to MCAP."""
    try:
        from mcap_protobuf.writer import Writer
        from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
        from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
    except ImportError as error:
        raise RuntimeError(
            "MCAP export requires mcap-protobuf-support and foxglove-schemas-protobuf"
        ) from error

    rows = list(read_manifest(manifest))
    if not rows:
        raise ValueError("Manifest is empty")
    selected_scene = scene_name or rows[0]["scene_name"]
    rows = sorted((row for row in rows if row["scene_name"] == selected_scene),
                  key=lambda row: row["timestamp_us"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"Scene not present in manifest: {selected_scene}")

    world_directory = Path(world_directory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = [row["sample_token"] for row in rows
               if not (world_directory/f"{row['sample_token']}.json").is_file()
               or not (world_directory/f"{row['sample_token']}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} compiled frames in {world_directory}; "
            f"first missing token: {missing[0]}")

    topics = {
        "CAM_FRONT_LEFT": "/drivescene/camera/front_left",
        "CAM_FRONT": "/drivescene/camera/front",
        "CAM_FRONT_RIGHT": "/drivescene/camera/front_right",
        "CAM_BACK_LEFT": "/drivescene/camera/back_left",
        "CAM_BACK": "/drivescene/camera/back",
        "CAM_BACK_RIGHT": "/drivescene/camera/back_right",
    }
    calibration_topics = {name: topic+"/calibration" for name, topic in topics.items()}
    with Writer(str(output)) as writer:
        for sequence, row in enumerate(rows):
            timestamp_us = int(row["timestamp_us"])
            log_time = timestamp_us*1_000
            stamp = _timestamp(timestamp_us)
            token = row["sample_token"]
            world = json.loads((world_directory/f"{token}.json").read_text(encoding="utf-8"))
            with np.load(world_directory/f"{token}.npz") as archive:
                semantic = archive["semantic"]
            writer.write_message("/drivescene/occupancy", _point_cloud(
                semantic, world["occupancy"]["grid"], stamp),
                log_time=log_time, publish_time=log_time, sequence=sequence)
            writer.write_message("/drivescene/scene", _scene_update(world, stamp),
                                 log_time=log_time, publish_time=log_time, sequence=sequence)
            transforms = []
            for camera in row["cameras"]:
                image_path = Path(camera["path"])
                calibration, transform = _camera_support(camera, stamp)
                transforms.append(transform)
                message = CompressedImage(timestamp=stamp, frame_id=camera["name"].lower(),
                                          data=image_path.read_bytes(), format="jpeg")
                writer.write_message(topics[camera["name"]], message,
                                     log_time=log_time, publish_time=log_time, sequence=sequence)
                writer.write_message(calibration_topics[camera["name"]], calibration,
                                     log_time=log_time, publish_time=log_time, sequence=sequence)
            writer.write_message("/tf", FrameTransforms(transforms=transforms),
                                 log_time=log_time, publish_time=log_time, sequence=sequence)

    report = {
        "mcap": str(output), "scene_name": selected_scene, "frames": len(rows),
        "duration_seconds": (rows[-1]["timestamp_us"]-rows[0]["timestamp_us"])/1e6,
        "topics": ["/drivescene/occupancy", "/drivescene/scene", "/tf",
                   *topics.values(), *calibration_topics.values()],
        "foxglove_fixed_frame": "base_link",
    }
    output.with_suffix(".mcap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
