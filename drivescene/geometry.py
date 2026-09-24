from __future__ import annotations
import numpy as np


def transform_matrix(translation, quaternion):
    """nuScenes quaternion convention: w,x,y,z; transforms local -> parent."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid rotation quaternion")
    w, x, y, z = q / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


def transform_points(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    return p @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def project(points, ego_to_image):
    uvz = transform_points(points, ego_to_image)
    valid = uvz[:, 2] > 1e-5
    uv = uvz[:, :2] / np.maximum(uvz[:, 2:3], 1e-5)
    return uv, valid


def resample_line(points, count):
    from shapely.geometry import LineString
    line = LineString(points)
    if line.length < 1e-6:
        raise ValueError("Degenerate map line")
    return np.asarray([line.interpolate(d).coords[0][:2]
                       for d in np.linspace(0, line.length, count)])


def box_corners_xy(center, size, yaw):
    length, width = size[:2]
    xy = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length/2, width/2]
    c, s = np.cos(yaw), np.sin(yaw)
    return xy @ np.array([[c, s], [-s, c]]) + np.asarray(center[:2])


def grid_centers(grid):
    nx, ny, nz = grid["shape"]
    axes = [(np.arange(n)+0.5)*(grid[k][1]-grid[k][0])/n+grid[k][0]
            for n, k in zip((nx, ny, nz), ("x", "y", "z"))]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
