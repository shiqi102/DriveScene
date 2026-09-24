from pathlib import Path
import numpy as np


def _draw_box_3d(ax, agent, color="deepskyblue"):
    from .geometry import box_corners_xy

    xy = box_corners_xy(agent["center"], agent["size"], agent["yaw"])
    center_z = float(agent["center"][2])
    half_height = float(agent["size"][2]) / 2
    lower = np.column_stack([xy, np.full(4, center_z-half_height)])
    upper = np.column_stack([xy, np.full(4, center_z+half_height)])
    corners = np.vstack([lower, upper])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for start, end in edges:
        segment = corners[[start, end]]
        # Display axes are lateral, forward, height for an intuitive camera view.
        ax.plot(segment[:, 1], segment[:, 0], segment[:, 2], color=color, linewidth=1.2)


def _draw_ego_2d(ax):
    from matplotlib.patches import Polygon
    from .geometry import box_corners_xy

    corners = box_corners_xy([0, 0, .75], [4.6, 1.9, 1.5], 0)
    ax.add_patch(Polygon(corners[:, ::-1], closed=True, facecolor="royalblue",
                         edgecolor="white", linewidth=1.3, alpha=.9, zorder=8))
    # A short white stripe makes the vehicle's forward direction obvious.
    ax.plot([-.45, .45], [1.55, 2.05], color="white", linewidth=2.2, zorder=9)


def _draw_ego_3d(ax):
    # Matplotlib display axes are (ego y, ego x, ego z).
    ax.bar3d(-.95, -2.3, .18, 1.9, 4.6, .68,
             color="royalblue", edgecolor="white", linewidth=.35, alpha=.88, shade=True)
    ax.bar3d(-.82, -.95, .86, 1.64, 2.35, .62,
             color="lightskyblue", edgecolor="white", linewidth=.3, alpha=.8, shade=True)
    for lateral in (-1.03, .87):
        for forward in (-1.55, 1.0):
            ax.bar3d(lateral, forward, .08, .16, .72, .58,
                     color="black", edgecolor="dimgray", linewidth=.2, shade=True)


def _render_bev_2d(ax, world, arrays):
    from .geometry import box_corners_xy

    config = world["occupancy"]["grid"]
    ax.imshow(arrays["cost"], extent=[*config["y"], *config["x"]], origin="lower",
              cmap="magma", vmin=0, vmax=1, alpha=.65)
    colors = {"divider": "white", "centerline": "lime", "boundary": "cyan",
              "pedestrian_crossing": "orange", "stop_line": "red"}
    for vector in world["layers"]["vector_map"]:
        xy = np.asarray(vector["points"])
        ax.plot(xy[:, 1], xy[:, 0], color=colors.get(vector["kind"], "white"), linewidth=1)
    for agent in world["layers"]["agents"]:
        corners = box_corners_xy(agent["center"], agent["size"], agent["yaw"])
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 1], corners[:, 0], color="deepskyblue")
    _draw_ego_2d(ax)
    ax.set_xlim(config["y"])
    ax.set_ylim(config["x"])
    ax.invert_xaxis()
    ax.set_xlabel("ego y (left), m")
    ax.set_ylabel("ego x (forward), m")
    ax.set_aspect("equal")
    ax.set_title(f"2D BEV / {world['provenance']['mode']}\nUnknown cells are NOT free space")


def _render_bev_3d(ax, world, arrays, max_voxels=8000):
    from .geometry import grid_centers

    config = world["occupancy"]["grid"]
    semantic = np.asarray(arrays["semantic"])
    centers = grid_centers(config)
    colors = {"divider": "white", "centerline": "lime", "boundary": "cyan",
              "pedestrian_crossing": "orange", "stop_line": "red"}

    def scatter_class(class_id, color, alpha, size, budget):
        points = centers[semantic == class_id]
        if len(points) > budget:
            # Even deterministic sampling keeps video frames stable and bounded.
            indices = np.linspace(0, len(points)-1, budget, dtype=np.int64)
            points = points[indices]
        if len(points):
            ax.scatter(points[:, 1], points[:, 0], points[:, 2], s=size,
                       c=color, alpha=alpha, depthshade=False, edgecolors="none")

    scatter_class(2, "mediumseagreen", .22, 5, max_voxels//2)
    scatter_class(1, "darkorange", .72, 10, max_voxels)
    map_z = config["z"][0] + .08
    for vector in world["layers"]["vector_map"]:
        xy = np.asarray(vector["points"])
        ax.plot(xy[:, 1], xy[:, 0], np.full(len(xy), map_z),
                color=colors.get(vector["kind"], "white"), linewidth=1.2)
    for agent in world["layers"]["agents"]:
        _draw_box_3d(ax, agent)
    _draw_ego_3d(ax)
    ax.set_xlim(config["y"])
    ax.set_ylim(config["x"])
    ax.set_zlim(config["z"])
    ax.set_xlabel("ego y (left), m", labelpad=7)
    ax.set_ylabel("ego x (forward), m", labelpad=7)
    ax.set_zlabel("z (up), m", labelpad=5)
    ax.view_init(elev=27, azim=-58)
    try:
        ax.set_box_aspect((1, 1, .34))
    except AttributeError:
        pass
    ax.set_title(f"3D occupancy / {world['provenance']['mode']}\n"
                 "orange: occupied, green: drivable, blue: agents")


def render_world(world, arrays, path, row=None, view="2d"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if view not in {"2d", "3d"}:
        raise ValueError(f"Unsupported visualization view: {view}")
    # BEV stays on the left. The six cameras occupy a compact 2x3 wall on
    # the right: front views on the first row, rear views on the second row.
    figure = plt.figure(figsize=(20, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=[1.55, 1, 1, 1])
    ax = figure.add_subplot(grid[:, 0], projection="3d" if view == "3d" else None)
    if view == "3d":
        _render_bev_3d(ax, world, arrays)
    else:
        _render_bev_2d(ax, world, arrays)
    camera_order = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    cameras = {camera["name"]: camera for camera in row["cameras"]} if row else {}
    for index, name in enumerate(camera_order):
        camera_ax = figure.add_subplot(grid[index//3, index%3+1])
        if row:
            camera = cameras[name]
            with Image.open(camera["path"]) as image:
                camera_ax.imshow(image)
            camera_ax.set_title(camera["name"], fontsize=9)
        camera_ax.axis("off")
    figure.suptitle(f"DriveScene — {world['sample_token']} | research output, not planning-validated")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
