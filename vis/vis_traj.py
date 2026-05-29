"""3D visualization helpers backed by a Viser server.

The Viser server is created lazily on the first call so that importing this
module (e.g. from the policy) does not immediately open a browser tab or
bind a port.
"""

import re
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import viser

_server: viser.ViserServer | None = None


def _get_server() -> viser.ViserServer:
    global _server
    if _server is None:
        _server = viser.ViserServer()
    return _server


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def remove_all(name: str = "all", server_local: viser.ViserServer | None = None) -> None:
    """Remove objects from the scene.  Pass a regex pattern to remove selectively."""
    srv = server_local or _get_server()
    for obj_name in list(srv.scene._handle_from_node_name.keys()):
        if obj_name == "/WorldAxes":
            continue
        if name == "all" or (re.match(name, obj_name) and re.match(name, obj_name).span()[1] > 0):
            srv.scene.remove_by_name(obj_name)


def add_lines(
    st_points: np.ndarray | torch.Tensor,
    ed_points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor,
    line_width: float = 0.002,
    name: str = "lines",
    server_local: viser.ViserServer | None = None,
) -> None:
    """Add line segments to the scene.

    Args:
        st_points: (N, 3) start points.
        ed_points: (N, 3) end points.
        colors: (N, 3) or (3,) RGB in [0, 1].
    """
    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    st_points, ed_points, colors = _np(st_points), _np(ed_points), _np(colors)
    points = np.stack([st_points, ed_points], axis=1)  # (N, 2, 3)
    (server_local or _get_server()).scene.add_line_segments(
        name=name, points=points.astype("float32"), colors=colors, line_width=line_width
    )


def add_point(
    point: np.ndarray | torch.Tensor,
    color: np.ndarray | torch.Tensor,
    point_size: float = 0.1,
    name: str = "point",
    server_local: viser.ViserServer | None = None,
) -> None:
    """Add a single point (rendered as a sphere) to the scene."""
    if isinstance(point, torch.Tensor):
        point = point.detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()
    (server_local or _get_server()).scene.add_icosphere(
        name=name, position=point, color=color, radius=point_size
    )


def add_point_cloud(
    points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor,
    point_size: float = 0.005,
    name: str = "pcd",
    visible: bool = True,
    server_local: viser.ViserServer | None = None,
) -> None:
    """Add a point cloud to the scene.

    Args:
        points: (N, 3) point positions.
        colors: (N, 3) per-point RGB in [0, 1], or (3,) for a uniform color.
        visible: Whether the point cloud is visible by default in the viewer.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    if colors.ndim == 1 and colors.shape[0] == 3:
        colors = np.tile(colors[None, :], (len(points), 1))
    (server_local or _get_server()).scene.add_point_cloud(
        name=name,
        points=points.astype("float32"),
        colors=colors,
        point_size=point_size,
        point_shape="circle",
        visible=visible,
    )


def add_frames(
    frame_traj: np.ndarray | torch.Tensor,
    color: tuple = (128, 128, 0),
    name: str = "traj",
    coord_visible: bool = False,
    radius: float = 0.01,
    visible: bool = True,
    server_local: viser.ViserServer | None = None,
) -> None:
    """Visualize a sequence of SE(3) frames as spheres (and optional coordinate frames).

    Args:
        frame_traj: (N, 4, 4) homogeneous transforms.
        color: Base RGB color (0-255). Fades toward white for early frames.
    """
    if isinstance(frame_traj, torch.Tensor):
        frame_traj = frame_traj.cpu().numpy()
    srv = server_local or _get_server()
    for i, frame in enumerate(frame_traj):
        blend = (i / len(frame_traj)) * 0.7
        c = tuple(np.clip(np.array(color) + (255.0 - np.array(color)) * blend, 0, 255).astype(int))
        quat = R.from_matrix(frame[:3, :3]).as_quat(scalar_first=True)
        pos = frame[:3, 3]
        srv.scene.add_frame(
            name=f"{name}_coord/{i:03d}",
            wxyz=quat, position=pos,
            axes_radius=radius * 0.3, axes_length=radius * 8,
            origin_color=c, origin_radius=radius * 1.1,
            visible=coord_visible and visible,
        )
        srv.scene.add_icosphere(
            name=f"{name}/{i:03d}", radius=radius, color=c, position=pos, visible=visible
        )


def add_sphere(
    center: np.ndarray | torch.Tensor,
    radius: float,
    color: tuple = (230, 204, 26),
    name: str = "sphere",
    visible: bool = True,
    server_local: viser.ViserServer | None = None,
) -> None:
    """Add a sphere to the scene using viser's icosphere primitive."""
    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
    (server_local or _get_server()).scene.add_icosphere(
        name=name, radius=radius, color=color, position=np.asarray(center, dtype=float), visible=visible
    )


def add_cube(
    points: np.ndarray | torch.Tensor,
    color: tuple = (128, 128, 0),
    name: str = "cube",
    server_local: viser.ViserServer | None = None,
) -> None:
    """Visualize an axis-aligned bounding box from its 8 corner vertices.

    Args:
        points: (8, 3) cube corner vertices.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    pts = points.reshape(-1, 3)
    (server_local or _get_server()).scene.add_box(
        name=name,
        dimensions=(pts.max(axis=0) - pts.min(axis=0)),
        position=((pts.max(axis=0) + pts.min(axis=0)) / 2.0),
        color=color,
    )
