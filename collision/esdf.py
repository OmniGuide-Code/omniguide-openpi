"""Euclidean Signed Distance Field (ESDF) for collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F


def world_to_grid_normalized(
    xyz_world: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    grid_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Convert world coordinates to normalized grid coords for F.grid_sample.

    Args:
        xyz_world: (..., 3) positions in meters.
        origin: (3,) world coordinate of voxel index [0, 0, 0].
        voxel_size: Meters per voxel.
        grid_size: (X, Y, Z) grid dimensions.

    Returns:
        (..., 3) coordinates normalized to [-1, 1], in (x, y, z) order
        matching grid_sample's (D, H, W) convention.
    """
    X, Y, Z = grid_size
    ijk = (xyz_world - origin) / voxel_size
    ix, iy, iz = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    gx = 2.0 * (ix / max(X - 1, 1)) - 1.0
    gy = 2.0 * (iy / max(Y - 1, 1)) - 1.0
    gz = 2.0 * (iz / max(Z - 1, 1)) - 1.0
    return torch.stack([gx, gy, gz], dim=-1)


class SampleSDF(torch.autograd.Function):
    """Differentiable trilinear sampler over a voxel ESDF grid."""

    @staticmethod
    def forward(
        ctx,
        sdf: torch.Tensor,
        sdf_grad: torch.Tensor,
        origin: torch.Tensor,
        voxel_size: float,
        max_range: torch.Tensor,
        xyz_world: torch.Tensor,
    ):
        """
        Args:
            sdf: (1, 1, X, Y, Z) ESDF grid.
            sdf_grad: (1, 3, X, Y, Z) gradient of the ESDF.
            origin: (3,) world coordinate of voxel [0, 0, 0].
            voxel_size: Meters per voxel.
            max_range: (3,) maximum world coordinate of the grid.
            xyz_world: (..., 3) query positions.

        Returns:
            (...,) SDF values via trilinear interpolation.
        """
        orig_shape = xyz_world.shape[:-1]
        in_range = ((xyz_world >= origin) & (xyz_world <= max_range)).all(dim=-1)
        coords = world_to_grid_normalized(
            xyz_world.reshape(-1, 3), origin, voxel_size,
            (sdf.shape[2], sdf.shape[3], sdf.shape[4]),
        )
        N = coords.shape[0]
        grid = coords[None, :, None, None, :]  # (1, N, 1, 1, 3)
        sdf_vals = F.grid_sample(
            sdf.permute(0, 1, 4, 3, 2), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (1, 1, N, 1, 1)
        sdf_vals = sdf_vals[0, 0, :, 0, 0]  # (N,)
        ctx.save_for_backward(sdf_grad, coords, in_range)
        return sdf_vals.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        sdf_grad, coords, in_range = ctx.saved_tensors
        sdf_vec = F.grid_sample(
            sdf_grad.permute(0, 1, 4, 3, 2),
            coords[None, :, None, None, :],
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (1, 3, N, 1, 1)
        sdf_vec = sdf_vec[0, :, :, 0, 0].T  # (N, 3)
        sdf_vec = F.normalize(sdf_vec, dim=-1)
        grad_xyz = in_range[:, None] * grad_output[:, None] * sdf_vec
        return None, None, None, None, None, grad_xyz


class SDFCost(torch.nn.Module):
    """Differentiable clearance penalty sampled from a voxel ESDF grid."""

    def __init__(
        self,
        sdf_xyz: torch.Tensor,
        origin_xyz: torch.Tensor,
        range_xyz: torch.Tensor,
        voxel_size: float,
    ):
        super().__init__()
        sdf_grad = compute_gradient_3d(sdf_xyz, voxel_size)
        self.register_buffer("sdf_grad", sdf_grad[None])   # (1, 3, X, Y, Z)
        self.register_buffer("sdf", sdf_xyz[None, None])   # (1, 1, X, Y, Z)
        self.register_buffer("origin", origin_xyz)          # (3,)
        self.register_buffer("range", range_xyz)            # (3,)
        self.voxel_size = float(voxel_size)

    def forward(self, xyz_world: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz_world: (..., 3) query positions in world frame.

        Returns:
            (...,) SDF values (positive = free, negative = inside obstacle).
        """
        return SampleSDF.apply(
            self.sdf, self.sdf_grad, self.origin, self.voxel_size, self.range, xyz_world
        )


def compute_gradient_3d(field: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Compute the gradient of a 3D scalar field using central differences.

    Args:
        field: (X, Y, Z) scalar field.
        voxel_size: Meters per voxel.

    Returns:
        (3, X, Y, Z) gradient (grad_x, grad_y, grad_z).
    """
    grads = torch.gradient(field, spacing=voxel_size)  # [dX, dY, dZ]
    return torch.stack(grads, dim=0)


@dataclass
class ESDFGrid:
    sdf: torch.Tensor        # (X, Y, Z) float32; + outside (free), − inside (occupied)
    origin: torch.Tensor     # (3,) world coords of voxel [0, 0, 0]
    range: torch.Tensor      # (3,) world coords of voxel [X-1, Y-1, Z-1] (bounds_max)
    voxel_size: float        # meters per voxel
    dims: Tuple[int, int, int]


def compute_esdf_from_point_cloud_torch(
    points_xyz: torch.Tensor,
    voxel_size: float = 0.01,
    padding: float = 0.10,
) -> ESDFGrid:
    """GPU-accelerated ESDF via iterative min-pooling (Chebyshev distance approximation).

    Propagates distance outward from occupied voxels using repeated 3×3×3
    min-pool passes (20 iterations → 20 cm maximum repulsion range at 1 cm voxel size).

    Args:
        points_xyz: (N, 3) GPU tensor of point cloud positions.
        voxel_size: Grid resolution in meters.
        padding: Bounding-box padding in meters.

    Returns:
        ESDFGrid with torch tensors for sdf, origin, range.
    """
    assert isinstance(points_xyz, torch.Tensor)
    device = points_xyz.device

    bounds_min = points_xyz.min(dim=0)[0] - padding
    bounds_max = points_xyz.max(dim=0)[0] + padding
    dims = torch.ceil((bounds_max - bounds_min) / voxel_size).int().tolist()
    dims = [max(1, d) for d in dims]
    X, Y, Z = dims

    idx = torch.round((points_xyz - bounds_min) / voxel_size).long()
    idx[:, 0] = idx[:, 0].clamp(0, X - 1)
    idx[:, 1] = idx[:, 1].clamp(0, Y - 1)
    idx[:, 2] = idx[:, 2].clamp(0, Z - 1)

    # Initialize: occupied voxels = 0, free = infinity
    grid = torch.full((1, 1, X, Y, Z), 10.0, device=device, dtype=torch.float32)
    grid[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 0.0

    # Propagate distances outward: 20 iterations covers ~20 cm at 1 cm resolution
    for _ in range(20):
        neighbor_dist = -F.max_pool3d(-grid, kernel_size=3, stride=1, padding=1) + voxel_size
        grid = torch.min(grid, neighbor_dist)

    return ESDFGrid(
        sdf=grid[0, 0],
        origin=bounds_min,
        range=bounds_max,
        voxel_size=voxel_size,
        dims=(X, Y, Z),
    )
