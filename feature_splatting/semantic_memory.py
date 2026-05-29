import torch
from typing import Tuple
import torch.nn.functional as F


## map XYZ world coordinates to grid index in the voxel grid
## but also normalized to [-1, 1] for F.grid_sample which does
## trilinear interpolation later
def world_to_grid_normalized(
    xyz_world: torch.Tensor, origin: torch.Tensor, voxel_size: float, grid_size: Tuple[int, int, int]
) -> torch.Tensor:
    """
    xyz_world: (..., 3) in meters (x, y, z)
    grid_size: (X, Y, Z) - dimensions of the grid
    returns grid coords normalized to [-1,1] for F.grid_sample in (x,y,z) order

    Our tensor is stored as (X, Y, Z) which maps to grid_sample's (D, H, W) format:
    X→D (depth), Y→H (height), Z→W (width)
    Grid coordinates are returned as (gx, gy, gz) corresponding to (D, H, W) indices
    """
    X, Y, Z = grid_size
    # Convert to voxel indices (x,y,z)
    ijk = (xyz_world - origin) / voxel_size  # (...,3) in (x,y,z) index space
    ix, iy, iz = ijk[..., 0], ijk[..., 1], ijk[..., 2]

    # Normalize each dimension to [-1, 1]
    gx = 2.0 * (ix / max(X - 1, 1)) - 1.0
    gy = 2.0 * (iy / max(Y - 1, 1)) - 1.0
    gz = 2.0 * (iz / max(Z - 1, 1)) - 1.0

    return torch.stack([gx, gy, gz], dim=-1)


class SemanticMemory:
    def __init__(self, grid_size: Tuple[int, int, int], voxel_size: float, origin: torch.Tensor):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.origin = origin

    def set_feature_grid(self, feat_grid: torch.Tensor):
        """
        feat_grid: (1, D, X, Y, Z) — CLIP feature volume stored per voxel
        """
        assert feat_grid.ndim == 5 and feat_grid.shape[0] == 1, "feat_grid must be (1, D, X, Y, Z)"
        self.feat_grid = feat_grid

    @property
    def max_range(self) -> torch.Tensor:
        # world coord of the *far corner* corresponding to index (X-1, Y-1, Z-1)
        X, Y, Z = self.grid_size
        return (
            self.origin
            + torch.tensor([X - 1, Y - 1, Z - 1], dtype=self.origin.dtype, device=self.origin.device) * self.voxel_size
        )

    @torch.no_grad()
    def query_features(self, xyz_world: torch.Tensor) -> torch.Tensor:
        """
        xyz_world: (N,3) [or (...,3)] world points.
        returns: (N,D) CLIP features trilinearly interpolated from the voxel grid.
        """
        assert self.feat_grid is not None, "Call set_feature_grid(...) first"
        orig_shape = xyz_world.shape[:-1]
        coords = world_to_grid_normalized(xyz_world.reshape(-1, 3), self.origin, self.voxel_size, self.grid_size)
        # grid_sample expects (N,C,D,H,W) input; our volume is (1,D,X,Y,Z).
        # It also expects coords as (N, outD, outH, outW, 3).
        grid = coords[None, :, None, None, :]  # (1, N, 1, 1, 3)

        # Match the ESDF convention: tensor stored as (X,Y,Z) → map to (D,H,W) via permute(0,1,4,3,2)
        # So depth=Z, height=Y, width=X after permute.
        vol = self.feat_grid.permute(0, 1, 4, 3, 2)  # (1, D, Z, Y, X)

        sampled = F.grid_sample(
            vol, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, D, N, 1, 1)
        feats = sampled[0, :, :, 0, 0].T  # (N, D)
        return feats.reshape(*orig_shape, feats.shape[-1])
