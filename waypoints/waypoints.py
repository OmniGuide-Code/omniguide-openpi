"""3D waypoint guidance from pre-recorded trajectories.

Waypoints are stored in a camera frame (or world frame directly) and
transformed to world frame at inference time using the observation's extrinsics.
"""

import numpy as np
import torch


class Waypoints3D:
    """Waypoints loaded from a pre-recorded 3D trajectory (.npy file).

    The trajectory is recorded in a specific camera frame (or world frame) and
    transformed to world frame at runtime using the current camera extrinsics.

    Args:
        path: Path to .npy file containing waypoints of shape (N, 3).
        threshold_distance: Distance (m) at which the EE is considered to have
            reached the current waypoint.
        num_guided: Number of action-sequence steps to apply guidance per denoising call.
        camera_view: Camera frame the trajectory was recorded in.
            One of "left", "right", "overhead", or "world" (no transform applied).
    """

    def __init__(
        self,
        path: str,
        threshold_distance: float = 0.01,
        num_guided: int = 2,
        camera_view: str = "right",
    ):
        self.traj_cam = torch.from_numpy(np.load(path)).to("cuda")
        self.camera_view = camera_view
        self.threshold_distance = threshold_distance
        self.num_guided = num_guided
        self._waypoints_world: torch.Tensor | None = None

    def prepare(self, observation) -> None:
        """Transform the trajectory to world frame for the current observation.

        Must be called once per guidance step before using any other methods.
        """
        if self.camera_view == "world":
            self._waypoints_world = self.traj_cam
            return

        extrin_map = {
            "left": observation.left_extrin,
            "right": observation.right_extrin,
            "overhead": observation.overhead_extrin,
        }
        if self.camera_view not in extrin_map:
            raise ValueError(
                f"Invalid camera_view: {self.camera_view!r}. "
                f"Choose from {list(extrin_map)} or 'world'."
            )
        extrin = extrin_map[self.camera_view]
        R, t = extrin[0, :3, :3], extrin[0, :3, 3]
        self._waypoints_world = (R @ self.traj_cam.T).T + t.unsqueeze(0)

    def all_waypoints(self) -> torch.Tensor:
        """Return all waypoints in world frame (N, 3)."""
        assert self._waypoints_world is not None, "Call prepare() first."
        return self._waypoints_world.clone()

    def start_waypoint(self) -> torch.Tensor:
        """Return the first waypoint (trajectory start) in world frame."""
        assert self._waypoints_world is not None, "Call prepare() first."
        return self._waypoints_world[0]

    def sequential_match(self, pos: torch.Tensor) -> torch.Tensor:
        """Match each position in pos to the next closest waypoint (no repeats).

        Precomputes all pairwise distances once with torch.cdist, then does
        sequential argmin lookups in a Python loop (O(T) iterations, no inner loop).

        Args:
            pos: (T, 3) end-effector positions from the action sequence.

        Returns:
            (T', 3) matched waypoints, T' <= T.
        """
        assert self._waypoints_world is not None, "Call prepare() first."
        W = self._waypoints_world.shape[0]
        # (T, W) distance matrix computed once on GPU
        dists = torch.cdist(pos, self._waypoints_world)
        first, matched = 0, []
        for i in range(len(pos)):
            best_idx = first + int(dists[i, first:].argmin().item())
            matched.append(self._waypoints_world[best_idx])
            first = best_idx + 1
            if first >= W:
                break
        return torch.stack(matched)

    @classmethod
    def from_guidance_config(cls, guidance_config) -> "Waypoints3D":
        """Construct directly from a GuidanceConfig."""
        return cls(
            path=guidance_config.waypoint.path,
            threshold_distance=guidance_config.waypoint.threshold_distance,
            num_guided=guidance_config.guidance.num_guided,
            camera_view=guidance_config.waypoint.camera_view,
        )
