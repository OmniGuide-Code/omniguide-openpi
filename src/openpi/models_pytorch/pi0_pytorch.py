import logging
import math
import concurrent.futures
import time as pytime

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from vis.vis_traj import *
from vis.franka import FK_batch
import os
from typing import Any

import matplotlib.pyplot as plt
import imageio.v2 as iio
import openpi.transforms as _transforms

from collision.vggt_pcd import PCD_Generator, preprocess_images
from collision.esdf import compute_esdf_from_point_cloud_torch, SDFCost
from collision.sam2_segmentation import SAM2Segmenter
from feature_splatting import CLIP3DLifter
from waypoints.waypoints import Waypoints3D
from progress_tracker.progress_tracker import ProgressTracker
from vlm_pointer.vlm_pointer import VLMPointer


# Visualization color convention (RGB float, matches docs/viz_guide.md)
_VIZ_REPULSIVE = torch.tensor([[0.85, 0.15, 0.15]])   # red   — SDF collision field
_VIZ_ATTRACTIVE = torch.tensor([[0.08, 0.20, 0.70]])  # dark blue — task targets
_VIZ_NEUTRAL = torch.tensor([[0.90, 0.80, 0.10]])     # yellow — robot body / exclusion zones
_VIZ_POINT_SIZE = 0.004
_CAMERA_VIEWS = ("left", "right", "overhead")


def _validate_camera_views(camera_views) -> tuple[str, ...]:
    assert isinstance(camera_views, (list, tuple)), f"camera_views must be a list or tuple, got {type(camera_views)}"
    assert len(camera_views) > 0, "point_cloud.camera_views must contain at least one camera"
    invalid = [view for view in camera_views if view not in _CAMERA_VIEWS]
    assert not invalid, f"invalid point_cloud.camera_views {invalid}; valid views are {_CAMERA_VIEWS}"
    assert len(set(camera_views)) == len(camera_views), f"duplicate camera views are not allowed: {camera_views}"
    return tuple(camera_views)


def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    """Ensure x is a torch tensor on device. Pass dtype=None to preserve original dtype."""
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return x.to(device=device) if dtype is None else x.to(device=device, dtype=dtype)


def _chunked_cdist_min(pts_a: torch.Tensor, pts_b: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """Per-row minimum distance from pts_a to pts_b, computed in chunks to avoid OOM.

    Args:
        pts_a: (N, 3) query points.
        pts_b: (M, 3) reference points.
        chunk_size: Rows of pts_a processed per cdist call.

    Returns:
        (N,) minimum distance from each point in pts_a to any point in pts_b.
    """
    min_dists = torch.empty(pts_a.shape[0], device=pts_a.device)
    for start in range(0, pts_a.shape[0], chunk_size):
        chunk = pts_a[start : start + chunk_size]
        min_dists[start : start + chunk_size] = torch.cdist(chunk, pts_b).min(dim=1)[0]
    return min_dists


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.`")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(
        self, config, guidance_config: dict[str, Any] | None = None, save_state_dir=None
    ):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.guidance_config = guidance_config
        self.save_state_dir = save_state_dir
        self.viz_viser = guidance_config.viz_viser if guidance_config else True
        self.save_debug_images = guidance_config.save_debug_images if guidance_config else False
        self.debug_pause_before_action_return = (
            guidance_config.debug_pause_before_action_return if guidance_config else False
        )
        if self.save_debug_images:
            os.makedirs("debug/images", exist_ok=True)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("highest")
        self.fk = FK_batch()
        self.tip_coord = torch.tensor([0, 0, 0.08, 1], device="cuda")  # end-effector offset in end-effector frame
        # Probe points around the wrist in local end-effector frame (homogeneous).
        # Used for both SDF collision sampling and visualization.
        self._wrist_probe = torch.tensor(
            [[0, 0, -0.05, 1.0], [0.04, 0, 0.0, 1.0], [0, 0.04, 0.0, 1.0], [0, -0.04, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.global_count = 0

        self.guidance_types = set(self.guidance_config.guidance_types)
        self.pcd_camera_views = _validate_camera_views(self.guidance_config.point_cloud.camera_views)

        self.needs_pcd = any(gtype in self.guidance_types for gtype in ["collision", "waypoint"])
        # CLIP is only needed for collision filtering when vlm_pointer is not active.
        # When vlm_pointer is present, SAM2 segments the target using the VLM-provided
        # point instead, so CLIP is not required.
        self.needs_clip = (
            "collision" in self.guidance_types and "vlm_pointer" not in self.guidance_types
        )

        if self.needs_pcd:
            self.pcd_gen = PCD_Generator()
        if self.needs_clip:
            self.clip_lifter = CLIP3DLifter(device="cuda", feature_type="object")

        if "waypoint" in self.guidance_types:
            self.waypoints = Waypoints3D.from_guidance_config(self.guidance_config)

        self.gradient_checkpointing_enabled = False
        self.progress_tracker = ProgressTracker(
            self.guidance_config.vlm.progress_tracker_url,
            max_horizon=1,
            grasping_threshold=self.guidance_config.freeze_zone.gripper_threshold,
        )
        self.vlm_pointer = VLMPointer(
            self.guidance_config.vlm.pointer_url,
            camera_view=self.guidance_config.vlm.camera_view,
        )
        if "vlm_pointer" in self.guidance_types:
            assert self.guidance_config.vlm.camera_view in self.pcd_camera_views, (
                f"vlm.camera_view='{self.guidance_config.vlm.camera_view}' must be included in "
                f"point_cloud.camera_views={self.pcd_camera_views}"
            )

        # Freeze zone state: suppress guidance when the end-effector is near the target
        self.semantic_freeze_zone = False
        self.cached_task_relevant_pcd = None
        self.just_exited_freezing_zone = False

        # SAM2 segmenter for collision object filtering (lazy initialization)
        self.sam2_segmenter = None
        self.cached_sam2_relevant_points = None
        self.cached_sam2_subtask = None  # subtask index the cache was built for

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    # -------------------------------------------------------------------------
    # Gradient checkpointing
    # -------------------------------------------------------------------------

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Scene / guidance setup helpers
    # -------------------------------------------------------------------------

    def _compute_robot_state(self, observation, output_transform):
        """Compute end-effector and joint transforms from current robot state."""
        bsize = observation.state.shape[0]
        raw_action_dict = dict(
            actions=torch.zeros((bsize, 1, 32)).to(observation.state.device), state=observation.state[0]
        )
        out0 = output_transform.transforms[0](raw_action_dict)
        joint_positions, all_trans = self.fk.forward(out0["state"][:7].float(), all_trans=True)
        ee = all_trans[-1]
        return ee, all_trans, joint_positions

    def _stack_camera_tensors(self, observation, suffix: str) -> torch.Tensor:
        return torch.cat([getattr(observation, f"{view}_{suffix}") for view in self.pcd_camera_views])

    def _stack_camera_depths(self, observation) -> torch.Tensor | None:
        depths = [getattr(observation, f"{view}_depth") for view in self.pcd_camera_views]
        has_depth = [depth is not None for depth in depths]
        assert all(has_depth) or not any(has_depth), (
            f"selected cameras must either all provide depth or all omit it; "
            f"point_cloud.camera_views={self.pcd_camera_views}, has_depth={has_depth}"
        )
        return torch.cat(depths) if all(has_depth) else None

    def _compute_point_cloud(self, observation, device):
        """Compute point cloud from RGB-D observations."""
        z_thresh = -0.1
        pcd_ret = self.pcd_gen.generate_pcd(
            self._stack_camera_tensors(observation, "image"),
            intrinsics=self._stack_camera_tensors(observation, "intrin"),
            extrinsics=self._stack_camera_tensors(observation, "extrin"),
            depth=self._stack_camera_depths(observation),
            joint_angles=torch.cat([observation.joint_position[0], observation.gripper_position[0]], dim=0).float(),
            conf_thresh=2.0,
            z_thresh=z_thresh,
            save_debug_images=self.save_debug_images,
        )
        return pcd_ret

    def _filter_point_cloud_for_collision(self, pcd_ret, ee, gripper_state, show_viz=True):
        """Filter point cloud for collision avoidance using proximity-based exclusion.

        Uses a dynamic exclusion radius around the end-effector:
        - When the gripper is open: excludes points within `ee_clearance_open`.
        - When the gripper is closed (holding object): expands to `ee_clearance_closed`,
          which absorbs the held object into the robot body exclusion zone and prevents
          it from emitting spurious collision repulsion.

        This replaces the previous CLIP-based in-hand object filtering, making the
        collision pipeline independent of the task description and consistent across
        all guidance modes (collision-only, vlm_pointer+collision, waypoint+collision).
        """
        col_params = self.guidance_config.collision_params
        tip_position = ee @ self.tip_coord
        tip_pos = tip_position[0, :3]  # (3,) GPU tensor

        points = _to_tensor(pcd_ret["points"], tip_pos.device)
        dist_to_hand = torch.norm(points - tip_pos, dim=1)

        gripper_closed = gripper_state < col_params.gripper_closed_threshold
        exclusion_radius = col_params.ee_clearance_closed if gripper_closed else col_params.ee_clearance_open
        if gripper_closed:
            print(
                f"[COLLISION] Gripper closed (state={gripper_state:.3f}), "
                f"using expanded exclusion radius={exclusion_radius:.2f}m for in-hand object"
            )

        keep_mask = dist_to_hand > exclusion_radius

        colors = _to_tensor(pcd_ret["colors"], tip_pos.device)

        if self.viz_viser and show_viz:
            excluded_pts = points[~keep_mask]
            label = "closed" if gripper_closed else "open"
            add_point_cloud(
                excluded_pts,
                colors=_VIZ_NEUTRAL.expand(excluded_pts.shape[0], 3),
                name=f"neutral/ee_clearance_{label}",
                point_size=_VIZ_POINT_SIZE,
            )
            add_sphere(
                tip_pos.cpu().numpy(), radius=exclusion_radius,
                color=(230, 204, 26),
                name="neutral/ee_clearance_sphere",
                visible=False,
            )

        return {
            "points": points[keep_mask],
            "colors": colors[keep_mask],
            "point_map": pcd_ret["point_map"],
            "color_map": pcd_ret["color_map"],
            "mask": pcd_ret["mask"],
        }

    def _apply_cached_sam2_filter(self, pcd_ret, device):
        """Filter collision point cloud using the cached SAM2 segmentation result.

        Removes any point in pcd_ret that lies within `target_proximity_threshold` of a
        previously segmented target-object point.  Must be called after the cache has
        been populated (e.g. by `_apply_vlm_sam2_filter` on the first frame).
        """
        if self.cached_sam2_relevant_points is None:
            return pcd_ret

        threshold = self.guidance_config.collision_params.target_proximity_threshold
        cached_pts = _to_tensor(self.cached_sam2_relevant_points, device)
        current_pts = _to_tensor(pcd_ret["points"], device)
        colors = _to_tensor(pcd_ret["colors"], device)
        keep_mask = _chunked_cdist_min(current_pts, cached_pts) > threshold

        print(f"[SAM2] Keeping {keep_mask.sum()} background points ({(~keep_mask).sum()} proximity-neutralised)")

        result = dict(pcd_ret)
        result["points"] = current_pts[keep_mask]
        result["colors"] = colors[keep_mask]

        if self.viz_viser:
            # SAM2-excluded points: removed from collision field; may also be the vlm_pointer target
            vlm_active = self._is_guidance_active("vlm_pointer")
            viz_color = _VIZ_ATTRACTIVE if vlm_active else _VIZ_NEUTRAL
            add_point_cloud(
                cached_pts,
                colors=viz_color.expand(cached_pts.shape[0], 3),
                name="sam2/excluded",
                point_size=_VIZ_POINT_SIZE * 10 if vlm_active else _VIZ_POINT_SIZE,
            )
            # Points near the SAM2 target but not part of the segmentation -> neutral
            proximity_neutral_pts = current_pts[~keep_mask]
            add_point_cloud(
                proximity_neutral_pts,
                colors=_VIZ_NEUTRAL.expand(proximity_neutral_pts.shape[0], 3),
                name="neutral/target_border",
                point_size=_VIZ_POINT_SIZE,
            )
        return result

    def _apply_vlm_sam2_filter(self, pcd_ret, observation, ee, gripper_state, device, prompt, subtask):
        """Filter the collision point cloud using the VLM pointer + SAM2 pipeline.

        Queries the VLM pointer once to get the 2D pixel coordinate of the task-relevant
        target, then uses that as the SAM2 source point to segment the object.  Segmented
        points are excluded from the collision field so the robot is not repelled from
        its intended target.  The dynamic proximity filter is applied afterward to handle
        the in-hand object.

        This takes priority over the CLIP-based filter when vlm_pointer is in guidance_types.
        """
        effective_subtask = subtask if (self.guidance_config.subtask_monitor.enabled and subtask) else prompt

        current_subtask_idx = (
            self.progress_tracker.current_subtask
            if self.guidance_config.subtask_monitor.enabled
            else 0
        )
        cache_stale = self.cached_sam2_subtask != current_subtask_idx

        # Query the VLM whenever the cache is empty or the subtask has changed.
        if cache_stale:
            print("[VLM] Querying VLM pointer for target pixel...")
            waypoint_2d = self.vlm_pointer.get_3d_pointer(
                observation=observation, task=prompt, subtask=effective_subtask
            )
            row, col = int(waypoint_2d[0]), int(waypoint_2d[1])
            cam_idx = self.pcd_camera_views.index(self.vlm_pointer.camera_view)

            if self.guidance_config.vlm.use_sam2:
                # SAM2 path: segment the full target object from the VLM pixel.
                if self.sam2_segmenter is None:
                    print("[SAM2+VLM] Initializing SAM2 segmenter...")
                    self.sam2_segmenter = SAM2Segmenter("facebook/sam2-hiera-large")
                uv = (col, row)  # SAM2 expects (col, row)
                print(f"[SAM2+VLM] Source point: uv={uv}, camera={cam_idx}")
                seg_result = self.sam2_segmenter.segment_and_filter_pcd(
                    uv=uv,
                    cam_idx=cam_idx,
                    pcd_ret=pcd_ret,
                    distance_threshold=self.guidance_config.collision_params.target_proximity_threshold,
                    save_debug_images=self.save_debug_images,
                )
                rel_pts = seg_result["relevant_points"]
                self.cached_sam2_relevant_points = rel_pts if len(rel_pts) > 0 else None
                n = len(self.cached_sam2_relevant_points) if self.cached_sam2_relevant_points is not None else 0
                print(f"[SAM2+VLM] Cached {n} target points for subtask {current_subtask_idx}")
            else:
                # Single-point path: lift the VLM pixel directly to 3D via the point map.
                point_3d = pcd_ret["point_map"][cam_idx][row, col]  # (3,)
                print(f"[VLM] Single-point target: pixel=({row},{col}), 3D={point_3d.tolist()}, camera={cam_idx}")
                if torch.isfinite(point_3d).all():
                    self.cached_sam2_relevant_points = point_3d.unsqueeze(0)  # (1, 3)
                else:
                    print("[VLM] WARNING: lifted point is not finite, skipping cache")
                    self.cached_sam2_relevant_points = None
                n = 1 if self.cached_sam2_relevant_points is not None else 0
                print(f"[VLM] Cached {n} target point for subtask {current_subtask_idx}")

            self.cached_sam2_subtask = current_subtask_idx
            if self.viz_viser and self.cached_sam2_relevant_points is not None:
                rel_pts = self.cached_sam2_relevant_points
                add_point_cloud(
                    _to_tensor(rel_pts, device),
                    colors=_VIZ_ATTRACTIVE.expand(len(rel_pts), 3),
                    name="attractive/vlm/target",
                    point_size=_VIZ_POINT_SIZE * 10,
                )

        # Apply cached segmentation filter
        filtered_pcd = self._apply_cached_sam2_filter(pcd_ret, device)

        # Apply dynamic proximity filter on top (handles in-hand object)
        return self._filter_point_cloud_for_collision(filtered_pcd, ee, gripper_state)

    def _setup_sdf_cost(self, collision_pcd, device):
        """Build an SDF cost field from the filtered collision point cloud (GPU-accelerated)."""
        pts = _to_tensor(collision_pcd["points"], device)
        esdf = compute_esdf_from_point_cloud_torch(pts, padding=0.3)
        # esdf.sdf/origin/range are already GPU tensors
        sdf_cost = SDFCost(
            esdf.sdf.to(device),
            esdf.origin.to(device),
            esdf.range.to(device),
            esdf.voxel_size,
        )
        return sdf_cost, self._wrist_probe.to(device)

    def _lift_clip_features(self, clip_feats, pcd_ret, images_preprocessed, device):
        """Lift pre-computed 2D CLIP features into 3D using the point map.

        Args:
            clip_feats: (N, C, pH, pW) from compute_2d_clip_features
            pcd_ret: point cloud dict (tensors on GPU)
            images_preprocessed: (N, 3, H', W') preprocessed images used for colors
        """
        return self.clip_lifter.lift_clip_features_to_3d(
            clip_feats=clip_feats,
            point_map=_to_tensor(pcd_ret["point_map"], device, dtype=None),
            images=images_preprocessed,
            valid_mask=_to_tensor(pcd_ret["depth_mask"], device, dtype=None),
            return_colors=True,
        )

    def _filter_semantic_objects(self, orig_pcd_ret, clip_points_3d, clip_features_3d, pixel_coords_2d, ee, gripper_state, device, prompt,
    negative_texts=None):
        """Filter point cloud to separate task-relevant objects from background for semantic guidance.

        All operations are GPU-native (tensors throughout). Returns tensors.

        Returns:
            tuple: (task_relevant_points tensor, background_pcd_ret dict with tensors)
        """
        # Use raw_points/raw_colors (depth_mask: conf & z_thresh, no arm dilation) — same
        # mask as clip_features_3d, so text_similarities indices align with points indices.
        # Arm exclusion is handled downstream by _filter_point_cloud_for_collision.
        points = _to_tensor(orig_pcd_ret["raw_points"], device)
        colors = _to_tensor(orig_pcd_ret["raw_colors"], device)
        color_map = _to_tensor(orig_pcd_ret["color_map"], device, dtype=None)  # (N, H, W, 3)
        pixel_coords_2d = _to_tensor(pixel_coords_2d, device, dtype=torch.long)

        if len(points) == 0:
            print("[CLIP] No points to filter — returning full pcd as background.")
            return torch.zeros((0, 3), device=device), dict(orig_pcd_ret)

        print(f"[CLIP] Computing text similarity for: '{prompt}'")
        text_similarities = self.clip_lifter.compute_text_similarity(
            positive_texts=[prompt],
            features=clip_features_3d,
            negative_texts=negative_texts,
            softmax_temperature=0.9,
            use_softmax=False,
        )  # (N,) GPU tensor

        if self.viz_viser:
            import matplotlib.cm as cm
            sim_np = text_similarities.cpu().numpy()
            cmap = cm.get_cmap("Blues")
            sim_normalized = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-8)
            heatmap_colors = torch.from_numpy((cmap(sim_normalized)[:, :3] * 255).astype(np.uint8)).float() / 255.0
            add_point_cloud(
                _to_tensor(clip_points_3d, device), colors=heatmap_colors.to(device),
                name="debug/clip/heatmap",
                point_size=_VIZ_POINT_SIZE,
                visible=True,
            )

        print(f"[CLIP] Similarity range: [{text_similarities.min():.3f}, {text_similarities.max():.3f}]")

        # Compute threshold
        similarity_threshold = torch.quantile(text_similarities, self.guidance_config.clip.percentile / 100.0)
        if self.guidance_config.clip.min_similarity is not None:
            min_thresh = self.guidance_config.clip.min_similarity * text_similarities.max()
            similarity_threshold = max(similarity_threshold, min_thresh)

        relevant_mask = text_similarities > similarity_threshold
        task_relevant_points = points[relevant_mask]
        task_relevant_colors = colors[relevant_mask]

        print(f"[CLIP] Threshold: {similarity_threshold:.3f} | Task-relevant: {relevant_mask.sum()} / {len(relevant_mask)} points")

        # Re-apply arm exclusion: raw_points (depth_mask) includes arm pixels; the collision
        # mask (pcd_ret["mask"]) does not.  Use pixel_coords_2d to look up which raw_points
        # are inside the arm-excluded collision mask, so arm pixels can't leak into the SDF.
        collision_mask_2d = _to_tensor(orig_pcd_ret["mask"], device, dtype=None)  # (N, H, W) bool
        in_collision_mask = collision_mask_2d[pixel_coords_2d[:, 0], pixel_coords_2d[:, 1], pixel_coords_2d[:, 2]]

        # Expand the target in metric 3D, not in image space.  A 2D dilation can cross
        # depth discontinuities when the target is partially occluded, pulling obstacle
        # pixels into the target region.  Distance in 3D keeps those surfaces separated.
        dilation_radius = self.guidance_config.collision_params.target_proximity_threshold
        if dilation_radius < 0:
            raise ValueError(f"target_proximity_threshold must be non-negative, got {dilation_radius}")

        if len(task_relevant_points) > 0:
            expand_relevant_mask = _chunked_cdist_min(points, task_relevant_points) <= dilation_radius
        else:
            expand_relevant_mask = torch.zeros_like(relevant_mask, dtype=torch.bool)

        target_region_mask = in_collision_mask & expand_relevant_mask
        confirmed_target_mask = in_collision_mask & relevant_mask
        target_border_mask = target_region_mask & ~relevant_mask
        near_target_3d_pts = points[target_border_mask]

        print(
            f"[CLIP-3D] Dilation radius: {dilation_radius:.3f}m | "
            f"target-region: {target_region_mask.sum()} collision points | "
            f"border-neutral: {target_border_mask.sum()} points"
        )

        if self.save_debug_images:
            N, H, W, _ = color_map.shape
            clip_mask_2d = torch.zeros((N, H, W), device=device, dtype=torch.bool)
            clip_coords = pixel_coords_2d[confirmed_target_mask]
            if len(clip_coords) > 0:
                clip_mask_2d[clip_coords[:, 0], clip_coords[:, 1], clip_coords[:, 2]] = True

            expanded_mask_2d = torch.zeros((N, H, W), device=device, dtype=torch.bool)
            expanded_coords = pixel_coords_2d[target_region_mask]
            if len(expanded_coords) > 0:
                expanded_mask_2d[expanded_coords[:, 0], expanded_coords[:, 1], expanded_coords[:, 2]] = True

            for i in range(N):
                clip_img = (color_map[i].cpu().numpy() * np.clip(clip_mask_2d[i].cpu().numpy()[..., None].astype(float) + 0.3, 0, 1.0))
                iio.imwrite(f"debug/images/debug_clip_pure_mask_{i}.png", (clip_img * 255).astype(np.uint8))

                out_img = (color_map[i].cpu().numpy() * np.clip(expanded_mask_2d[i].cpu().numpy()[..., None].astype(float) + 0.3, 0, 1.0))
                iio.imwrite(f"debug/images/debug_clip_3d_expanded_mask_{i}.png", (out_img * 255).astype(np.uint8))

        background_mask = ~expand_relevant_mask & in_collision_mask

        col_params = self.guidance_config.collision_params
        tip_position = ee @ self.tip_coord
        tip_pos = tip_position[0, :3]  # (3,) GPU tensor
        gripper_closed = gripper_state < col_params.gripper_closed_threshold
        clearance = col_params.ee_clearance_closed if gripper_closed else col_params.ee_clearance_open

        # EE clearance on task-relevant points: in-hand target should be neutral, not attractive.
        # This prevents self-attraction if an attractive guidance type is active.
        dist_task = torch.norm(task_relevant_points - tip_pos, dim=1)
        task_in_clearance = dist_task <= clearance
        task_relevant_points_neutral = task_relevant_points[task_in_clearance]
        task_relevant_points = task_relevant_points[~task_in_clearance]
        task_relevant_colors = task_relevant_colors[~task_in_clearance]
        print(f"[CLIP] EE clearance removed {task_in_clearance.sum()} task-relevant pts (in-hand); {len(task_relevant_points)} remain")

        # Hand self-filter on background
        bg_pts = points[background_mask]
        bg_colors = colors[background_mask]
        dist_to_hand = torch.norm(bg_pts - tip_pos, dim=1)
        hand_filter = dist_to_hand > clearance

        result_pcd = dict(orig_pcd_ret)
        result_pcd["points"] = bg_pts[hand_filter]
        result_pcd["colors"] = bg_colors[hand_filter]

        print(f"[CLIP] Background PCD: {len(result_pcd['points'])} points")

        n_raw = len(points)
        n_col = int(in_collision_mask.sum().item())
        n_target_region = int(target_region_mask.sum().item())
        n_confirmed_target = int(confirmed_target_mask.sum().item())
        n_background = int((in_collision_mask & ~expand_relevant_mask).sum().item())
        n_3d_removed = len(near_target_3d_pts)
        n_repulsive = len(result_pcd["points"])
        print(
            f"[COVERAGE] raw={n_raw}  collision={n_col}  "
            f"confirmed_target={n_confirmed_target}  target_region={n_target_region}  background={n_background}  "
            f"proximity_neutral={n_3d_removed}  repulsive={n_repulsive}  "
            f"target_region+background={'OK' if n_target_region + n_background == n_col else f'MISMATCH {n_target_region+n_background}!={n_col}'}"
        )

        if self.viz_viser:
            label = "closed" if gripper_closed else "open"

            # Split CLIP-confirmed target points by EE clearance.  The 3D-expanded
            # border stays neutral: excluded from repulsion, not used as target evidence.
            confirmed_target_pts = points[confirmed_target_mask]
            dist_confirmed = torch.norm(confirmed_target_pts - tip_pos, dim=1)
            confirmed_attractive_pts = confirmed_target_pts[dist_confirmed > clearance]
            confirmed_neutral_pts = confirmed_target_pts[dist_confirmed <= clearance]

            # attractive: CLIP-confirmed target outside EE clearance
            add_point_cloud(
                confirmed_attractive_pts,
                colors=_VIZ_ATTRACTIVE.expand(confirmed_attractive_pts.shape[0], 3),
                name="attractive/clip/target",
                point_size=_VIZ_POINT_SIZE,
            )

            # neutral/ee_clearance: confirmed target within clearance + background within clearance
            bg_neutral_pts = bg_pts[~hand_filter]
            ee_neutral_pts = torch.cat([confirmed_neutral_pts, bg_neutral_pts], dim=0)
            add_point_cloud(
                ee_neutral_pts,
                colors=_VIZ_NEUTRAL.expand(ee_neutral_pts.shape[0], 3),
                name=f"neutral/ee_clearance_{label}",
                point_size=_VIZ_POINT_SIZE,
            )

            # neutral/target_proximity: 3D-expanded target border, not CLIP-confirmed
            if len(near_target_3d_pts) > 0:
                prox_pts = _to_tensor(near_target_3d_pts, device)
                add_point_cloud(
                    prox_pts,
                    colors=_VIZ_NEUTRAL.expand(prox_pts.shape[0], 3),
                    name="neutral/target_border",
                    point_size=_VIZ_POINT_SIZE,
                )

            # Wireframe sphere showing the clearance boundary around the EE tip
            add_sphere(
                tip_pos.cpu().numpy(), radius=clearance,
                color=(230, 204, 26),  # yellow matching neutral
                name="neutral/ee_clearance_sphere",
                visible=False,
            )

        return task_relevant_points, result_pcd

    def _build_guidance_scene(self, observation, device, ee, all_trans, joint_positions, gripper_state, prompt, subtask):
        """Build the full guidance scene: point clouds, SDF cost, semantic targets.

        Returns a dict with keys:
          pcd_ret, sdf_cost, wrist_pts,
          task_relevant_points, subtask_relevant_points
        """
        scene = {
            "pcd_ret": None,
            "sdf_cost": None,
            "wrist_pts": None,
            "task_relevant_points": None,
            "subtask_relevant_points": None,
        }

        if not self.needs_pcd:
            return scene

        # --- Parallel VGGT + CLIP 2D ---
        # PyBullet renderer must run on the main thread; CLIP 2D is GPU-bound and can
        # run concurrently.  We submit CLIP 2D to a thread, run VGGT on main, then join.
        clip_feats = None
        images_preprocessed = None
        if self.needs_clip:
            images_tensor = self._stack_camera_tensors(observation, "image")
            images_preprocessed = preprocess_images(images_tensor, mode="crop")
            if self.save_debug_images:
                assert images_preprocessed.shape[0] == len(self.pcd_camera_views), (
                    f"expected {len(self.pcd_camera_views)} CLIP input views, got {images_preprocessed.shape[0]}"
                )
                for i, view_name in enumerate(self.pcd_camera_views):
                    img_np = images_preprocessed[i].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    iio.imwrite(
                        f"debug/images/debug_clip_input_{i}_{view_name}.png",
                        (img_np * 255).astype(np.uint8),
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_clip = (
                executor.submit(self.clip_lifter.compute_2d_clip_features, images_preprocessed)
                if self.needs_clip else None
            )
            pcd_ret = self._compute_point_cloud(observation, device)
            if future_clip is not None:
                clip_feats, _ = future_clip.result()

        scene["pcd_ret"] = pcd_ret

        # --- Lift CLIP features to 3D ---
        clip_points = clip_features = clip_colors = pixel_coords = None
        if self.needs_clip and clip_feats is not None:
            clip_points, clip_features, clip_colors, pixel_coords = self._lift_clip_features(
                clip_feats, pcd_ret, images_preprocessed, device
            )

        # --- Collision: build filtered point cloud and SDF cost ---
        # Priority for task-object filtering (so the robot is not repelled from its target):
        #   1. vlm_pointer + SAM2 — VLM identifies the target pixel; SAM2 segments it out.
        #      Used when vlm_pointer is already part of guidance (avoids running CLIP too).
        #   2. CLIP — text-similarity filter using the task prompt.  Default for
        #      collision-only or waypoint+collision modes.
        #   3. Proximity-only — only removes robot body / in-hand object via dynamic
        #      gripper exclusion radius.  Reached only if neither CLIP nor VLM is available.
        if "collision" in self.guidance_types:
            collision_active = self._is_guidance_active("collision")
            if "vlm_pointer" in self.guidance_types:
                # VLM+SAM2 always runs (regardless of active_subtask) so that:
                #   (a) subtask_relevant_points is populated for the freeze zone, and
                #   (b) the target object is excluded from the collision PCD.
                # The SDF cost is only built when collision is actually active.
                collision_pcd = self._apply_vlm_sam2_filter(
                    pcd_ret, observation, ee, gripper_state, device, prompt, subtask
                )
                if self.cached_sam2_relevant_points is not None:
                    scene["subtask_relevant_points"] = self.cached_sam2_relevant_points
            elif clip_points is not None:
                # CLIP removes task-relevant objects from the collision field.
                # The CLIP-matched target points are stored for the freeze zone check.
                # Dynamic proximity filter is applied on top to handle in-hand objects.
                clip_query = (subtask or prompt) if self.guidance_config.clip.use_subtask_as_query else prompt
                task_relevant_points, clip_background_pcd = self._filter_semantic_objects(
                    pcd_ret, clip_points, clip_features, pixel_coords, ee, gripper_state, device, clip_query
                )
                scene["subtask_relevant_points"] = task_relevant_points
                # show_viz=False: _filter_semantic_objects already rendered neutral/ee_clearance
                collision_pcd = self._filter_point_cloud_for_collision(clip_background_pcd, ee, gripper_state, show_viz=False)
            else:
                collision_pcd = self._filter_point_cloud_for_collision(pcd_ret, ee, gripper_state, show_viz=True)

            # Only build SDF cost when collision guidance is actually active this subtask.
            if collision_active:
                scene["sdf_cost"], scene["wrist_pts"] = self._setup_sdf_cost(collision_pcd, device)
            if self.viz_viser:
                # Repulsive region: the actual SDF collision field (arm+target excluded)
                add_point_cloud(
                    collision_pcd["points"],
                    colors=_VIZ_REPULSIVE.expand(collision_pcd["points"].shape[0], 3),
                    name="repulsive/collision_field",
                    point_size=_VIZ_POINT_SIZE,
                )

        # --- Waypoint: refresh world-frame trajectory for this observation ---
        if "waypoint" in self.guidance_types and self._is_guidance_active("waypoint"):
            self.waypoints.prepare(observation)

        # --- Scene visualization ---
        if self.viz_viser:
            self._visualize_scene_breakdown(pcd_ret, device)
            self._visualize_robot(ee, all_trans, joint_positions, device)
            if scene["subtask_relevant_points"] is not None:
                pts = _to_tensor(scene["subtask_relevant_points"], device)
                add_point_cloud(
                    pts,
                    colors=_VIZ_ATTRACTIVE.expand(pts.shape[0], 3),
                    name="attractive/freeze_zone_target",
                    point_size=_VIZ_POINT_SIZE,
                )

        return scene

    def _visualize_scene_breakdown(self, pcd_ret, device):
        """Visualize the full scene broken down by region type.

        reconstruction/raw   — full VGGT output, actual RGB (depth_mask: conf + z_thresh)
        neutral/arm          — rendered tight arm mask; excluded so robot doesn't repel itself
        neutral/arm_dilation — clearance margin around arm; excluded even though not the arm
        neutral/ee_clearance_* — dynamic EE proximity zone (populated by _filter_point_cloud_for_collision)
        """
        n_raw = len(pcd_ret["raw_points"])
        n_arm = len(pcd_ret["arm_points"])
        n_dil = len(pcd_ret["dilation_points"])
        n_col = len(pcd_ret["points"])
        total = n_arm + n_dil + n_col
        print(
            f"[PCD] raw={n_raw}  arm={n_arm}  dilation={n_dil}  collision={n_col}  "
            f"sum={total}  {'OK' if total == n_raw else f'MISMATCH gap={n_raw-total}'}"
        )

        raw_pts = _to_tensor(pcd_ret["raw_points"], device)
        raw_cols = _to_tensor(pcd_ret["raw_colors"], device)
        add_point_cloud(raw_pts, colors=raw_cols, name="reconstruction/raw", point_size=_VIZ_POINT_SIZE,
                        visible=True)

        arm_pts = _to_tensor(pcd_ret["arm_points"], device)
        add_point_cloud(
            arm_pts,
            colors=_VIZ_NEUTRAL.expand(arm_pts.shape[0], 3),
            name="neutral/arm",
            point_size=_VIZ_POINT_SIZE,
        )

        dil_pts = _to_tensor(pcd_ret["dilation_points"], device)
        add_point_cloud(
            dil_pts,
            colors=_VIZ_NEUTRAL.expand(dil_pts.shape[0], 3),
            name="neutral/arm_dilation",
            point_size=_VIZ_POINT_SIZE,
        )

    def _visualize_robot(self, ee, all_trans, joint_positions, device):
        """Visualize robot links and end-effector in the 3D viewer."""
        wrist_pos = ((all_trans[-2] @ self._wrist_probe.to(device).T)[0, :3, :]).T
        tip_position = ee @ self.tip_coord
        add_point_cloud(
            wrist_pos,
            colors=torch.tensor([[0.2, 0.4, 1.0]], device=device).expand(wrist_pos.shape[0], 3),
            name="robot/wrist_probe",
            point_size=_VIZ_POINT_SIZE,
        )
        add_point_cloud(
            joint_positions[0],
            colors=torch.tensor([[0.9, 0.2, 0.2]], device=joint_positions.device).expand(joint_positions.shape[1], 3),
            name="robot/joints",
            point_size=_VIZ_POINT_SIZE,
        )
        add_point_cloud(
            tip_position[:, :3],
            colors=torch.tensor([[0.1, 0.9, 0.1]], device=device).expand(tip_position.shape[0], 3),
            name="robot/ee",
            point_size=_VIZ_POINT_SIZE,
        )

    def _get_current_subtask(self, observation, prompt):
        """Query the progress tracker for the current subtask.

        Returns the subtask string, or None if subtask monitoring is disabled
        or the tracker returns nothing.
        """
        if not self.guidance_config.subtask_monitor.enabled:
            return None

        self.progress_tracker.set_subtasks(prompt)
        self.progress_tracker.save_obs(observation, self.global_count, downsample=True)

        # Only check subtask completion when relevant:
        # - No freeze zone: always check.
        # - With freeze zone: only check right after exiting (prevents premature advancement).
        if self.guidance_config.freeze_zone.enabled:
            just_exit_flag = True if self.just_exited_freezing_zone else False
        else:
            just_exit_flag = None

        subtask = self.progress_tracker.get_subtask(just_exit_freezing_zone=just_exit_flag)
        print(f"[SUBTASK MONITOR] Subtask: {subtask} | Just exited freezing zone: {just_exit_flag}")
        self.just_exited_freezing_zone = False
        return subtask

    def _update_freeze_zone(self, endpos_tip, subtask_relevant_points, subtask, device):
        """Update the freeze zone state based on current end-effector position.

        The freeze zone suppresses guidance when the end-effector is close enough
        to the target that the policy should handle the interaction unguided.
        Guidance resumes once the end-effector moves away from the target region.

        Returns:
            do_guidance (bool): True if guidance should be applied this step.
        """
        assert subtask_relevant_points is not None, "subtask_relevant_points required for freeze zone"
        subtask_relevant_tensor = _to_tensor(subtask_relevant_points, device)
        distances = torch.cdist(endpos_tip[0:1], subtask_relevant_tensor)
        min_distance = distances.min().item()

        # Per-subtask or global enter threshold
        if isinstance(self.guidance_config.freeze_zone.threshold, list):
            assert self.guidance_config.subtask_monitor.enabled
            enter_threshold = self.guidance_config.freeze_zone.threshold[self.progress_tracker.current_subtask]
        else:
            enter_threshold = self.guidance_config.freeze_zone.threshold

        if not self.semantic_freeze_zone and min_distance < enter_threshold:
            print(f"[FREEZE ZONE] Entering (dist={min_distance:.4f}m < {enter_threshold}m)")
            self.semantic_freeze_zone = True
            self.cached_task_relevant_pcd = subtask_relevant_tensor.clone()
        elif self.semantic_freeze_zone and self.cached_task_relevant_pcd is not None:
            cached_distances = torch.cdist(endpos_tip[0:1], self.cached_task_relevant_pcd)
            min_cached_distance = cached_distances.min().item()

            if isinstance(self.guidance_config.freeze_zone.exit_threshold, list):
                assert self.guidance_config.subtask_monitor.enabled
                exit_threshold = self.guidance_config.freeze_zone.exit_threshold[self.progress_tracker.current_subtask]
            else:
                exit_threshold = self.guidance_config.freeze_zone.exit_threshold

            if min_cached_distance > exit_threshold:
                print(f"[FREEZE ZONE] Exiting (dist={min_cached_distance:.4f}m > {exit_threshold}m)")
                self.semantic_freeze_zone = False
                self.cached_task_relevant_pcd = None
                self.just_exited_freezing_zone = True
            else:
                print(f"[FREEZE ZONE] Still active (dist={min_cached_distance:.4f}m <= {exit_threshold}m)")

        do_guidance = not self.semantic_freeze_zone
        print(f"[FREEZE ZONE] Active: {self.semantic_freeze_zone} | Min dist: {min_distance:.4f}m | Do guidance: {do_guidance}")
        return do_guidance, subtask_relevant_tensor

    # -------------------------------------------------------------------------
    # Per-guidance-type cost computation
    # -------------------------------------------------------------------------

    def _compute_collision_cost(self, pos, all_trans, sdf_cost, wrist_pts):
        """Compute SDF-based collision avoidance cost.

        Both the end-effector tip and the wrist body contribute a log-barrier cost
        when inside their respective SDF thresholds.
        """
        col_params = self.guidance_config.collision_params

        sdf_vals = sdf_cost(pos)
        wrist_pos = (
            torch.matmul(all_trans[-2], wrist_pts.T).permute(0, 2, 1)[:, :, :3]
        ).reshape(-1, 3)
        wrist_sdf_vals = sdf_cost(wrist_pos)

        ee_mask = sdf_vals <= col_params.ee_sdf_threshold
        wrist_mask = wrist_sdf_vals <= col_params.wrist_sdf_threshold

        cost = (col_params.ee_cost_weight * ee_mask * torch.log(sdf_vals.clamp(min=1e-5))).sum() + (
            col_params.wrist_cost_weight * wrist_mask * torch.log(wrist_sdf_vals.clamp(min=1e-5))
        ).sum()
        return cost, wrist_pos.reshape(-1, 4, 3)

    def _compute_waypoint_cost(self, pos, pcd_ret, observation):
        """Compute waypoint following cost.

        Uses sequential trajectory matching (post_traj=True when on the second
        subtask) or single-waypoint attraction otherwise.
        """
        post_traj = self.progress_tracker.current_subtask == 1
        print(f"[WAYPOINT] Post trajectory: {post_traj} | subtask: {self.progress_tracker.current_subtask}")

        if post_traj:
            matched_pts = self.waypoints.sequential_match(pos)
            if self.viz_viser:
                all_waypoints = self.waypoints.all_waypoints()
                add_point_cloud(
                    all_waypoints,
                    colors=_VIZ_ATTRACTIVE.expand(all_waypoints.shape[0], 3),
                    name="attractive/waypoint/all",
                    point_size=_VIZ_POINT_SIZE,
                )
                add_lines(
                    pos[: matched_pts.shape[0]],
                    matched_pts,
                    colors=torch.tensor((0.2, 1.0, 0.3)),
                    name="attractive/waypoint/matched_lines",  # green — EE-to-waypoint attraction lines
                    line_width=1.0,
                )
            cost = -(pos[: matched_pts.shape[0]] - matched_pts).pow(2).sum()
            target_pts = matched_pts
        else:
            current_waypoint = self.waypoints.start_waypoint().unsqueeze(0)
            if self.viz_viser:
                add_point_cloud(
                    current_waypoint,
                    colors=_VIZ_ATTRACTIVE.expand(current_waypoint.shape[0], 3),
                    name="attractive/waypoint/current",
                    point_size=_VIZ_POINT_SIZE,
                )
            cost = -(pos[: self.guidance_config.guidance.num_guided] - current_waypoint).pow(2).sum()
            dist = torch.norm(pos[0] - current_waypoint, p=2)
            if dist < 0.1:
                cost = cost * 0.0
            target_pts = current_waypoint

        print(f"[WAYPOINT] Cost: {cost.item():.4f}")
        return cost, target_pts

    def _is_guidance_active(self, guidance_type: str) -> bool:
        """Return False if active_subtask restricts this guidance to a different subtask."""
        sm = self.guidance_config.subtask_monitor
        if not sm.enabled:
            return True
        target = sm.active_subtask.get(guidance_type)
        if target is None:
            return True
        active = self.progress_tracker.current_subtask == target
        if not active:
            print(f"[{guidance_type.upper()}] Skipping: current subtask "
                  f"{self.progress_tracker.current_subtask} != active_subtask {target}")
        return active

    def _compute_vlm_pointer_cost(self, pos, pcd_ret, observation, prompt, subtask, device):
        """Compute VLM-pointer attraction cost.

        Attracts the end-effector toward the centroid of the SAM2-segmented target
        point cloud (cached from scene building).  The VLM is used only as the SAM2
        prompt — it does not directly produce the gradient target here.
        """
        if self.cached_sam2_relevant_points is None:
            print("[VLM POINTER] No SAM2 target cached yet; skipping")
            return torch.tensor(0.0, device=device, requires_grad=True), None

        attractive_pts = _to_tensor(self.cached_sam2_relevant_points, device).float()
        centroid = attractive_pts.mean(dim=0, keepdim=True)  # (1, 3)

        cost = -(pos[: self.guidance_config.guidance.num_guided] - centroid).pow(2).sum()
        return cost, centroid

    # -------------------------------------------------------------------------
    # Attention mask helpers
    # -------------------------------------------------------------------------

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    # -------------------------------------------------------------------------
    # Training helpers
    # -------------------------------------------------------------------------

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    # -------------------------------------------------------------------------
    # Inference: sample_actions
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self, device, observation, noise=None, num_steps=10, output_transform=None, prompt=None
    ) -> Tensor:
        """Run guided diffusion to produce an action trajectory.

        Inference proceeds in three stages:
        1. Scene setup: compute point cloud, SDF cost field, and CLIP features.
        2. Unguided denoising: a standard flow-matching rollout used for visualization
           and to compute the freeze zone decision.
        3. Guided denoising: re-runs the rollout from the same noise with gradient-based
           guidance added at each step. Skipped when the freeze zone is active.
        """
        import time as pytime

        print("Observation available attributes:", observation.__dict__.keys())
        print("Prompt:", prompt)
        if self.viz_viser:
            remove_all()

        bsize = observation.state.shape[0]
        gripper_state = observation.state[0, -1].item()

        # --- Robot kinematics ---
        transform1 = _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1))
        ee, all_trans, joint_positions = self._compute_robot_state(observation, output_transform)

        # --- Wrist camera frame (for visualization and saving) ---
        wrist_cam_trans = torch.eye(4)
        wrist_cam_trans[:3, 3] = torch.tensor([-0.051, -0.031, -0.064])
        wrist_cam_trans[:3, :3] = torch.tensor([[0.7071, 0.7071, 0.0], [-0.7071, 0.7071, 0.0], [0.0, 0.0, 1.0]])
        wrist_cam_trans = wrist_cam_trans @ torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.9397, 0.3420, 0.0], [0.0, -0.3420, 0.9397, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        wrist_cam_trans = wrist_cam_trans.to(device)
        wrist_cam = ee[0] @ wrist_cam_trans
        if self.viz_viser:
            add_frames(wrist_cam[None], color=(200, 100, 255), name="robot/wrist_cam", radius=0.005, coord_visible=True, visible=True)

        # --- Data collection ---
        save_data = (
            {
                "positions": [],
                "unguided_positions": [],
                "times": [],
                "guided": False,
                "wrist_cam": wrist_cam.detach().cpu(),
                "subtask": None,
                "do_guidance": None,
                "dist_to_freeze_zone": None,
                "task_relevant_points": None,
            }
            if self.save_state_dir
            else None
        )

        # --- Subtask monitoring ---
        subtask = self._get_current_subtask(observation, prompt)
        if save_data is not None:
            save_data["subtask"] = subtask

        # --- Build guidance scene ---
        scene = self._build_guidance_scene(observation, device, ee, all_trans, joint_positions, gripper_state, prompt, subtask)

        # --- Build image/language KV cache (shared across all denoising steps) ---
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        if noise is None:
            noise = self.sample_noise(actions_shape, device)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        if self.viz_viser:
            step_colors = (plt.cm.hsv(torch.linspace(0, 1, num_steps).cpu().numpy())[:, :3] * 255).astype(int)

        # =====================================================================
        # Stage 1: Unguided denoising pass
        # Used for visualization of the unguided trajectory and for computing
        # the end-effector position needed by the freeze zone check.
        # =====================================================================
        x_t = noise.clone()
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        step = 0
        endpos_tip = None

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t_0 = x_t + (-time) * v_t
            raw_action_dict = dict(actions=x_t_0[0], state=observation.state[0])
            out0 = output_transform.transforms[0](raw_action_dict)
            out1 = transform1(out0)
            action_t = out1["actions"][:, :8]
            _, endpos_t = self.fk.forward(action_t[..., :7])
            endpos_tip = torch.matmul(endpos_t, self.tip_coord[None][..., None])[:, :3, 0]

            x_t = x_t + dt * v_t
            time += dt

            if step == num_steps - 1:
                if self.viz_viser:
                    endpos_t[:, :3, 3] = endpos_tip
                    add_frames(endpos_t, color=step_colors[min(step, len(step_colors) - 1)], name=f"traj/unguided/{time.item():.2f}", radius=0.005, coord_visible=False, visible=True)
                if save_data is not None:
                    save_data["unguided_positions"].append(endpos_tip.detach().cpu())
            step += 1

        unguided_x_t = x_t.clone()

        # =====================================================================
        # Stage 2: Freeze zone decision
        # =====================================================================
        print(f"global count: {self.global_count}")
        print(f"[FREEZE ZONE] enter_threshold={self.guidance_config.freeze_zone.threshold}, "
              f"exit_threshold={self.guidance_config.freeze_zone.exit_threshold}")

        do_guidance = True
        subtask_relevant_tensor = None
        if self.guidance_config.freeze_zone.enabled:
            do_guidance, subtask_relevant_tensor = self._update_freeze_zone(
                endpos_tip, scene["subtask_relevant_points"], subtask, device
            )
            if save_data is not None:
                if subtask_relevant_tensor is not None:
                    save_data["task_relevant_points"] = subtask_relevant_tensor.detach().cpu()
                if subtask_relevant_tensor is not None:
                    cached = self.cached_task_relevant_pcd if self.semantic_freeze_zone else subtask_relevant_tensor
                    dist = torch.cdist(endpos_tip[0:1], cached).min().item()
                    save_data["dist_to_freeze_zone"] = dist

        if save_data is not None:
            save_data["do_guidance"] = do_guidance

        # =====================================================================
        # Stage 3: Guided denoising pass
        # =====================================================================
        valid_types = {"collision", "waypoint", "vlm_pointer"}
        invalid_types = self.guidance_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid guidance types: {invalid_types}")

        st = pytime.time()
        remaining_init_candidates = self.guidance_config.guidance.num_init_candidates
        if self.viz_viser:
            init_colors = (plt.cm.viridis(torch.linspace(0, 1, remaining_init_candidates + 1).cpu().numpy())[:, :3] * 255).astype(int)
            guided_colors = (plt.cm.cool(torch.linspace(0, 1, num_steps).cpu().numpy())[:, :3] * 255).astype(int)

        if do_guidance:
            x_t = noise.clone()
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            step = 0
            best_init_cost = -float("inf")
            best_init_x_t = x_t.clone()

            while time >= -dt / 2:
                expanded_time = time.expand(bsize)

                with torch.enable_grad():
                    x_t.requires_grad_()
                    v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

                    x_t_0 = x_t + (-time) * v_t
                    raw_action_dict = dict(actions=x_t_0[0], state=observation.state[0])
                    out0 = output_transform.transforms[0](raw_action_dict)
                    out1 = transform1(out0)
                    action_t = out1["actions"][:, :8]

                    _, all_trans_t = self.fk.forward(action_t[..., :7], all_trans=True)
                    pos = torch.matmul(all_trans_t[-1], self.tip_coord[None][..., None])[:, :3, 0]
                    if self.viz_viser:
                        pos.retain_grad()
                        all_trans_t[-2].retain_grad()

                    # retain_graph is True for all backwds except the last, so intermediate
                    # tensors in the shared x_t→pos graph aren't freed prematurely.
                    _n_remaining = sum(t in self.guidance_types for t in ["collision", "waypoint", "vlm_pointer"])
                    prev_grad = None  # accumulated gradient, used only for per-type viz
                    total_cost = 0.0

                    # --- Collision guidance ---
                    if "collision" in self.guidance_types and self._is_guidance_active("collision"):
                        assert scene["sdf_cost"] is not None
                        cost, wrist_pos = self._compute_collision_cost(pos, all_trans_t, scene["sdf_cost"], scene["wrist_pts"])
                        total_cost += cost.item()
                        _n_remaining -= 1
                        cost.backward(retain_graph=_n_remaining > 0)

                        if self.viz_viser:
                            collision_grad = pos.grad if prev_grad is None else pos.grad - prev_grad
                            add_lines(pos, pos + collision_grad * 0.02, colors=torch.tensor((1.0, 0.2, 0.8)),
                                      name=f"grad/collision/{time.item():.2f}", line_width=2.0)
                            prev_grad = pos.grad.clone()

                    # --- Waypoint guidance ---
                    if "waypoint" in self.guidance_types and self._is_guidance_active("waypoint"):
                        assert scene["pcd_ret"] is not None
                        cost, target_pts = self._compute_waypoint_cost(pos, scene["pcd_ret"], observation)
                        total_cost += cost.item()
                        _n_remaining -= 1
                        cost.backward(retain_graph=_n_remaining > 0)

                        if self.viz_viser:
                            waypoint_grad = pos.grad if prev_grad is None else pos.grad - prev_grad
                            add_lines(pos, pos + waypoint_grad * 0.02, colors=torch.tensor((0.2, 1.0, 0.3)),
                                      name=f"grad/waypoint/{time.item():.2f}", line_width=2.0)
                            prev_grad = pos.grad.clone()

                    # --- VLM pointer guidance ---
                    if "vlm_pointer" in self.guidance_types and self._is_guidance_active("vlm_pointer"):
                        assert scene["pcd_ret"] is not None
                        cost, target_pts = self._compute_vlm_pointer_cost(pos, scene["pcd_ret"], observation, prompt, subtask, device)
                        total_cost += cost.item()
                        _n_remaining -= 1
                        cost.backward(retain_graph=_n_remaining > 0)

                        if self.viz_viser:
                            vlm_grad = pos.grad if prev_grad is None else pos.grad - prev_grad
                            add_lines(pos, pos + vlm_grad * 0.02, colors=torch.tensor((0.0, 0.9, 1.0)),
                                      name=f"grad/vlm/{time.item():.2f}", line_width=2.0)
                            prev_grad = pos.grad.clone()

                    # --- Combined gradient (viz only) ---
                    if self.viz_viser:
                        guidance_vec = pos.grad if pos.grad is not None else torch.zeros_like(pos)
                        add_lines(pos, pos + guidance_vec * 0.02, colors=torch.tensor((1.0, 0.95, 0.0)),
                                  name=f"grad/total/{time.item():.2f}", line_width=3.0)
                        add_point_cloud(pos, colors=guided_colors[min(step, len(guided_colors) - 1)],
                                        name=f"traj/pos/{time.item():.2f}")

                    if save_data is not None:
                        save_data["positions"].append(pos.detach().cpu())
                        save_data["times"].append(time.item())
                        save_data["guided"] = True

                    grad_raw = x_t.grad if x_t.grad is not None else torch.zeros_like(x_t)
                    guide = grad_raw

                # --- Best-of-N initial noise selection ---
                if remaining_init_candidates > 0:
                    if self.viz_viser:
                        add_point_cloud(pos, colors=init_colors[remaining_init_candidates],
                                        name=f"traj/init/{remaining_init_candidates}")
                    print(
                        f"[GUIDANCE INIT] candidate={self.guidance_config.guidance.num_init_candidates - remaining_init_candidates + 1}/"
                        f"{self.guidance_config.guidance.num_init_candidates} total_cost={total_cost:.4f} best_cost={best_init_cost:.4f}"
                    )
                    if total_cost > best_init_cost:
                        best_init_cost = total_cost
                        best_init_x_t = x_t.detach().clone()
                    remaining_init_candidates -= 1
                    if remaining_init_candidates == 0:
                        print(f"[GUIDANCE INIT] Selected best initial noise with cost={best_init_cost:.4f}")
                        x_t = best_init_x_t.clone()
                    else:
                        x_t = self.sample_noise(actions_shape, device)
                    continue

                # --- Guidance update (Euler + guidance) ---
                guide_length = guide.norm(p=2, dim=-1, keepdim=True).clamp(max=self.guidance_config.guidance.max_guide_length)
                guide = guide / (guide.norm(p=2, dim=-1, keepdim=True) + 1e-8) * guide_length
                x_t = x_t + dt * v_t + self.guidance_config.guidance.factor * guide
                time += dt

                if self.viz_viser and step == num_steps - 1:
                    all_trans_t[-1][:, :3, 3] = pos
                    add_frames(all_trans_t[-1], color=guided_colors[min(step, len(guided_colors) - 1)], name=f"traj/guided/{time.item():.2f}", radius=0.005, coord_visible=False, visible=True)

                step += 1
        else:
            x_t = unguided_x_t

        self.global_count += 1
        print(f"Guided diffusion time: {pytime.time() - st:.3f}s")

        # --- Save all collected data ---
        if save_data is not None:
            if len(save_data["positions"]) > 0:
                save_data["positions"] = torch.stack(save_data["positions"])
                save_data["unguided_positions"] = torch.stack(save_data["unguided_positions"])
                save_data["times"] = torch.tensor(save_data["times"])
            save_data["global_count"] = self.global_count - 1
            save_path = f"{self.save_state_dir}/actions/sample_data_{self.global_count - 1}.pt"
            torch.save(save_data, save_path)
            print(f"[SAVE] Saved sample data to {save_path}")

        if self.debug_pause_before_action_return:
            print("[DEBUG] Action chunk ready.")
            print("[DEBUG] Open http://localhost:8080 to inspect Viser before the action is returned.")
            input("[DEBUG] Press Enter to return the action to the client...")

        return x_t
        

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
