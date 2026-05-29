import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import numpy as np
import imageio.v2 as iio
from scipy.spatial.transform import Rotation as R
from vis.pybullet_render import PyBulletRender, cvExtrinMatW2C2XYZEulerC2W


class PCD_Generator:
    def __init__(self, device="cuda", arm_mask_dilation: int = 29):
        self.device = device
        self.arm_mask_dilation = arm_mask_dilation
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        self.renderer = PyBulletRender(urdf_dir="franka_panda/panda_gripper.urdf")

    @staticmethod
    def _invert_extrinsics(extrin: torch.Tensor) -> torch.Tensor:
        """Invert a batch of (N, 4, 4) rigid-body extrinsic matrices (works w2c↔c2w)."""
        R_t = extrin[:, :3, :3].transpose(1, 2)          # (N, 3, 3)
        t_inv = -(R_t @ extrin[:, :3, 3:4]).squeeze(-1)  # (N, 3)
        inv = extrin.clone()
        inv[:, :3, :3] = R_t
        inv[:, :3, 3] = t_inv
        return inv

    def compute_scale_trans(self, extrinsics_gt, extrinsics_est):
        """Align estimated camera poses to ground-truth scale and orientation.

        Args:
            extrinsics_gt:  (N, 4, 4) ground-truth w2c extrinsics
            extrinsics_est: (N, 4, 4) VGGT-estimated w2c extrinsics

        Returns:
            (scale, aligned_w2c): scale factor and (N, 4, 4) aligned w2c tensor.
        """
        assert extrinsics_gt.shape == extrinsics_est.shape, (
            f"extrinsic shape mismatch: gt={extrinsics_gt.shape}, est={extrinsics_est.shape}"
        )
        assert extrinsics_gt.shape[0] >= 2, "VGGT scale alignment requires at least two camera views"

        extrinsics_gt_c2w  = self._invert_extrinsics(extrinsics_gt)
        extrinsics_est_c2w = self._invert_extrinsics(extrinsics_est)

        # Inter-camera distances to compute relative scale
        t_gt  = extrinsics_gt_c2w[:, :3, 3]   # (N, 3)
        t_est = extrinsics_est_c2w[:, :3, 3]
        pairs = [(i, j) for i in range(t_gt.shape[0]) for j in range(i + 1, t_gt.shape[0])]
        dist_gt = [torch.norm(t_gt[i] - t_gt[j]).item() for i, j in pairs]
        dist_est = [torch.norm(t_est[i] - t_est[j]).item() for i, j in pairs]
        scales = [dist_gt[i] / dist_est[i] for i in range(len(pairs))]
        scale_str = ", ".join(f"{s:.4f}" for s in scales)
        print(f"Scale factors: {scale_str}, avg: {sum(scales)/len(scales):.4f}")

        # Use scale_01 to rescale estimated translations (consistent with original)
        scale = scales[0]
        extrinsics_est_c2w[:, :3, 3] *= scale

        # Rotation & translation alignment (average cameras 0 and 1)
        R_gt  = extrinsics_gt_c2w[:, :3, :3]   # (3, 3, 3)
        R_est = extrinsics_est_c2w[:, :3, :3]
        rot_mats = R_gt[:2] @ R_est[:2].transpose(1, 2)  # (2, 3, 3)
        trans_vecs = t_gt[:2] - (rot_mats @ t_est[:2, :, None]).squeeze(-1)  # (2, 3)
        trans_final = trans_vecs.mean(dim=0)  # (3,)

        # Average rotations via rotvec interpolation (scipy for SLERP)
        rot_vecs = R.from_matrix(rot_mats.cpu().numpy()).as_rotvec()  # (2, 3)
        rot_final = torch.from_numpy(
            R.from_rotvec(rot_vecs.mean(axis=0)).as_matrix()
        ).to(self.device).float()

        # Apply alignment to all three cameras
        extrin_est_c2w_aligned = extrinsics_est_c2w.clone()
        extrin_est_c2w_aligned[:, :3, :3] = rot_final @ R_est        # (3, 3, 3)
        extrin_est_c2w_aligned[:, :3, 3]  = (rot_final @ t_est.unsqueeze(-1)).squeeze(-1) + trans_final

        return scale, self._invert_extrinsics(extrin_est_c2w_aligned)  # c2w → w2c

    def _render_arm_mask(self, cam_idx, extrinsic, intrinsic, image, joint_angles, save_debug_images=False):
        """Render the robot arm mask for a single camera view.

        Args:
            cam_idx: Camera index (used for debug image naming only)
            extrinsic: (4, 4) c2w extrinsic matrix (torch tensor)
            intrinsic: (3, 3) intrinsic matrix (torch tensor)
            image: (3, H, W) float tensor in [0, 1]
            joint_angles: (8,) float tensor — 7 joint angles + 1 gripper position
            save_debug_images: Write debug overlay PNGs to disk (expensive I/O, off by default)

        Returns:
            tuple: (mask_render, mask_dilate) — bool torch tensors of shape (H, W) on GPU
        """
        device = image.device
        self.renderer.set_camera(
            cvExtrinMatW2C2XYZEulerC2W(extrinsic.cpu().numpy()),
            intrinsic.cpu().numpy(),
        )
        _, _, mask_render_np = self.renderer.render(
            joint_positions=joint_angles[:-1].cpu().numpy(),
            gripper_position=joint_angles[-1:].cpu().numpy(),
        )
        mask_render = torch.from_numpy(mask_render_np).to(device)
        k = self.arm_mask_dilation
        mask_dilate = F.max_pool2d(
            mask_render.float().unsqueeze(0).unsqueeze(0),
            kernel_size=k, stride=1, padding=k // 2,
        ).squeeze(0).squeeze(0) > 0.5

        if save_debug_images:
            img_np = image.cpu().permute(1, 2, 0).numpy().copy()
            mask_np = mask_render_np.astype(bool)
            img_np[mask_np] = img_np[mask_np] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
            iio.imwrite(f"debug/images/arm_cam{cam_idx}_rendered.png", (img_np * 255).astype(np.uint8))
            img_np2 = image.cpu().permute(1, 2, 0).numpy().copy()
            dil_np = mask_dilate.cpu().numpy()
            img_np2[dil_np] = img_np2[dil_np] * 0.5 + np.array([0.0, 0.0, 1.0]) * 0.5
            iio.imwrite(f"debug/images/arm_cam{cam_idx}_dilate.png", (img_np2 * 255).astype(np.uint8))

        return mask_render.bool(), mask_dilate

    @torch.inference_mode()
    def generate_pcd(self, images, intrinsics, extrinsics, depth=None, joint_angles=None, conf_thresh=2.0, z_thresh=0.0,
                     save_debug_images=False):
        """
        Create point cloud from multiple camera views and their intrinsics and extrinsics.

        Args:
            images (torch.Tensor): Tensor of shape (N, 3, H, W) where N is number of views
            intrinsics (torch.Tensor): Tensor of shape (N, 3, 3)
            extrinsics (torch.Tensor): Tensor of shape (N, 4, 4)
            joint_angles (torch.Tensor, optional): Tensor of joint angles, if available. (8,)
            conf_thresh (float, optional): Confidence threshold for filtering points.
            z_thresh (float, optional): Z threshold for filtering points by height.

        Returns:
            dict: A dictionary containing point cloud data and optionally color information.
        """
        device = self.device
        num_views, _, H, W = images.shape
        assert intrinsics.shape[0] == num_views, (
            f"intrinsics view count {intrinsics.shape[0]} != image view count {num_views}"
        )
        assert extrinsics.shape[0] == num_views, (
            f"extrinsics view count {extrinsics.shape[0]} != image view count {num_views}"
        )
        assert depth is None or depth.shape[0] == num_views, (
            f"depth view count {depth.shape[0]} != image view count {num_views}"
        )
        assert depth is not None or num_views >= 2, "VGGT point-cloud generation requires at least two camera views"

        extrinsics = self._invert_extrinsics(extrinsics)  # w2c → c2w

        # Render arm masks for all cameras and dilate on GPU
        renders = [
            self._render_arm_mask(i, extrinsics[i], intrinsics[i], images[i], joint_angles,
                                  save_debug_images=save_debug_images)
            for i in range(num_views)
        ]
        # renders[i] = (mask_render_bool, mask_dilate_bool) — both (H, W) GPU tensors
        dilated = torch.stack([r[1] for r in renders])[:, None].float()  # (N, 1, H, W)
        tight = torch.stack([r[0] for r in renders])[:, None].float()   # (N, 1, H, W)

        dilated_proc = preprocess_images(dilated, mode="crop")  # (N, 1, H', W')
        tight_proc = preprocess_images(tight, mode="crop")

        # valid_no_arm: region NOT covered by dilated arm mask → used for the collision cloud
        valid_no_arm_masks = (dilated_proc == 0.0)[:, 0, :, :]   # (N, H', W') bool
        # arm_only: tight arm mask after preprocessing → visualised separately
        arm_only_masks = (tight_proc > 0.5)[:, 0, :, :]           # (N, H', W') bool

        images = preprocess_images(images, mode="crop")

        images = images[None]  # add batch dimension

        if depth is None:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic_est, intrinsic_est = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            final_scale, extrinsic_est_aligned = self.compute_scale_trans(
                extrinsics,
                torch.cat(
                    [
                        extrinsic_est[0],
                        torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device).repeat(extrinsic_est.shape[1], 1, 1),
                    ],
                    dim=1,
                ),
            )

            depth_map = depth_map * final_scale

            def _unproject(dm, extrin, intrin):
                result = unproject_depth_map_to_point_map(dm, extrin, intrin)
                if not isinstance(result, torch.Tensor):
                    result = torch.from_numpy(result)
                return result.to(device)

            point_map_by_unprojection_raw = _unproject(
                depth_map.squeeze(0), extrinsics.squeeze(0), intrinsic_est.squeeze(0)
            )
            point_map_by_unprojection_aligned = _unproject(
                depth_map.squeeze(0), extrinsic_est_aligned, intrinsic_est.squeeze(0)
            )

            if z_thresh is not None:
                valid_space_mask = point_map_by_unprojection_raw[..., 2] > z_thresh
            else:
                valid_space_mask = torch.ones_like(point_map_by_unprojection_raw[..., 2], dtype=torch.bool)

            valid_space_mask = valid_space_mask & torch.isfinite(point_map_by_unprojection_raw).all(dim=-1)

            conf = depth_conf[0] > conf_thresh
            mask = conf & valid_no_arm_masks.to(device) & valid_space_mask
            arm_mask = conf & arm_only_masks.to(device) & valid_space_mask
            depth_mask = conf & valid_space_mask
            # dilation_mask: valid points in the arm's clearance margin (excluded from
            # collision field by dilation but not actually on the arm itself).
            dilation_mask = depth_mask & ~mask & ~arm_mask

        else:
            # Ground-truth depth: skip VGGT entirely.
            # extrinsics are already c2w (inverted at the top of this function).
            if depth.ndim == 4:
                depth = depth[:, 0]  # (N, 1, H, W) → (N, H, W)

            # Preprocess depth to match preprocessed image resolution (nearest to preserve values)
            H_proc, W_proc = images.shape[-2], images.shape[-1]
            depth_proc = preprocess_images(
                depth.unsqueeze(1).float(), mode="crop", interpolation="nearest"
            )[:, 0].to(device)  # (N, H_proc, W_proc)

            # Scale GT intrinsics to the preprocessed image resolution
            x_scale = W_proc / W
            y_scale = H_proc / H
            intrinsic_scaled = intrinsics.clone().float().to(device)
            intrinsic_scaled[:, 0, :] *= x_scale  # fx, cx
            intrinsic_scaled[:, 1, :] *= y_scale  # fy, cy

            def _unproject(dm, extrin, intrin):
                result = unproject_depth_map_to_point_map(dm, extrin, intrin)
                if not isinstance(result, torch.Tensor):
                    result = torch.from_numpy(result)
                return result.to(device)

            # For GT depth, raw == aligned (no pose estimation needed)
            point_map = _unproject(depth_proc.unsqueeze(-1), extrinsics, intrinsic_scaled)  # (N, H_proc, W_proc, 3)
            point_map_by_unprojection_raw = point_map
            point_map_by_unprojection_aligned = point_map

            if z_thresh is not None:
                valid_space_mask = point_map_by_unprojection_raw[..., 2] > z_thresh
            else:
                valid_space_mask = torch.ones_like(point_map_by_unprojection_raw[..., 2], dtype=torch.bool)

            valid_space_mask = valid_space_mask & torch.isfinite(point_map_by_unprojection_raw).all(dim=-1)

            # Use depth validity (> 0 and < 10 m) instead of VGGT confidence
            conf = (depth_proc > 0) & (depth_proc < 10.0)
            mask = conf & valid_no_arm_masks.to(device) & valid_space_mask
            arm_mask = conf & arm_only_masks.to(device) & valid_space_mask
            depth_mask = conf & valid_space_mask
            dilation_mask = depth_mask & ~mask & ~arm_mask

            extrinsic_est_aligned = extrinsics
            intrinsic_est = intrinsic_scaled.unsqueeze(0)  # (1, N, 3, 3) to match VGGT shape

        color_map = images[0].permute(0, 2, 3, 1)  # (N, H', W', 3) GPU tensor
        ret = {
            "points": point_map_by_unprojection_aligned[mask].float(),
            "colors": color_map[mask].float(),
            "arm_points": point_map_by_unprojection_aligned[arm_mask].float(),
            "arm_colors": color_map[arm_mask].float(),
            "raw_points": point_map_by_unprojection_aligned[depth_mask].float(),
            "raw_colors": color_map[depth_mask].float(),
            "point_map": point_map_by_unprojection_aligned,
            "color_map": color_map,
            "mask": mask,
            "depth_mask": depth_mask,
            "arm_mask": arm_mask,
            "dilation_mask": dilation_mask,
            "dilation_points": point_map_by_unprojection_aligned[dilation_mask].float(),
            "dilation_colors": color_map[dilation_mask].float(),
            "images": images[0],
            "pointmap_by_unprojection": point_map_by_unprojection_raw,
            "extrinsic_est_aligned": extrinsic_est_aligned,
            "intrinsic_est": intrinsic_est,
        }

        return ret


def preprocess_images(image_tensor, mode="crop", interpolation="bicubic"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_tensor (torch.Tensor) : N, C, H, W, float32 tensor with values in [0, 1]
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    assert image_tensor.ndim == 4, "Input tensor must have shape (N, 3, H, W)"
    N, C, H, W = image_tensor.shape

    ratio = 518.0 / W
    new_W = 518
    new_H = round(H * (new_W / W) / 14) * 14  # must be divisible by 14

    if mode == "crop":
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(new_H, new_W), mode=interpolation, align_corners=False if interpolation != "nearest" else None
        )
        if new_H > 518:
            top = (new_H - 518) // 2
            image_tensor = image_tensor[:, :, top : top + 518, :]
    else:  # mode == "pad"
        # Resize and pad to square
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(new_H, new_W), mode=interpolation, align_corners=False if interpolation != "nearest" else None
        )
        if new_H < 518:
            pad_top = (518 - new_H) // 2
            pad_bottom = 518 - new_H - pad_top
            padding = (0, 0, pad_top, pad_bottom)  # (left, right, top, bottom)
            image_tensor = torch.nn.functional.pad(image_tensor, padding, mode="constant", value=1.0)  # Pad with white

    return image_tensor
