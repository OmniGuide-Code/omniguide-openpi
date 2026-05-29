import numpy as np
import torch
import imageio.v3 as iio
import os
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module


class SAM2Segmenter:
    """Handler for SAM2-based object segmentation from point prompts."""
    
    def __init__(self, model_name="facebook/sam2-hiera-large"):
        """Initialize SAM2 predictor.
        
        Args:
            model_name (str): HuggingFace model ID for SAM2
        """
        # Clear Hydra's global state to avoid config path conflicts
        # This is necessary because SAM2 uses Hydra internally and may conflict
        # with any existing Hydra initialization from the main application
        GlobalHydra.instance().clear()
        
        # Re-initialize Hydra with SAM2's config module (same as SAM2's __init__.py)
        # This sets up the correct search paths for SAM2's configs
        initialize_config_module("sam2", version_base="1.2")
        
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
    
    def segment_and_filter_pcd(self, uv, cam_idx, pcd_ret, distance_threshold=0.01, save_debug_images=False):
        """Segment an object from a single point prompt and remove it from the collision cloud.

        Runs SAM2 on the camera image at `cam_idx` using `uv` as the prompt point.
        The resulting 3D object points are then used to filter the full combined point
        cloud (all cameras) via GPU-native 3D proximity exclusion (torch.cdist).  This
        guarantees that the target is fully cleared from the collision field even in
        camera views where it is partially occluded and would otherwise be missed by
        per-view filters.

        Args:
            uv (tuple): (u, v) = (col, row) pixel coordinates in `cam_idx`'s image space.
            cam_idx (int): Camera index to segment in (0=left, 1=right, 2=overhead).
            pcd_ret (dict): Point cloud data; "points" and "colors" are filtered in-place.
            distance_threshold (float): 3D exclusion radius (m) around segmented points.

        Returns:
            dict: {"relevant_points": torch.Tensor (N,3), "relevant_colors": torch.Tensor (N,3)}
        """
        u, v = uv
        img = pcd_ret["images"][cam_idx]  # (3, H, W) GPU tensor
        device = img.device
        img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)  # (H, W, 3)

        self.predictor.set_image(img_np)
        masks, _, _ = self.predictor.predict(
            point_coords=np.array([[u, v]]),
            point_labels=np.ones(1),
            multimask_output=False,
        )
        # Convert SAM2 numpy mask to GPU bool tensor
        sam_mask = torch.from_numpy(masks[0].astype(bool)).to(device)  # (H, W)

        if save_debug_images:
            os.makedirs("debug/images", exist_ok=True)
            color_map = pcd_ret["color_map"][cam_idx]  # (H, W, 3) GPU tensor
            color_np = color_map.cpu().numpy()
            out_img = color_np * np.clip(sam_mask.cpu().numpy()[..., None].astype(float) + 0.3, 0, 1.0)
            iio.imwrite(f"debug/images/sam_mask_{cam_idx}.png", (out_img * 255).astype(np.uint8))

        point_map = pcd_ret["point_map"][cam_idx]   # (H, W, 3) GPU tensor
        color_map_cam = pcd_ret["color_map"][cam_idx]  # (H, W, 3) GPU tensor

        relevant_points_3d = point_map[sam_mask]    # (M, 3) GPU tensor
        relevant_colors = color_map_cam[sam_mask]   # (M, 3) GPU tensor

        valid = torch.isfinite(relevant_points_3d).all(dim=1)
        relevant_points_3d = relevant_points_3d[valid]
        relevant_colors = relevant_colors[valid]

        print(f"[SAM2] Camera {cam_idx}: segmented {len(relevant_points_3d)} object points")

        if len(relevant_points_3d) == 0:
            print("[SAM2] No valid object points found; collision cloud unchanged")
            return {"relevant_points": relevant_points_3d, "relevant_colors": relevant_colors}

        # 3D proximity exclusion: GPU-native, chunked to avoid OOM.
        # For each query point find its minimum distance to any segmented object point.
        current_pts = pcd_ret["points"]
        if not isinstance(current_pts, torch.Tensor):
            current_pts = torch.from_numpy(current_pts).to(device)
        current_pts = current_pts.float()
        obj_pts = relevant_points_3d.float()

        chunk_size = 8192
        min_distances = torch.empty(current_pts.shape[0], device=device)
        for start in range(0, current_pts.shape[0], chunk_size):
            chunk = current_pts[start : start + chunk_size]
            min_distances[start : start + chunk_size] = torch.cdist(chunk, obj_pts).min(dim=1)[0]
        background_mask = min_distances > distance_threshold

        colors = pcd_ret["colors"]
        if not isinstance(colors, torch.Tensor):
            colors = torch.from_numpy(colors).to(device)

        pcd_ret["points"] = current_pts[background_mask]
        pcd_ret["colors"] = colors[background_mask]

        print(f"[SAM2] Removed {(~background_mask).sum()} object points; "
              f"{len(pcd_ret['points'])} background points remain")

        return {"relevant_points": relevant_points_3d, "relevant_colors": relevant_colors}

