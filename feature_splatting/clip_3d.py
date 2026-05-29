"""
CLIP 3D Lifting: Extract CLIP features from images and lift to 3D point cloud.

Minimal version adapted from feature-splatting-inria for integration with OpenPI backend.
Includes object-aware CLIP features using MobileSAMv2 + YOLO (similar to reference code).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Optional, Tuple, List, Any, Generator
import maskclip_onnx


# ----------------------------- Utils -----------------------------


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """Helper to iterate over batches."""
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args)
    n = len(args[0])
    n_batches = n // batch_size + int(n % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


class MaskCLIPFeaturizer(nn.Module):
    """CLIP feature extractor that returns patch-level features."""

    def __init__(self, model_name="ViT-L/14@336px"):
        super().__init__()
        download_root = os.getenv("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
        self.model, self.preprocess = maskclip_onnx.clip.load(model_name, download_root=download_root)
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img_bchw: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP patch features.

        Args:
            img_bchw: (B, 3, H, W) normalized images

        Returns:
            feats: (B, C, P_h, P_w) CLIP patch tokens arranged on a grid
        """
        b, _, H, W = img_bchw.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        feats = self.model.get_patch_encodings(img_bchw).to(torch.float32)  # (B, ph*pw, C)
        feats = feats.reshape(b, ph, pw, -1).permute(0, 3, 1, 2).contiguous()
        return feats


class CLIP3DLifter:
    """
    Extract CLIP features from images and lift to 3D point cloud using VGGT point maps.
    """

    def __init__(
        self,
        device="cuda",
        clip_model_name="ViT-L/14@336px",
        feature_type="plain",
        mobilesamv2_encoder="mobilesamv2_efficientvit_l2",
    ):
        """
        Initialize CLIP 3D lifter.

        Args:
            device: torch device
            clip_model_name: CLIP model variant
            feature_type: "plain" for pixel-level CLIP features,
                         "object" for object-aware aggregated features (requires MobileSAMv2+YOLO)
            mobilesamv2_encoder: MobileSAMv2 encoder variant (only used if feature_type="object")
        """
        self.device = device
        self.clip_model = MaskCLIPFeaturizer(model_name=clip_model_name).to(device).eval()
        self.feature_type = feature_type

        # CLIP normalization transform
        self.clip_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Store the base CLIP model for text encoding
        self.base_clip_model = self.clip_model.model

        # Load MobileSAMv2 + YOLO for object detection (only if object mode)
        if feature_type == "object":
            print(f"[CLIP3DLifter] Loading MobileSAMv2 ({mobilesamv2_encoder}) + YOLO for object detection...")

            self.mobilesamv2, self.yolo_model, self.sam_predictor = torch.hub.load(
                "RogerQi/MobileSAMV2", mobilesamv2_encoder
            )

            self.mobilesamv2.to(device=device).eval()
            print("[CLIP3DLifter] MobileSAMv2 + YOLO loaded successfully")
        else:
            self.mobilesamv2 = None
            self.yolo_model = None
            self.sam_predictor = None

        print(f"[CLIP3DLifter] Initialized with model: {clip_model_name}, feature_type: {feature_type}")

    @torch.no_grad()
    def extract_clip_features(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract CLIP features from images. Can optionally aggregate by object masks.

        Args:
            images: (N, 3, H, W) torch tensor, values in [0, 1]
            masks: Optional (N, M, H, W) object masks for object-aware aggregation

        Returns:
            features: (N, C, fH, fW) CLIP features
        """
        assert images.ndim == 4 and images.shape[1] == 3, f"Expected (N, 3, H, W), got {images.shape}"

        # Normalize for CLIP
        images_normalized = torch.stack([self.clip_norm(img) for img in images])
        images_normalized = images_normalized.to(self.device)

        # Extract CLIP features
        clip_feats = self.clip_model(images_normalized)  # (N, C, pH, pW)

        # If object mode and masks provided, aggregate within masks
        if self.feature_type == "object" and masks is not None:
            clip_feats = self._aggregate_features_by_masks(images, clip_feats, masks)

        return clip_feats

    @torch.no_grad()
    def _aggregate_features_by_masks(
        self, images: torch.Tensor, clip_feats: torch.Tensor, masks: torch.Tensor, obj_feat_res: int = 100
    ) -> torch.Tensor:
        """
        Aggregate CLIP features within object masks (inspired by reference code).

        Args:
            images: (N, 3, H, W) original images
            clip_feats: (N, C, pH, pW) CLIP patch features
            masks: (N, M, H, W) object masks for each view
            obj_feat_res: intermediate resolution for aggregation

        Returns:
            aggregated_feats: (N, C, pH, pW) object-aware CLIP features
        """
        N, C, pH, pW = clip_feats.shape
        _, _, H, W = images.shape

        aggregated_list = []

        for i in range(N):
            # Compute intermediate resolution maintaining aspect ratio
            obj_W = obj_feat_res
            obj_H = int(round(H * (obj_W / float(W))))

            # Upsample CLIP features to intermediate resolution
            clip_resized = F.interpolate(
                clip_feats[i : i + 1], size=(obj_H, obj_W), mode="bilinear", align_corners=False
            )[0]  # (C, obj_H, obj_W)

            # Resize masks to intermediate resolution
            mask_i = masks[i]  # (M, H, W)
            mask_resized = (
                F.interpolate(mask_i.unsqueeze(1).float(), size=(obj_H, obj_W), mode="nearest").squeeze(1).bool()
            )  # (M, obj_H, obj_W)

            # Aggregate features within each mask
            agg_map = torch.zeros((C, obj_H, obj_W), dtype=torch.float32, device=self.device)
            agg_cnt = torch.zeros((obj_H, obj_W), dtype=torch.int32, device=self.device)

            for m_idx in range(mask_resized.shape[0]):
                region = mask_resized[m_idx]  # (obj_H, obj_W)
                if region.any():
                    feats = clip_resized[:, region]  # (C, N_pixels)
                    feats_mean = feats.mean(dim=1)  # (C,)
                    agg_map[:, region] += feats_mean[:, None]
                    agg_cnt[region] += 1

            # Normalize by count
            agg_map = agg_map / (agg_cnt.clamp_min(1)[None, :, :].float())

            # Resize back to original CLIP feature resolution
            agg_final = F.interpolate(agg_map.unsqueeze(0), size=(pH, pW), mode="bilinear", align_corners=False)[
                0
            ]  # (C, pH, pW)

            aggregated_list.append(agg_final)

        return torch.stack(aggregated_list, dim=0)  # (N, C, pH, pW)

    @torch.no_grad()
    def detect_objects_and_masks(
        self, rgb_images: np.ndarray, sam_size: int = 1024, conf: float = 0.4, iou: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect objects using YOLO and generate masks with MobileSAMv2.
        Based on reference code's detect_objects_and_masks function.

        Args:
            rgb_images: (N, H, W, 3) uint8 numpy array of RGB images
            sam_size: max resolution for SAM processing
            conf: confidence threshold for YOLO
            iou: IOU threshold for YOLO

        Returns:
            all_masks: List of (M_i, H_i, W_i) masks for each image (sorted by area)
            all_boxes: List of (M_i, 4) bounding boxes in xyxy format for each image
        """
        assert self.feature_type == "object", "Object detection requires feature_type='object'"
        assert self.mobilesamv2 is not None, "MobileSAMv2 not loaded"
        assert self.yolo_model is not None, "YOLO not loaded"

        all_masks = []
        all_boxes = []

        for img_idx in range(len(rgb_images)):
            rgb_proc = rgb_images[img_idx]  # (H, W, 3)

            # Detect objects with YOLO
            obj_results = self.yolo_model(
                rgb_proc, device=self.device, imgsz=sam_size, conf=conf, iou=iou, verbose=False
            )

            if (
                obj_results is None
                or len(obj_results) == 0
                or obj_results[0].boxes is None
                or len(obj_results[0].boxes) == 0
            ):
                print(f"WARNING!!! [CLIP3DLifter] No objects detected in image {img_idx}, skipping mask generation")
                # Return empty masks and boxes for this image
                H, W = rgb_proc.shape[:2]
                all_masks.append(torch.zeros((0, H, W), dtype=torch.float32, device=self.device))
                all_boxes.append(torch.zeros((0, 4), dtype=torch.float32, device=self.device))
                continue

            # Set image for SAM predictor
            self.sam_predictor.set_image(rgb_proc)
            input_boxes_xyxy = obj_results[0].boxes.xyxy  # (N, 4)
            boxes_np = input_boxes_xyxy.detach().cpu().numpy()
            boxes_np = self.sam_predictor.transform.apply_boxes(boxes_np, self.sam_predictor.original_size)
            input_boxes = torch.from_numpy(boxes_np).to(self.device)

            # Generate SAM masks in batches
            masks_list = []
            with torch.no_grad():
                img_embed = self.sam_predictor.features  # (1, C, H/4, W/4)
                dense_pe = self.mobilesamv2.prompt_encoder.get_dense_pe()

                for (boxes_chunk,) in batch_iterator(320, input_boxes):
                    # Repeat embeddings to match batch size
                    img_emb_rep = img_embed.repeat(boxes_chunk.shape[0], 1, 1, 1)
                    pe_rep = dense_pe.repeat(boxes_chunk.shape[0], 1, 1, 1)

                    sparse_emb, dense_emb = self.mobilesamv2.prompt_encoder(points=None, boxes=boxes_chunk, masks=None)
                    low_res_masks, _ = self.mobilesamv2.mask_decoder(
                        image_embeddings=img_emb_rep,
                        image_pe=pe_rep,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                        simple_type=True,
                    )

                    # Upsample to original size
                    low_res_masks = self.sam_predictor.model.postprocess_masks(
                        low_res_masks, self.sam_predictor.input_size, self.sam_predictor.original_size
                    )
                    masks_bin = (low_res_masks > self.mobilesamv2.mask_threshold).float()  # (B, 1, H, W)
                    masks_list.append(masks_bin.squeeze(1))  # (B, H, W)

            if len(masks_list) > 0:
                sam_mask = torch.cat(masks_list, dim=0)  # (M, H, W)
            else:
                sam_mask = torch.zeros(
                    (0, self.sam_predictor.original_size[0], self.sam_predictor.original_size[1]),
                    dtype=torch.float32,
                    device=self.device,
                )

            # Sort by area (descending)
            if sam_mask.shape[0] > 0:
                areas = torch.sum(sam_mask, dim=(1, 2))
                sorted_idx = torch.argsort(areas, descending=True)
                sam_mask = sam_mask[sorted_idx]
                input_boxes_xyxy = input_boxes_xyxy[sorted_idx]

            all_masks.append(sam_mask)
            all_boxes.append(input_boxes_xyxy)

        return all_masks, all_boxes

    @torch.no_grad()
    def lift_features_to_3d(
        self,
        point_map: torch.Tensor,
        clip_feats: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Lift 2D CLIP features to 3D using point map (GPU-native).

        Args:
            point_map: (H, W, 3) torch tensor of 3D coordinates
            clip_feats: (C, fH, fW) CLIP features
            images: (3, H, W) optional RGB image for colors
            valid_mask: (H, W) optional boolean mask

        Returns:
            points: (N, 3), features: (N, C), colors: (N, 3) or None,
            pixel_coords: (N, 2) [row, col]
        """
        C, fH, fW = clip_feats.shape
        H, W = point_map.shape[:2]
        device = point_map.device

        if (fH, fW) != (H, W):
            clip_feats_resized = F.interpolate(
                clip_feats.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)
        else:
            clip_feats_resized = clip_feats

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        pixel_coords = torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2)  # (H*W, 2) [row, col]
        points = point_map.reshape(-1, 3)
        features = clip_feats_resized.reshape(C, -1).T  # (H*W, C)

        colors = None
        if images is not None:
            assert images.shape[-2:] == (H, W)
            colors = images.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3) float [0,1]

        valid = torch.isfinite(points).all(dim=1)
        if valid_mask is not None:
            assert valid_mask.shape == (H, W)
            valid = valid & valid_mask.reshape(-1)

        points = points[valid]
        features = features[valid]
        if colors is not None:
            colors = colors[valid]
        pixel_coords = pixel_coords[valid]
        return points, features, colors, pixel_coords

    @torch.no_grad()
    def compute_2d_clip_features(
        self,
        images: torch.Tensor,
        sam_size: int = 1024,
        yolo_conf: float = 0.4,
        yolo_iou: float = 0.9,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract 2D CLIP features for all views (can run in parallel with VGGT).

        Args:
            images: (N, 3, H, W) torch tensor, values in [0, 1]

        Returns:
            clip_feats: (N, C, pH, pW), object_masks or None
        """
        object_masks = None
        if self.feature_type == "object":
            color_map_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            object_masks_list, _ = self.detect_objects_and_masks(
                rgb_images=color_map_np, sam_size=sam_size, conf=yolo_conf, iou=yolo_iou
            )
            max_masks = max(m.shape[0] for m in object_masks_list) if object_masks_list else 0
            if max_masks > 0:
                padded_masks = []
                for masks in object_masks_list:
                    if masks.shape[0] < max_masks:
                        mask_h, mask_w = masks.shape[1:]
                        padding = torch.zeros(
                            (max_masks - masks.shape[0], mask_h, mask_w), dtype=torch.float32, device=self.device
                        )
                        masks = torch.cat([masks, padding], dim=0)
                    padded_masks.append(masks)
                object_masks = torch.stack(padded_masks, dim=0)  # (N, M, H, W)
                print(f"[CLIP3D] Detected {max_masks} objects for mask aggregation")

        clip_feats = self.extract_clip_features(images, masks=object_masks)  # (N, C, pH, pW)
        return clip_feats, object_masks

    @torch.no_grad()
    def lift_clip_features_to_3d(
        self,
        clip_feats: torch.Tensor,
        point_map: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_colors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Lift pre-computed 2D CLIP features to 3D across all views.

        Args:
            clip_feats: (N, C, pH, pW) 2D features from compute_2d_clip_features
            point_map: (N, H, W, 3) GPU tensor of 3D coordinates
            images: (N, 3, H, W) optional; used for per-point colors
            valid_mask: (N, H, W) optional boolean mask
            return_colors: whether to return RGB colors

        Returns:
            points: (N_total, 3), features: (N_total, C),
            colors: (N_total, 3) or None, pixel_coords: (N_total, 3) [view, y, x]
        """
        N = clip_feats.shape[0]
        all_points, all_features, all_colors, all_pixel_coords = [], [], [], []

        for i in range(N):
            points, features, colors, pixel_coords = self.lift_features_to_3d(
                point_map[i],
                clip_feats[i],
                images=images[i] if (return_colors and images is not None) else None,
                valid_mask=valid_mask[i] if valid_mask is not None else None,
            )
            view_idx = torch.full((pixel_coords.shape[0], 1), i,
                                  device=pixel_coords.device, dtype=pixel_coords.dtype)
            all_points.append(points)
            all_features.append(features)
            if return_colors and colors is not None:
                all_colors.append(colors)
            all_pixel_coords.append(torch.cat([view_idx, pixel_coords], dim=1))

        merged_points = torch.cat(all_points, dim=0)
        merged_features = torch.cat(all_features, dim=0)
        merged_colors = torch.cat(all_colors, dim=0) if all_colors else None
        merged_pixel_coords = torch.cat(all_pixel_coords, dim=0)
        print(f"[CLIP3D] Extracted {merged_points.shape[0]} points with {merged_features.shape[1]}-dim features")
        return merged_points, merged_features, merged_colors, merged_pixel_coords

    @torch.no_grad()
    def compute_3d_clip_features(
        self,
        images: torch.Tensor,
        point_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_colors: bool = True,
        sam_size: int = 1024,
        yolo_conf: float = 0.4,
        yolo_iou: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """End-to-end: extract CLIP features and lift to 3D (convenience wrapper)."""
        N, _, H, W = images.shape
        assert point_map.shape == (N, H, W, 3), f"Point map shape mismatch: {point_map.shape}"
        clip_feats, _ = self.compute_2d_clip_features(
            images, sam_size=sam_size, yolo_conf=yolo_conf, yolo_iou=yolo_iou
        )
        return self.lift_clip_features_to_3d(
            clip_feats=clip_feats, point_map=point_map, images=images,
            valid_mask=valid_mask, return_colors=return_colors,
        )

    def expand_mask(self, pixel_coords, num_imgs, H, W, kernel=5) -> torch.Tensor:
        """Expand a mask given pixel coordinates using GPU dilation.

        pixel_coords: (N, 3) tensor [view, y, x]
        Returns: (num_imgs, H, W) bool tensor
        """
        if not isinstance(pixel_coords, torch.Tensor):
            pixel_coords = torch.from_numpy(pixel_coords).long()
        device = pixel_coords.device
        masks = torch.zeros((num_imgs, 1, H, W), device=device, dtype=torch.float32)
        view_idx = pixel_coords[:, 0].long()
        y_idx = pixel_coords[:, 1].long()
        x_idx = pixel_coords[:, 2].long()
        masks[view_idx, 0, y_idx, x_idx] = 1.0
        pad = kernel // 2
        masks_dilated = F.max_pool2d(masks, kernel_size=kernel, stride=1, padding=pad)
        return (masks_dilated > 0.5).squeeze(1)  # (num_imgs, H, W)

    @torch.no_grad()
    def encode_text_prompt(self, text: str) -> torch.Tensor:
        """Encode text using CLIP model. Returns (C,) normalized embedding on GPU."""
        tok = maskclip_onnx.clip.tokenize([text]).to(self.device)
        txt_feat = self.base_clip_model.encode_text(tok).float()
        return F.normalize(txt_feat, dim=-1).squeeze(0)

    @torch.no_grad()
    def compute_text_similarity(
        self,
        features: torch.Tensor | np.ndarray,
        positive_texts: List[str],
        negative_texts: Optional[List[str]] = None,
        softmax_temperature: float = 1.0,
        use_softmax: bool = False,
    ) -> torch.Tensor:
        """Compute cosine similarity between point features and text embeddings (GPU-native).

        Args:
            features: (N, C) point features (torch Tensor or numpy)
            positive_texts: list of positive text queries
            negative_texts: optional list of negative text queries

        Returns:
            similarities: (N,) float tensor on GPU
        """
        is_numpy = isinstance(features, np.ndarray)
        features_t = torch.from_numpy(features).to(self.device) if is_numpy else features

        features_norm = F.normalize(features_t, dim=1)

        pos_sims = [features_norm @ F.normalize(self.encode_text_prompt(t), dim=0)
                    for t in positive_texts]
        pos_sim = torch.stack(pos_sims).mean(dim=0) if pos_sims else torch.zeros(features_t.shape[0], device=self.device)

        if negative_texts:
            neg_sims = [features_norm @ F.normalize(self.encode_text_prompt(t), dim=0)
                        for t in negative_texts]
            final_sim = pos_sim - torch.stack(neg_sims).mean(dim=0)
        else:
            final_sim = pos_sim

        if use_softmax and (positive_texts or negative_texts):
            sims_with_bg = torch.stack([torch.zeros_like(final_sim), final_sim], dim=1)
            probs = F.softmax(sims_with_bg / softmax_temperature, dim=-1)
            final_sim = torch.nan_to_num(probs[:, 1], nan=1e-7)

        return final_sim


def resize_features_to_image(feat_chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize CLIP features from (C, Hf, Wf) to (C, H, W) using bilinear interpolation.

    Args:
        feat_chw: (C, Hf, Wf) numpy array
        target_h: target height
        target_w: target width

    Returns:
        (C, H, W) numpy array resized to target size
    """
    feat_tensor = torch.from_numpy(feat_chw).float().unsqueeze(0)  # (1, C, Hf, Wf)
    feat_resized = F.interpolate(feat_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return feat_resized.squeeze(0).numpy()  # (C, H, W)
