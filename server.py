import dataclasses
import datetime
import os
import random
from dataclasses import dataclass, field
from typing import Union, List

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config
from utils import save_json


# ---------------------------------------------------------------------------
# Nested config dataclasses — each corresponds to a YAML block.
# ---------------------------------------------------------------------------

@dataclass
class GuidanceParams:
    """Core denoising guidance hyperparameters."""
    factor: float = 0.05                 # Guidance strength multiplier
    num_init_candidates: int = 0         # Best-of-N initial noises to score before denoising
    num_guided: int = 10                 # Action-sequence steps to guide per denoising call
    max_guide_length: float = 1.0        # Gradient magnitude clamp (meters)
    stop_grad: bool = False              # If True, detach x_t before the denoiser (faster but less accurate DPS)


@dataclass
class FreezeZoneParams:
    """Suppress guidance when the EE is near the target (let the policy take over)."""
    enabled: bool = True
    # Enter freeze zone when EE is within this distance of the target (m).
    # Can be a list for per-subtask thresholds when subtask_monitor is enabled.
    threshold: Union[float, List[float]] = 0.1
    # Resume guidance once EE moves farther than this distance from the target (m).
    exit_threshold: Union[float, List[float]] = None
    # Gripper joint state below this value is treated as grasping.
    gripper_threshold: float = 0.3


@dataclass
class SubtaskMonitorParams:
    """VLM-based progress tracking across task subtasks."""
    enabled: bool = False
    # Maps each guidance type to the subtask index during which it is active.
    # Guidance fires only when current_subtask == value; zero cost otherwise.
    # Example: {vlm_pointer: 0}  — vlm_pointer is active only during subtask 0.
    active_subtask: dict = field(default_factory=dict)


@dataclass
class ClipParams:
    """CLIP-based point-cloud relevance filtering (only used with collision guidance)."""
    percentile: float = 95      # Keep points above this similarity percentile
    min_similarity: float = None  # Absolute minimum: max(percentile_thresh, min_sim * max_sim)
    use_subtask_as_query: bool = False  # If True, query CLIP with the current subtask string
                                        # instead of the full task prompt.


@dataclass
class PointCloudParams:
    """Camera views used for point-cloud and CLIP feature construction."""
    camera_views: list[str] = field(default_factory=lambda: ["left", "right", "overhead"])


@dataclass
class CollisionParams:
    """SDF cost function weights and proximity-based filtering thresholds."""
    # Log-barrier cost: applied when the EE/wrist SDF value falls below threshold.
    ee_sdf_threshold: float = 0.2      # End-effector threshold (m)
    ee_cost_weight: float = 0.1        # Weight on EE log-cost term
    wrist_sdf_threshold: float = 0.1   # Wrist threshold (m)
    wrist_cost_weight: float = 0.01    # Weight on wrist log-cost term

    # Proximity filter: remove point-cloud points near the end-effector tip.
    # Prevents the robot from being repelled from what it's about to touch or hold.
    ee_clearance_open: float = 0.05    # (m) gripper open — clearance around approaching EE
    ee_clearance_closed: float = 0.1   # (m) gripper closed — larger to absorb in-hand object
    gripper_closed_threshold: float = 0.05  # Joint state below this = grasping

    # Target proximity: neutralise background points within this distance of the confirmed
    # target object (SAM2 segmentation or CLIP matches).  These points are near the target
    # but not confirmed as part of it — excluded from repulsion but not used for attraction.
    target_proximity_threshold: float = 0.05


@dataclass
class VlmParams:
    """External VLM and progress-tracker service endpoints."""
    pointer_url: str = ""              # URL of the VLM pointer service
    progress_tracker_url: str = ""     # URL of the subtask progress tracker
    camera_view: str = "right"         # Which camera image to send to the VLM
    use_sam2: bool = True              # True: run SAM2 to segment the full target object.
                                       # False: use the single VLM pixel lifted to 3D.


@dataclass
class WaypointParams:
    """Pre-recorded 3D waypoint trajectory configuration."""
    path: str = ""                # Path to .npy file with shape (N, 3)
    threshold_distance: float = 0.01  # EE considered "at waypoint" within this distance (m)
    camera_view: str = "right"    # Camera frame the trajectory was recorded in


@dataclass
class GuidanceConfig:
    """Top-level guidance configuration.

    guidance_types controls which guidance signals are active.
    Supported values: "collision", "waypoint", "vlm_pointer".
    """
    guidance_types: list = field(default_factory=lambda: ["collision"])
    guidance: GuidanceParams = field(default_factory=GuidanceParams)
    freeze_zone: FreezeZoneParams = field(default_factory=FreezeZoneParams)
    subtask_monitor: SubtaskMonitorParams = field(default_factory=SubtaskMonitorParams)
    clip: ClipParams = field(default_factory=ClipParams)
    point_cloud: PointCloudParams = field(default_factory=PointCloudParams)
    collision_params: CollisionParams = field(default_factory=CollisionParams)
    vlm: VlmParams = field(default_factory=VlmParams)
    waypoint: WaypointParams = field(default_factory=WaypointParams)
    viz_viser: bool = True  # Enable/disable all viser visualization (disable for speed)
    save_debug_images: bool = True  # Save intermediate debug images to debug/images/.
    debug_pause_before_action_return: bool = True  # Wait for Enter before returning each sampled action chunk.


# ---------------------------------------------------------------------------
# Config construction from Hydra / OmegaConf
# ---------------------------------------------------------------------------

def _seed_everything(seed: int | None) -> None:
    if seed is None:
        print("[SEED] No seed configured; policy server randomness is uncontrolled.")
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    print(f"[SEED] Policy server seeded with seed={seed}.")

def _build_dataclass(cls, d: dict):
    """Build a dataclass from a dict, failing on unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - known)
    assert not unknown, f"Unknown keys for {cls.__name__}: {unknown}"
    return cls(**d)


def _build_guidance_config(cfg_dict: dict, top_cfg) -> GuidanceConfig:
    """Build GuidanceConfig from a resolved Hydra config dict.

    OmegaConf already coerces scalar types (int/float/bool) correctly, so we only
    need to handle two special cases:
      1. guidance_types may come as a string instead of a list.
      2. gripper_threshold may live at top-level cfg (legacy field) — fall back to it
         when freeze_zone.gripper_threshold is not set in the YAML.
    """
    d = dict(cfg_dict)

    gt = d.get("guidance_types", ["collision"])
    guidance_types = gt if isinstance(gt, list) else [gt]

    freeze_zone_d = dict(d.get("freeze_zone", {}) or {})
    if "gripper_threshold" not in freeze_zone_d and "gripper_threshold" in top_cfg:
        freeze_zone_d["gripper_threshold"] = top_cfg["gripper_threshold"]

    # Debug/runtime controls live at top-level config (not inside guidance_config YAML block).
    viz_viser = bool(top_cfg.get("viz_viser", True))
    save_debug_images = bool(top_cfg.get("save_debug_images", True))
    debug_pause_before_action_return = bool(top_cfg.get("debug_pause_before_action_return", True))

    return GuidanceConfig(
        guidance_types=guidance_types,
        guidance=_build_dataclass(GuidanceParams, dict(d.get("guidance", {}) or {})),
        freeze_zone=_build_dataclass(FreezeZoneParams, freeze_zone_d),
        subtask_monitor=_build_dataclass(SubtaskMonitorParams, dict(d.get("subtask_monitor", {}) or {})),
        clip=_build_dataclass(ClipParams, dict(d.get("clip", {}) or {})),
        point_cloud=_build_dataclass(PointCloudParams, dict(d.get("point_cloud", {}) or {})),
        collision_params=_build_dataclass(CollisionParams, dict(d.get("collision_params", {}) or {})),
        vlm=_build_dataclass(VlmParams, dict(d.get("vlm", {}) or {})),
        waypoint=_build_dataclass(WaypointParams, dict(d.get("waypoint", {}) or {})),
        viz_viser=viz_viser,
        save_debug_images=save_debug_images,
        debug_pause_before_action_return=debug_pause_before_action_return,
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    _seed_everything(cfg.seed)

    training_config = _config.get_config("pi05_droid" if cfg.real_robot else "pi05_droid_jointpos")

    mode_str = "JOINT VELOCITY (FOR REAL ROBOT ONLY)" if cfg.real_robot else "JOINT POSITION (FOR SIMULATION ONLY)"
    print(f"\nPolicy Mode: {mode_str}")

    checkpoint_dir = cfg.checkpoint_dir

    save_state_dir = None
    if cfg.save_states_base_dir:
        date = datetime.datetime.now(tz=datetime.UTC).strftime("%m%d")
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d_%H:%M:%S")
        save_state_dir = f"{cfg.save_states_base_dir}/{date}/{timestamp}"
        os.makedirs(save_state_dir, exist_ok=True)
        os.makedirs(f"{save_state_dir}/observations", exist_ok=True)
        os.makedirs(f"{save_state_dir}/actions", exist_ok=True)
        os.makedirs(f"{save_state_dir}/visualizations", exist_ok=True)

    print("Using checkpoint dir:", checkpoint_dir)

    guidance_cfg_dict = OmegaConf.to_container(cfg.guidance_config, resolve=True)
    guidance_config = _build_guidance_config(guidance_cfg_dict, cfg)

    if save_state_dir:
        save_json(dataclasses.asdict(guidance_config), f"{save_state_dir}/guidance_config.json")

    policy = _policy_config.create_trained_policy(
        training_config,
        checkpoint_dir,
        save_state_dir=save_state_dir,
        guidance_config=guidance_config,
    )
    _seed_everything(cfg.seed)

    server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
