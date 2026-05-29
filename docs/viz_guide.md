# Viser Visualization Guide

Enable with `viz_viser: true` in the top-level config. The viewer is organized into top-level folders that map to the guidance field decomposition.

| Folder | Meaning | Color |
|--------|---------|-------|
| `reconstruction/` | Raw VGGT sensor output before guidance logic | Actual RGB |
| `neutral/` | Points removed from the collision field for robot-body or target-border reasons | Yellow |
| `repulsive/` | ESDF collision field points that push the robot away | Red |
| `attractive/` | Task targets and waypoint targets | Dark blue, cyan, magenta, or green |
| `robot/` | Robot geometry and probe points | Green, red, blue, purple |
| `grad/` | Per-denoising-step guidance gradients | Magenta, green, cyan, yellow |
| `traj/` | Unguided, guided, and best-of-N trajectories | Rainbow, cool, viridis |

## `reconstruction/`

| Name | Color | What it is |
|------|-------|------------|
| `reconstruction/raw` | Actual RGB | All points that pass depth confidence and z-threshold. Use this to check reconstruction quality before debugging guidance. |

## `neutral/`

Points removed from the collision field for robot-body or clearance reasons. These explain why nearby space is not repulsive.

| Name | Color | What it is |
|------|-------|------------|
| `neutral/arm` | Yellow | Points on the robot arm from the rendered URDF mask. |
| `neutral/arm_dilation` | Yellow | Clearance margin around the arm mask. If an obstacle disappears near the arm, check this first. |
| `neutral/ee_clearance_open` | Yellow | Points within `ee_clearance_open` of the EE tip when the gripper is open. |
| `neutral/ee_clearance_closed` | Yellow | Points within `ee_clearance_closed` of the EE tip when the gripper is closed, usually absorbing the in-hand object. |
| `neutral/ee_clearance_sphere` | Yellow sphere | Wireframe sphere showing the active EE clearance radius. |
| `neutral/target_border` | Yellow | Points within `target_proximity_threshold` of the confirmed target in 3D. These are excluded from repulsion but are not themselves attractive target points. |

If the robot gets too close to an obstacle near the arm, check whether `neutral/arm_dilation` is eating into that obstacle. If the robot collides near the gripper, check whether `neutral/ee_clearance_*` is removing too much of the scene.

## `repulsive/`

| Name | Color | What it is |
|------|-------|------------|
| `repulsive/collision_field` | Red | ESDF input: `reconstruction/raw` minus `neutral/*` minus confirmed task target points. These points actively repel the robot during guided denoising. |

If the robot collides with an object, first check whether that object appears in `repulsive/collision_field`. If it appears, tune SDF thresholds or cost weights. If it does not appear, check whether it was incorrectly removed by `neutral/*` or mistaken for a target.

## `attractive/`

CLIP target folders are used when semantic guidance comes from CLIP.

| Name | Color | What it is |
|------|-------|------------|
| `debug/clip/heatmap` | Blues colormap | Valid points colored by cosine similarity to the prompt. Bright blue means high similarity. |
| `attractive/clip/target` | Dark blue | CLIP-matched points above the configured threshold and outside EE clearance. |

SAM2 target folders are used when `vlm_pointer` is active.

| Name | Color | What it is |
|------|-------|------------|
| `attractive/sam2/target` | Dark blue | Points segmented by SAM2 on frame 0 using the VLM 2D point as prompt. Their centroid defines the VLM pointer target. |
| `attractive/sam2/cached` | Dark blue | Cached SAM2 segmentation reused on later frames. |

Waypoint folders are used when `waypoint` is active.

| Name | Color | What it is |
|------|-------|------------|
| `attractive/waypoint/all` | Magenta | Full set of 3D waypoints. |
| `attractive/waypoint/current` | Cyan | Next waypoint for the robot to reach. |
| `attractive/waypoint/matched_lines` | Green lines | Lines from predicted EE positions to sequentially matched waypoints. |
| `attractive/freeze_zone_target` | Bright green | Points used for freeze-zone distance checks. Guidance turns off after the EE enters this cloud. |

## `robot/`

| Name | Color | What it is |
|------|-------|------------|
| `robot/ee` | Green | End-effector tip position after applying `tip_coord`. |
| `robot/joints` | Red | Joint positions from forward kinematics. |
| `robot/wrist_probe` | Blue | Probe points used for wrist-body SDF collision costs. |
| `robot/wrist_cam` | Purple axes | Wrist camera coordinate frame. |

## `grad/`

Each arrow shows the guidance cost gradient at the predicted EE position for one denoising timestep. Arrows are scaled for visibility.

| Name | Color | What it is |
|------|-------|------------|
| `grad/collision/{t}` | Magenta | Collision repulsion gradient from the SDF log barrier. |
| `grad/waypoint/{t}` | Green | Waypoint attraction gradient. |
| `grad/vlm/{t}` | Cyan | VLM pointer attraction gradient toward the SAM2 target centroid. |
| `grad/total/{t}` | Yellow | Combined guidance vector applied to the denoising step. |

If `grad/total` points away from the target, collision avoidance is dominating. If it points toward the target but the robot still misses, check `guidance.factor`, `guidance.num_guided`, and whether the target cloud is correct.

## `traj/`

| Name | Colormap | What it is |
|------|----------|------------|
| `traj/unguided/{t}` | HSV | EE frames from the unguided denoising pass. |
| `traj/guided/{t}` | Cool | EE frames from the guided denoising pass. |
| `traj/pos/{t}` | Cool | EE tip positions at each guided denoising timestep. |
| `traj/init/{candidate}` | Viridis | EE positions for best-of-N initial noise candidates from `guidance.num_init_candidates`. |

## Debug Workflows

Why is the robot colliding?

1. Check `repulsive/collision_field`: is the obstacle present?
2. Check `grad/collision/{t}` near the obstacle: are arrows pointing away from it?
3. Check `robot/wrist_probe`: is the wrist probe inside the obstacle SDF range?
4. Check `neutral/ee_clearance_*`: is the object being swallowed by EE clearance?

Why is the robot not going to the target?

1. Check `attractive/clip/target` or `attractive/sam2/target`: was the target identified correctly?
2. Check `debug/clip/heatmap`: does CLIP peak on the right object?
3. Check `grad/waypoint/{t}` or `grad/vlm/{t}`: is the attraction gradient pointing toward the target?
4. Check `grad/total/{t}`: is collision avoidance overpowering attraction?

Why did guidance turn off?

1. Check `attractive/freeze_zone_target`.
2. The freeze zone activates when `robot/ee` is within `enter_threshold` of this cloud.

Is reconstruction good?

1. Compare `reconstruction/raw` to the actual scene.
2. Check `neutral/arm` and `neutral/arm_dilation` for mask quality.
3. Confirm `repulsive/collision_field` does not include the target object.
