"""Shared Franka Panda robot configuration constants."""

import numpy as np
import torch

# Maps gripper action value (bucketed to 6 levels) to 8 joint angles.
# Each row corresponds to one gripper state level.
GRIPPER_LOOKUP_TABLE = np.array([
    [0.00292, 0.00269, 0.00717, -0.0128, 0.00292, 0.00269, 0.00717, -0.0128],
    [0.163,   0.00228, 0.166,   -0.163,  0.163,   0.00228, 0.166,   -0.163 ],
    [0.323,   0.0019,  0.325,   -0.314,  0.323,   0.0019,  0.325,   -0.314 ],
    [0.482,   0.00155, 0.484,   -0.466,  0.482,   0.00155, 0.484,   -0.466 ],
    [0.642,   0.00123, 0.644,   -0.619,  0.642,   0.00123, 0.644,   -0.619 ],
    [0.797,   0.00456, 0.795,   -0.791,  0.797,   0.00456, 0.795,   -0.791 ],
])

GRIPPER_LOOKUP_TABLE_TORCH = torch.tensor(GRIPPER_LOOKUP_TABLE, dtype=torch.float32)
