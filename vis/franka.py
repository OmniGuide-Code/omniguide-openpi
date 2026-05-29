import torch
import numpy as np

from vis.robot_config import GRIPPER_LOOKUP_TABLE_TORCH


def action2joint(action: torch.Tensor) -> torch.Tensor:
    """Convert a normalized gripper action to 8 joint angles.

    Args:
        action: (N,) or (N, 1) gripper values in [0, 1].

    Returns:
        (N, 8) joint angles.
    """
    if len(action.shape) == 2:
        action = action[:, 0]
    action_index = torch.clamp(action * 6, 0, 5).long()
    output = GRIPPER_LOOKUP_TABLE_TORCH[action_index]
    if hasattr(action, 'device'):
        output = output.to(action.device)
    return output


class FK():

    def __init__(self):
        self.joints = torch.tensor([
            [ 0,      0,      0],
            [ 0,      0.192,  0],
            [ 0,      0,      0],
            [-0.0825, 0.121,  0],
            [-0.0825, 0,      0],
            [ 0,      0.259,  0],
            [-0.088, -0.015,  0],
            [ 0,      0,     -0.159],
        ], dtype=torch.float32)
        self.joints = torch.cat((self.joints, torch.ones((8, 1))), dim=1)

    def forward(self, q):
        """
        Args:
            q: (7,) or (1, 7) joint angles [q0..q6].

        Returns:
            joint_positions: (8, 3) world-frame positions of each joint / EE.
            T0e: (4, 4) end-effector transform in world frame.
        """
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, dtype=torch.float32)

        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7 = q
        theta_4 = -theta_4

        link_transforms = torch.zeros((7, 4, 4), device=q.device, dtype=q.dtype)

        c1, s1 = torch.cos(theta_1), torch.sin(theta_1)
        c2, s2 = torch.cos(theta_2), torch.sin(theta_2)
        c3, s3 = torch.cos(theta_3), torch.sin(theta_3)
        t4_mod = theta_4 - torch.pi
        c4, s4 = torch.cos(t4_mod), torch.sin(t4_mod)
        c5, s5 = torch.cos(theta_5), torch.sin(theta_5)
        t6_mod = theta_6 - torch.pi
        c6, s6 = torch.cos(t6_mod), torch.sin(t6_mod)
        t7_mod = theta_7 - torch.pi / 4
        c7, s7 = torch.cos(t7_mod), torch.sin(t7_mod)

        link_transforms[0, 0, 0] = c1;  link_transforms[0, 0, 2] = -s1
        link_transforms[0, 1, 0] = s1;  link_transforms[0, 1, 2] = c1
        link_transforms[0, 2, 1] = -1.0; link_transforms[0, 2, 3] = 0.333
        link_transforms[0, 3, 3] = 1.0

        link_transforms[1, 0, 0] = c2;  link_transforms[1, 0, 2] = s2
        link_transforms[1, 1, 0] = s2;  link_transforms[1, 1, 2] = -c2
        link_transforms[1, 2, 1] = 1.0; link_transforms[1, 3, 3] = 1.0

        link_transforms[2, 0, 0] = c3;  link_transforms[2, 0, 2] = -s3; link_transforms[2, 0, 3] = 0.0825 * c3
        link_transforms[2, 1, 0] = s3;  link_transforms[2, 1, 2] = c3;  link_transforms[2, 1, 3] = 0.0825 * s3
        link_transforms[2, 2, 1] = -1.0; link_transforms[2, 2, 3] = 0.316
        link_transforms[2, 3, 3] = 1.0

        link_transforms[3, 0, 0] = c4;  link_transforms[3, 0, 2] = -s4; link_transforms[3, 0, 3] = 0.0825 * c4
        link_transforms[3, 1, 0] = s4;  link_transforms[3, 1, 2] = c4;  link_transforms[3, 1, 3] = 0.0825 * s4
        link_transforms[3, 2, 1] = -1.0; link_transforms[3, 3, 3] = 1.0

        link_transforms[4, 0, 0] = c5;  link_transforms[4, 0, 2] = -s5
        link_transforms[4, 1, 0] = s5;  link_transforms[4, 1, 2] = c5
        link_transforms[4, 2, 1] = -1.0; link_transforms[4, 2, 3] = 0.384
        link_transforms[4, 3, 3] = 1.0

        link_transforms[5, 0, 0] = c6;  link_transforms[5, 0, 2] = s6;  link_transforms[5, 0, 3] = 0.088 * c6
        link_transforms[5, 1, 0] = s6;  link_transforms[5, 1, 2] = -c6; link_transforms[5, 1, 3] = 0.088 * s6
        link_transforms[5, 2, 1] = 1.0; link_transforms[5, 3, 3] = 1.0

        link_transforms[6, 0, 0] = c7;  link_transforms[6, 0, 1] = -s7
        link_transforms[6, 1, 0] = s7;  link_transforms[6, 1, 1] = c7
        link_transforms[6, 2, 2] = 1.0; link_transforms[6, 2, 3] = 0.21
        link_transforms[6, 3, 3] = 1.0

        transforms = [torch.eye(4, device=q.device, dtype=q.dtype)]
        for i in range(7):
            transforms.append(torch.matmul(transforms[-1], link_transforms[i]))

        T0e = transforms[7]

        positions = []
        for i in range(7):
            positions.append(torch.matmul(transforms[i + 1], self.joints[i + 1]))
        positions.append(T0e[:, 3])

        joint_positions = torch.stack(positions)[:, :3]
        return joint_positions, T0e


class FK_batch():

    def __init__(self, tip_offset: float = 0.0):
        self.joints = torch.tensor([
            [ 0,      0,      0],
            [ 0,      0.192,  0],
            [ 0,      0,      0],
            [-0.0825, 0.121,  0],
            [-0.0825, 0,      0],
            [ 0,      0.259,  0],
            [-0.088, -0.015,  0],
            [ 0,      0,     -0.159],
        ], dtype=torch.float32)
        self.joints = torch.cat((self.joints, torch.ones((8, 1))), dim=1)
        self.tip_offset = tip_offset

    def forward(self, q, all_trans: bool = False):
        """
        Args:
            q: (N, 7) batch of joint angles.
            all_trans: If True, also return the list of intermediate transforms.

        Returns:
            joint_positions: (N, 8, 3) world-frame positions.
            T0e or transforms: (N, 4, 4) EE transform, or list of transforms.
        """
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, dtype=torch.float32)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        N = q.shape[0]
        self.joints = self.joints.to(q.device)

        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7 = q.unbind(dim=1)
        theta_4 = -theta_4

        link_transforms = torch.zeros((N, 7, 4, 4), device=q.device, dtype=q.dtype)

        c1, s1 = torch.cos(theta_1), torch.sin(theta_1)
        c2, s2 = torch.cos(theta_2), torch.sin(theta_2)
        c3, s3 = torch.cos(theta_3), torch.sin(theta_3)
        t4_mod = theta_4 - torch.pi
        c4, s4 = torch.cos(t4_mod), torch.sin(t4_mod)
        c5, s5 = torch.cos(theta_5), torch.sin(theta_5)
        t6_mod = theta_6 - torch.pi
        c6, s6 = torch.cos(t6_mod), torch.sin(t6_mod)
        t7_mod = theta_7 - torch.pi / 4
        c7, s7 = torch.cos(t7_mod), torch.sin(t7_mod)

        link_transforms[:, 0, 0, 0] = c1;  link_transforms[:, 0, 0, 2] = -s1
        link_transforms[:, 0, 1, 0] = s1;  link_transforms[:, 0, 1, 2] = c1
        link_transforms[:, 0, 2, 1] = -1.0; link_transforms[:, 0, 2, 3] = 0.333
        link_transforms[:, 0, 3, 3] = 1.0

        link_transforms[:, 1, 0, 0] = c2;  link_transforms[:, 1, 0, 2] = s2
        link_transforms[:, 1, 1, 0] = s2;  link_transforms[:, 1, 1, 2] = -c2
        link_transforms[:, 1, 2, 1] = 1.0; link_transforms[:, 1, 3, 3] = 1.0

        link_transforms[:, 2, 0, 0] = c3;  link_transforms[:, 2, 0, 2] = -s3; link_transforms[:, 2, 0, 3] = 0.0825 * c3
        link_transforms[:, 2, 1, 0] = s3;  link_transforms[:, 2, 1, 2] = c3;  link_transforms[:, 2, 1, 3] = 0.0825 * s3
        link_transforms[:, 2, 2, 1] = -1.0; link_transforms[:, 2, 2, 3] = 0.316
        link_transforms[:, 2, 3, 3] = 1.0

        link_transforms[:, 3, 0, 0] = c4;  link_transforms[:, 3, 0, 2] = -s4; link_transforms[:, 3, 0, 3] = 0.0825 * c4
        link_transforms[:, 3, 1, 0] = s4;  link_transforms[:, 3, 1, 2] = c4;  link_transforms[:, 3, 1, 3] = 0.0825 * s4
        link_transforms[:, 3, 2, 1] = -1.0; link_transforms[:, 3, 3, 3] = 1.0

        link_transforms[:, 4, 0, 0] = c5;  link_transforms[:, 4, 0, 2] = -s5
        link_transforms[:, 4, 1, 0] = s5;  link_transforms[:, 4, 1, 2] = c5
        link_transforms[:, 4, 2, 1] = -1.0; link_transforms[:, 4, 2, 3] = 0.384
        link_transforms[:, 4, 3, 3] = 1.0

        link_transforms[:, 5, 0, 0] = c6;  link_transforms[:, 5, 0, 2] = s6;  link_transforms[:, 5, 0, 3] = 0.088 * c6
        link_transforms[:, 5, 1, 0] = s6;  link_transforms[:, 5, 1, 2] = -c6; link_transforms[:, 5, 1, 3] = 0.088 * s6
        link_transforms[:, 5, 2, 1] = 1.0; link_transforms[:, 5, 3, 3] = 1.0

        link_transforms[:, 6, 0, 0] = c7;  link_transforms[:, 6, 0, 1] = -s7
        link_transforms[:, 6, 1, 0] = s7;  link_transforms[:, 6, 1, 1] = c7
        link_transforms[:, 6, 2, 2] = 1.0; link_transforms[:, 6, 2, 3] = 0.21 + self.tip_offset
        link_transforms[:, 6, 3, 3] = 1.0

        T = torch.eye(4, device=q.device, dtype=q.dtype).expand(N, 4, 4)
        transforms = [T]
        for i in range(7):
            T = T @ link_transforms[:, i]
            transforms.append(T)

        T0e = transforms[-1]

        T_stack = torch.stack(transforms[1:8], dim=1)          # (N, 7, 4, 4)
        j_stack = self.joints[1:8, :].unsqueeze(0).unsqueeze(-1)  # (1, 7, 4, 1)
        positions_7 = (T_stack @ j_stack).squeeze(-1)          # (N, 7, 4)
        ee_pos = T0e[:, :, 3].unsqueeze(1)                     # (N, 1, 4)
        joint_positions = torch.cat([positions_7, ee_pos], dim=1)[:, :, :3]  # (N, 8, 3)

        if all_trans:
            return joint_positions, transforms
        return joint_positions, T0e
