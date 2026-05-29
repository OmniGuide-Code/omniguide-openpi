import os
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
import pybullet as p
import pybullet_data as pd
import numpy as np
import GPUtil as GPU
from scipy.spatial.transform import Rotation as rot
import torch

from vis.robot_config import GRIPPER_LOOKUP_TABLE, GRIPPER_LOOKUP_TABLE_TORCH


def action2joint(action):
    """Convert a normalized gripper action to 8 joint angles.

    Args:
        action: (N,) or (N, 1) gripper values in [0, 1].

    Returns:
        (N, 8) joint angles.
    """
    if len(action.shape) == 2:
        action = action[:, 0]
    if isinstance(action, np.ndarray):
        action_index = np.clip(action * 6, 0, 5).astype(int)
        return GRIPPER_LOOKUP_TABLE[action_index]
    else:
        action_index = torch.clamp(action * 6, 0, 5).long()
        return GRIPPER_LOOKUP_TABLE_TORCH[action_index].to(action.device)


def cvK2BulletP(K, w, h, near, far):
	"""
	cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
	and ROS to the projection matrix used in openGL and Pybullet.

	:param K:  OpenCV 3x3 camera intrinsic matrix
	:param w:  Image width
	:param h:  Image height
	:near:     The nearest objects to be included in the render
	:far:      The furthest objects to be included in the render
	:return:   4x4 projection matrix as used in openGL and pybullet
	""" 
	f_x = K[0,0]
	f_y = K[1,1]
	c_x = K[0,2]
	c_y = K[1,2]
	A = (near + far)/(near - far)
	B = 2 * near * far / (near - far)

	projection_matrix = [
											[2/w * f_x,  0,          (w - 2*c_x)/w,  0],
											[0,          2/h * f_y,  (2*c_y - h)/h,  0],
											[0,          0,          A,              B],
											[0,          0,          -1,             0]]
	#The transpose is needed for respecting the array structure of the OpenGL
	return np.array(projection_matrix).T.reshape(16).tolist()

def cvExtrinMatW2C2XYZEulerC2W(T):
    R = T[:3, :3].T
    t = - R @ T[:3, 3]
    r = rot.from_matrix(R)
    euler = r.as_euler('xyz')
    xyz_euler = np.hstack([t, euler])
    return xyz_euler

def cvPose2BulletView(xyz_euler):
    t = xyz_euler[:3]
    euler = xyz_euler[3:]
    r = rot.from_euler('xyz', euler)
    r = r.as_matrix()

    R = r

    T = np.vstack([np.hstack([R, np.array(t).reshape(3,1)]),
                                                        np.array([0, 0, 0, 1])])
    # Convert opencv convention to python convention
    # By a 180 degrees rotation along X
    Tc = np.array([[1,   0,    0,  0],
                                    [0,  -1,    0,  0],
                                    [0,   0,   -1,  0],
                                    [0,   0,    0,  1]]).reshape(4,4)

    # pybullet pse is the inverse of the pose from the ROS-TF
    T=Tc@np.linalg.inv(T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = T.T.reshape(16)

    viewMatrix = tuple(viewMatrix.T.flatten().tolist())
    return viewMatrix


class PyBulletRender():
    def __init__(self, urdf_dir, save_dir=None, resolution=(720, 1280)):
        self.model_urdf_dir = urdf_dir
        self.save_dir = save_dir
        self.resolution = resolution
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        enableGPU = False
        GPU.showUtilization()
        numGPUs = len(GPU.getGPUs())
        print("numGPUs=",numGPUs)
        if numGPUs > 0:
            enableGPU = True
        eglPluginId = -1
        if enableGPU:
            import pkgutil
            egl = pkgutil.get_loader('eglRenderer')
            if (egl):
                eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                eglPluginId = p.loadPlugin("eglRendererPlugin")

        if eglPluginId>=0:
            print("Using GPU hardware (eglRenderer)")
        else:
            print("using CPU renderer (TinyRenderer)")
        p.resetSimulation()

        p.configureDebugVisualizer(p.COV_ENABLE_GUI)
        p.configureDebugVisualizer(rgbBackground=[0.0, 0.0, 0.0])
        self.arm = p.loadURDF(self.model_urdf_dir, [0.0, 0.0, 0.0], useFixedBase=True)
        
        self.num_joints = p.getNumJoints(self.arm)  
        print("num_joints=",self.num_joints)
        # for i in range(self.num_joints):
        #     print("joint[",i,"]=",p.getJointInfo(self.arm, i))
    
    def set_camera(self, xyz_euler, K):
        self.viewMatrix = cvPose2BulletView(xyz_euler)
        self.projectionMatrix = cvK2BulletP(K, self.resolution[1], self.resolution[0], 0.1, 1000.0)

    def render(self, joint_positions, gripper_position):
        """
        joint_positions: Joint positions of the arm, 7
        gripper_position: Gripper position, 1
        """
        for i in range(7):
            p.resetJointState(self.arm, i, joint_positions[i])
        gripper_joints = action2joint(gripper_position)
        gripper_joints = gripper_joints[0, [0, 1, 2, 3, 6, 7, 4, 5]]
        gripper_joints[4:] = gripper_joints[4:] * -1.0
        
        for i in range(8):
            p.resetJointState(self.arm, i+8, gripper_joints[i])
        img_arr = p.getCameraImage(self.resolution[1], self.resolution[0], self.viewMatrix, self.projectionMatrix)
        rgb = img_arr[2]
        depth = img_arr[3]
        np_img_arr = np.reshape(rgb, (self.resolution[0], self.resolution[1], 4))
        # convert non-linear to linear depth
        depth = np.reshape(depth, (self.resolution[0], self.resolution[1]))
        depth = 0.1 * 1000 / (1000 - (1000 - 0.1)*depth)

        mask = img_arr[4] >= 0
        mask = np.reshape(mask, (self.resolution[0], self.resolution[1]))

        np_img_arr[~mask, :3] = 127
        depth = np.reshape(depth, (self.resolution[0], self.resolution[1]))
        return np_img_arr, depth, mask


