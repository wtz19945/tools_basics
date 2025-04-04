import os

import numpy as np
import mujoco
import mujoco.viewer

import time

import osqp
import scipy.sparse as sparse
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as Rlib

"""
    In this file we will learn how to locate a mujoco xml file
    and get its file path using the os module from the standard library.

    We will also learn how to load the xml file and simulate the system.
"""
def rotm_2_euler(R):
    """ This function computes the x-y-z euler angles of a rotation matrix """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6  # Check for singularity
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0  # Set yaw to zero (or any arbitrary value)

    return np.array([roll, pitch, yaw])

def quat_2_euler(quat):
    """ This function computes the x-y-z euler angles of a quaterion matrix """
    r = Rlib.from_quat(quat)  # Convert quaternion to rotation object
    return r.as_euler('xyz', degrees=False)  # Convert to XYZ Euler angles

def initialize_configuration(data):
    # Define the default configuration as a dictionary
    default_configuration = {
        "left-hip-roll": .31,
        "left-hip-yaw": 0.0,
        "left-hip-pitch": 0.3,
        "left-knee": 0.19,
        "left-shin": 0.0,
        "left-tarsus": 0.0,
        "left-toe-pitch": -0.026,
        "left-toe-roll": -0.12,
        "left-shoulder-roll": -0.15,
        "left-shoulder-pitch": 1.1,
        "left-shoulder-yaw": 0.0,
        "left-elbow": -0.145,
        "right-hip-roll": -0.31,
        "right-hip-yaw": 0.0,
        "right-hip-pitch": -0.3,
        "right-knee": -0.19,
        "right-shin": 0.0,
        "right-tarsus": 0.0,
        "right-toe-pitch": 0.026,
        "right-toe-roll": 0.12,
        "right-shoulder-roll": 0.15,
        "right-shoulder-pitch": -1.1,
        "right-shoulder-yaw": 0.0,
        "right-elbow": 0.145
    }

    default_Ctrl_configuration = {
        "left-hip-roll": .31,
        "left-hip-yaw": 0.0,
        "left-hip-pitch": 0.3,
        "left-knee": 0.19,
        "left-tarsus": 0.0,
        "left-toe-pitch": -0.026,
        "left-toe-roll": -0.12,
        "left-shoulder-roll": -0.15,
        "left-shoulder-pitch": 1.1,
        "left-shoulder-yaw": 0.0,
        "left-elbow": -0.145,
        "right-hip-roll": -0.31,
        "right-hip-yaw": 0.0,
        "right-hip-pitch": -0.3,
        "right-knee": -0.19,
        "right-tarsus": 0.0,
        "right-toe-pitch": 0.026,
        "right-toe-roll": 0.12,
        "right-shoulder-roll": 0.15,
        "right-shoulder-pitch": -1.1,
        "right-shoulder-yaw": 0.0,
        "right-elbow": 0.145
    }

    # Iterate over the default configuration and set the corresponding joint position
    for joint, position in default_configuration.items():
        data.joint(joint).qpos[0] = position
    
    for i, (joint, value) in enumerate(default_Ctrl_configuration.items()):
        # print(f"Index: {i}, Joint: {joint}, Value: {value}")
        data.ctrl[i] = value
    



def main(argv=None):
    # xml_path = os.path.expanduser('~/Tianze/mujoco_work/mujoco_menagerie/unitree_g1/scene.xml')
    # Load the XML file
    filepath = os.path.join(os.path.dirname(__file__), 'digit/scene.xml')
    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)

    # Set timestep
    model.opt.timestep = 0.002

    # Reset to keyframe
    initialize_configuration(data)
    mujoco.mj_forward(model, data)

    # Get control limits
    print(model.nv)
    print(model.nu)
    # Start visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        while viewer.is_running():
            # Compute Inertia Matrix
   
            mujoco.mj_step(model, data)

            # Sync Viewer
            viewer.sync()
            time.sleep(0.002)  # Sleep to match timestep


if __name__ == '__main__':
    main()
