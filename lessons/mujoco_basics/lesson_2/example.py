import os

import numpy as np
import mujoco
import mujoco.viewer

import time

import osqp
import scipy.sparse as sparse
from scipy.linalg import block_diag

# Code for OSC without Quadratic Programming

def main(argv=None):
    filepath = os.path.join(os.path.dirname(__file__), 'arm.xml')

    # Load the xml file
    model = mujoco.MjModel.from_xml_path(filepath)
    print('Model loaded successfully!')

    # Update timestep:
    model.opt.timestep = 0.002

    # Create a data object#
    data = mujoco.MjData(model)
    print('Data loaded successfully!')

    # Update the initial state
    data.qpos = np.array([0.0, 0.0, -np.pi / 2])
    data.qvel = np.array([0.0, 0.0, 0.0])
    data.ctrl = np.array([0.0])
    mujoco.mj_forward(model, data)


    # Dynamics Matrix
    q_dim = model.nv
    u_dim = model.nu

    M = np.zeros((q_dim,q_dim))
    jacp = np.zeros((3, q_dim))
    jacd = np.zeros((3, q_dim))
    # Visualize and Simulate the system
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Our Controller Here
            mujoco.mj_fullM(model, M, data.qM)
            coriolis_matrix = data.qfrc_bias

            site_name = "end_effector"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, site_name)
            body_id = model.site_bodyid[site_id]

            arm_pos = data.site_xpos[site_id]
            mujoco.mj_jac(model, data, jacp, None, arm_pos, body_id)
            mujoco.mj_jacDot(model, data, jacd, None, arm_pos, body_id)
            arm_vel = jacp @ data.qvel

            # EEF Desired Acceleration
            des_pos = np.array([.1,0,0.2])
            des_vel = np.zeros(3)
            des_acc = np.zeros(3)

            eef_acc_des = 10 * (des_pos - arm_pos) + 1 * (des_vel - arm_vel)

            # Compute torque
            qqdot = np.linalg.pinv(jacp) @ (eef_acc_des - jacd @ data.qvel)
            u = M @ qqdot + coriolis_matrix

            data.ctrl = u

            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(0.002)


if __name__ == '__main__':
    main()
