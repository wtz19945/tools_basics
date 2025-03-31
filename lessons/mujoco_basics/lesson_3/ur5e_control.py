import os

import numpy as np
import mujoco
import mujoco.viewer

import time

import osqp
import scipy.sparse as sparse
from scipy.linalg import block_diag

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


def main(argv=None):
    # xml_path = os.path.expanduser('~/Tianze/mujoco_work/mujoco_menagerie/unitree_g1/scene.xml')
    # Load the XML file
    filepath = os.path.join(os.path.dirname(__file__), 'universal_robots_ur5e/ur5e.xml')
    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)

    # Set timestep
    model.opt.timestep = 0.002

    # Reset to keyframe
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    # Get control limits
    ctrl_limits = model.actuator_ctrlrange
    ctrl_lower = ctrl_limits[:, 0]
    ctrl_upper = ctrl_limits[:, 1]

    u_dim = model.nu
    q_dim = model.nv

    # Correct B matrix
    B = np.vstack([
        np.zeros((q_dim - u_dim, u_dim)),  # Non-actuated joints
        np.eye(u_dim)                      # Actuated joints
    ])

    # OSC control
    jacp = np.zeros((3, q_dim))
    jacv = np.zeros((3, q_dim))

    jacr = np.zeros((3, q_dim))
    jacdr = np.zeros((3, q_dim))

    var_num = u_dim + q_dim
    R = np.diag(.0 * np.ones(u_dim))
    initialized = False
    m = osqp.OSQP()

    radius = 0.2
    prd = 5
    omega = np.pi * 2 / prd
    
    kp = 50.0
    kd = 5.0

    # Start visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Compute Inertia Matrix
            M_flat = np.zeros((q_dim, q_dim))
            mujoco.mj_fullM(model, M_flat, data.qM)

            # Compute Gravity Compensation
            gravity = data.qfrc_bias

            # Add Task space tracking 
            site_name = "attachment_site"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, site_name)
            body_id = model.site_bodyid[site_id]

            # Task space Kinematics
            arm_pos = data.site_xpos[site_id] 
            rotm = data.site_xmat[site_id].reshape(3,3)
            rot_ang = rotm_2_euler(rotm)

            mujoco.mj_jac(model, data, jacp, jacr, arm_pos, body_id)
            mujoco.mj_jacDot(model, data, jacv, jacdr, arm_pos, body_id)
            arm_vel = jacp @ data.qvel
            rot_vel = jacr @ data.qvel

            des_pos = np.array([-.13 + radius* np.sin(omega * data.time), .49 + radius* np.cos(omega * data.time), .5])
            des_vel = np.array([omega * radius * np.cos(omega * data.time), -omega * radius * np.sin(omega * data.time), 0])
            des_acc = np.array([-pow(omega,2) * radius * np.sin(omega * data.time), -pow(omega,2) * radius * np.cos(omega * data.time), 0])

            xddot_des = kp * (des_pos - arm_pos) + kd * (des_vel - arm_vel) + des_acc
            # Find the correct rddot_des
            
            des_rot = np.array([-np.pi/2, 0.0, 0.0])
            rddot_des = kp * (des_rot - rot_ang) + kd * (np.zeros(3) - rot_vel)
            endddot_des = np.hstack([xddot_des, rddot_des])

            jac_end = np.vstack([jacp,jacr])
            jacv_end = np.vstack([jacv,jacdr])

            offset = endddot_des - jacv_end @ data.qvel
            
            Afull = np.block([[M_flat, -np.eye(u_dim)], [np.eye(var_num)]])
            A = sparse.csc_matrix(Afull)

            q = np.hstack([-1 * offset.T @ jac_end, np.zeros(u_dim)])
            lb = np.hstack([-gravity, -np.inf * np.ones(q_dim), ctrl_lower])
            ub = np.hstack([-gravity, np.inf * np.ones(q_dim), ctrl_upper])

            if(initialized):
                Hessian = block_diag(jac_end.T @ jac_end, R)
                P = sparse.csc_matrix(Hessian)
                m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
                m.warm_start(x = results.x, y = results.y)
            else:
                Hessian = block_diag(jac_end.T @ jac_end, R)
                P = sparse.csc_matrix(Hessian)
                m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
                initialized = True
            
            results = m.solve()
            data.ctrl = results.x[q_dim : var_num]
            # Step Simulation
            mujoco.mj_step(model, data)

            # Sync Viewer
            viewer.sync()
            time.sleep(0.002)  # Sleep to match timestep


if __name__ == '__main__':
    main()
