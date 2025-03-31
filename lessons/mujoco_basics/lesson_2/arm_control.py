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

    # Define Control System Parameters
    radius = 0.3
    prd = 5
    omega = np.pi * 2 / prd
    
    kp = 100.0
    kd = 10.0
    
    ctrl_limits = model.actuator_ctrlrange
    ctrl_lower = ctrl_limits[:, 0]  # Lower limit vector
    ctrl_upper = ctrl_limits[:, 1]  # Upper limit vector

    u_dim = model.nu
    q_dim = model.nv
    
    jacp = np.zeros((3, q_dim))
    jacv = np.zeros((3, q_dim))
    
    # Define OSC Parameters
    var_num = u_dim + q_dim
    R = np.diag(.0 * np.ones(u_dim))
    initialized = False
    m = osqp.OSQP()

    B = np.zeros((model.nv, model.nu))
    # Iterate through all actuators and assign 1s at the correct positions
    for i in range(model.nu):  # Loop over actuators
        jnt_id = model.dof_jntid[i]  # Get the joint ID controlled by this actuator
        B[jnt_id, i] = 1  # Assign 1 in the corresponding row and column

    print(B)
    # Visualize and Simulate the system
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Get the Inertia Matrix
            M_flat = np.zeros((q_dim , q_dim))  # 
            mujoco.mj_fullM(model, M_flat, data.qM)   #

            # Get the bias term
            coriolis_matrix = data.qfrc_bias

            # Add Task space tracking 
            site_name = "end_effector"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, site_name)
            body_id = model.site_bodyid[site_id]

            # Task space Kinematics
            arm_pos = data.site_xpos[site_id] 
            mujoco.mj_jac(model, data, jacp, None, arm_pos, body_id)
            mujoco.mj_jacDot(model, data, jacv, None, arm_pos, body_id)
            arm_vel = jacp @ data.qvel

            # Task Space Acc Cmd
            des_pos = np.array([radius* np.sin(omega * data.time), 0.0, radius* np.cos(omega * data.time)])
            des_vel = np.array([omega * radius * np.cos(omega * data.time), 0, -omega * radius * np.sin(omega * data.time)])
            des_acc = np.array([-pow(omega,2) * radius * np.sin(omega * data.time), 0, -pow(omega,2) * radius * np.cos(omega * data.time)])

            xddot_des = kp * (des_pos - arm_pos) + kd * (des_vel - arm_vel) + des_acc
            # qddot_des = np.linalg.pinv(jacp) @ (xddot_des - jacv @ data.qvel)
            # u = M_flat @ qddot_des + coriolis_matrix
            # data.ctrl = u
            
            # An OSQP equivalent
            offset = xddot_des - jacv @ data.qvel

            Afull = np.block([[M_flat, -np.eye(u_dim)], [np.eye(var_num)]])
            A = sparse.csc_matrix(Afull)

            q = np.hstack([-1 * offset.T @ jacp, np.zeros(u_dim)])
            lb = np.hstack([-coriolis_matrix, -np.inf * np.ones(q_dim), ctrl_lower])
            ub = np.hstack([-coriolis_matrix, np.inf * np.ones(q_dim), ctrl_upper])

            if(initialized):
                Hessian = np.triu(block_diag(jacp.T @ jacp, R))
                P = sparse.csc_matrix(Hessian)
                m.update(Px=P.data, q=q, Ax=A.data, l=lb, u=ub)
                m.warm_start(x = results.x, y = results.y)
            else:
                Hessian = block_diag(jacp.T @ jacp, R)
                P = sparse.csc_matrix(Hessian)
                m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
                initialized = True
            
            results = m.solve()
            data.ctrl = results.x[q_dim : var_num]

            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(0.002)


if __name__ == '__main__':
    main()
