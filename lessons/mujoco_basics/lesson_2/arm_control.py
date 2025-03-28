import os

import numpy as np
import mujoco
import mujoco.viewer

import time

import osqp

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
    data.qpos = np.array([0.0, -np.pi / 2])
    data.qvel = np.array([0.0, 0.0])
    data.ctrl = np.array([0.0])
    mujoco.mj_forward(model, data)

    prd = 5
    omega = np.pi * 2 / prd
    kp = 200.0
    kd = 20.0
    
    # Visualize and Simulate the system
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Get the Inertia Matrix
            M_flat = np.zeros((model.nv , model.nv))  # 
            mujoco.mj_fullM(model, M_flat, data.qM)   #

            # Get the bias term
            coriolis_matrix = data.qfrc_bias

            # Add Task space tracking 
            site_name = "end_effector"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, site_name)
            body_id = model.site_bodyid[site_id]

            # Task space position
            arm_pos = data.site_xpos[site_id] 

            jacp = np.zeros((3, model.nv))
            jacv = np.zeros((3, model.nv))

            mujoco.mj_jac(model, data, jacp, None, arm_pos, body_id)
            mujoco.mj_jacDot(model, data, jacv, None, arm_pos, body_id)

            # Task Space Velocity
            arm_vel = jacp @ data.qvel

            des_pos = np.array([0.4 * np.sin(omega * data.time), 0, 0.4 * np.cos(omega * data.time)])
            des_vel = np.array([omega * 0.4 * np.cos(omega * data.time), 0, -omega * 0.4 * np.sin(omega * data.time)])
            des_acc = np.array([-pow(omega,2) * 0.4 * np.sin(omega * data.time), 0, -pow(omega,2) * 0.4 * np.cos(omega * data.time)])

            xddot_des = kp * (des_pos - arm_pos) + kd * (des_vel - arm_vel) + des_acc
            qddot_des = np.linalg.pinv(jacp) @ (xddot_des - jacv @ data.qvel)

            u = M_flat @ qddot_des + coriolis_matrix


            # An OSQP equivalent
            m = osqp.OSQP()
            offset = xddot_des - jacv @ data.qvel
            # Hessian = 
            # Gradient = 
            # A = 
            # lb = 
            # ub = 

            # m.setup(P=P, q=q, A=A, lb=lb, ub=ub)
            # results = m.solve()

            data.ctrl = u
            
            # print(jacp)
            # print(jacv)
            # void mj_jacBody(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int body);

            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(0.002)


if __name__ == '__main__':
    main()
