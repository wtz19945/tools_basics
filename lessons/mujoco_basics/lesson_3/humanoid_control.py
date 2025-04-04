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

def main(argv=None):
    # xml_path = os.path.expanduser('~/Tianze/mujoco_work/mujoco_menagerie/unitree_g1/scene.xml')
    # Load the XML file
    filepath = os.path.join(os.path.dirname(__file__), 'unitree_g1/scene.xml')
    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)

    # Set timestep
    model.opt.timestep = 0.002

    # Reset to keyframe
    # keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    # mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    # Get control limits
    ctrl_limits = model.actuator_ctrlrange
    ctrl_lower = ctrl_limits[:, 0]
    ctrl_upper = ctrl_limits[:, 1]

    u_dim = model.nu
    q_dim = model.nv

    # This is the correct way to compute B matrix
    actuator_moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        actuator_moment,
        data.actuator_moment.reshape(-1),
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind.reshape(-1),
    )

    B = actuator_moment.T

    # OSC control
    hand_site = [
        "left_hand",
        "right_hand"
    ]
    num_hand_sites = len(hand_site)
    hand_dim = 3 * num_hand_sites
    hand_jacp = np.zeros((hand_dim, q_dim))
    hand_jacv = np.zeros((hand_dim, q_dim))
    hand_jacr = np.zeros((hand_dim, q_dim))
    hand_jacdr = np.zeros((hand_dim, q_dim))
    

    imu_site = [
        "imu_in_pelvis",
        "imu_in_torso"
    ]
    num_imu_sites = len(imu_site)

    jacp = np.zeros((3 * num_imu_sites, q_dim))
    jacv = np.zeros((3 * num_imu_sites, q_dim))
    jacr = np.zeros((3 * num_imu_sites, q_dim))
    jacdr = np.zeros((3 * num_imu_sites, q_dim))

    foot_site = [
        "left_foot_con1",
        "left_foot_con2",
        "left_foot_con3",
        "left_foot_con4",
        "right_foot_con1",
        "right_foot_con2",
        "right_foot_con3",
        "right_foot_con4",
    ]
    num_foot_sites = len(foot_site)
    foot_jac = np.zeros((3 * num_foot_sites, model.nv))  # Position Jacobians
    foot_jacd = np.zeros((3 * num_foot_sites, model.nv))  # Position Jacobians Derivatives

    f_dim = 3 * num_foot_sites
    var_num = u_dim + q_dim + f_dim
    Ru = np.diag(.001 * np.ones(u_dim))
    Rf = np.diag(.0 * np.ones(f_dim))
    initialized = False
    m = osqp.OSQP()

    mu = 0.6
    fric_cone = np.array([[1, 1, -mu ], [1, -1, -mu ], [-1, 1, -mu ], [-1, -1, -mu ]])
    force_cons = np.kron(np.eye(num_foot_sites), fric_cone)

    F_upper_lim = model.body_mass.sum() * 9.81 / 4 * np.ones(f_dim)
    F_lower_lim = -F_upper_lim
    F_lower_lim[2::3] = 0

    W = block_diag(np.diag(np.array([100,100,100,100,100,100, 100,100,100,100,100,100])), 10000 * np.eye(f_dim), 10 * np.eye(hand_dim*2 + 2))

    kp = 30.0
    kd =  5.0

    hand_kp = 30
    hand_kd = 10

    hand_dummy_jac = np.zeros((2,q_dim))
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")
    body_id = model.jnt_bodyid[joint_id]
    hand_dummy_jac[0, body_id + 4] = 1

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint")
    body_id = model.jnt_bodyid[joint_id]
    hand_dummy_jac[1, body_id + 4] = 1

    prd = .5
    omega = np.pi * 2 / prd

    # Start visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Compute Inertia Matrix
            M_flat = np.zeros((q_dim, q_dim))
            mujoco.mj_fullM(model, M_flat, data.qM)

            # Compute Gravity Compensation
            gravity = data.qfrc_bias
            
            # Get all the sites
            hand_site_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, name) for name in hand_site]
            imu_site_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, name) for name in imu_site]
            foot_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, name) for name in foot_site]

            hand_body_id = [model.site_bodyid[id_name] for id_name in hand_site_id]
            imu_body_id = [model.site_bodyid[id_name] for id_name in imu_site_id]
            foot_body_ids = [model.site_bodyid[id_name] for id_name in foot_site_ids]

            # Track site position
            hand_pos = np.array([data.site_xpos[site_id] for site_id in hand_site_id])
            imu_pos = np.array([data.site_xpos[site_id] for site_id in imu_site_id])
            foot_pos = np.array([data.site_xpos[site_id] for site_id in foot_site_ids])

            # Track imu site rotation
            imu_ang = np.array([rotm_2_euler(data.site_xmat[site_id].reshape(3,3)) for site_id in imu_site_id])
            hand_ang = np.array([rotm_2_euler(data.site_xmat[site_id].reshape(3,3)) for site_id in hand_site_id])

            # IMU Jacobian
            for i, site_name in enumerate(imu_site):
                mujoco.mj_jac(model, data, jacp[3*i:3*(i+1)], jacr[3*i:3*(i+1)], imu_pos[i], imu_body_id[i])
                mujoco.mj_jacDot(model, data, jacv[3*i:3*(i+1)], jacdr[3*i:3*(i+1)], imu_pos[i], imu_body_id[i])

            for i, site_name in enumerate(hand_site):
                mujoco.mj_jac(model, data, hand_jacp[3*i:3*(i+1)], hand_jacr[3*i:3*(i+1)], hand_pos[i], hand_body_id[i])
                mujoco.mj_jacDot(model, data, hand_jacv[3*i:3*(i+1)], hand_jacdr[3*i:3*(i+1)], hand_pos[i], hand_body_id[i])

            # Foot Jacobian
            for i, site_name in enumerate(foot_site):
                mujoco.mj_jac(model, data, foot_jac[3*i:3*(i+1)], None, foot_pos[i], foot_body_ids[i])
                mujoco.mj_jacDot(model, data, foot_jacd[3*i:3*(i+1)], None, foot_pos[i], foot_body_ids[i])

            imu_vel = jacp @ data.qvel
            imu_omg = jacr @ data.qvel
            hand_vel = hand_jacp @ data.qvel
            hand_omg = hand_jacr @ data.qvel
            foot_vel = foot_jac @ data.qvel

            # Task Space Acceleration   
            imu_pos_des = np.array([0.04, 0, .71, -0.04, 0, 0.98])
            imu_ang_des = np.array([0, 0, 0, 0, 0, 0])
            imu_pos_acc_des = kp * (imu_pos_des - imu_pos.reshape(1,3 * num_imu_sites)) + kd * (np.zeros(3 * num_imu_sites) - imu_vel)
            imu_ang_acc_des = kp * (imu_ang_des - imu_ang.reshape(1,3 * num_imu_sites)) + kd * (np.zeros(3 * num_imu_sites) - imu_omg)
            imu_ddot_des = np.hstack([imu_pos_acc_des, imu_ang_acc_des])

            hand_pos_des = np.array([0.25, 0.17 + .17 * np.sin(omega * data.time), .9, 0.25, -0.17 - .17 * np.sin(omega * data.time), .9])
            hand_vel_des = np.array([0, omega * .17 * np.cos(omega * data.time), 0, 0, -.17 * omega * np.cos(omega * data.time), 0])
            hand_ang_des = np.array([0, 0, 0, 0, 0, 0])
            hand_pos_acc_des = hand_kp * (hand_pos_des - hand_pos.reshape(1,3 * num_hand_sites)) + hand_kd * (hand_vel_des - hand_vel)
            hand_ang_acc_des = hand_kp * (hand_ang_des - hand_ang.reshape(1,3 * num_hand_sites)) + hand_kd * (np.zeros(3 * num_hand_sites) - hand_omg)
            hand_ddot_des = np.hstack([hand_pos_acc_des, hand_ang_acc_des])

            
            dummy_hand_pos = np.array([data.qpos[-13], data.qpos[-6]])
            dummy_hand_vel = hand_dummy_jac @ data.qvel
            dummy_hand_acc = hand_kp * (np.array([.4, -0.4]) - dummy_hand_pos) + hand_kd * (np.zeros(2) - dummy_hand_vel)
            foot_ddot_des = np.zeros(3 * num_foot_sites)


            eef_ddot_des = np.hstack([imu_ddot_des.flatten(), foot_ddot_des, hand_ddot_des.flatten(),dummy_hand_acc])
            jac_end = np.vstack([jacp,jacr,foot_jac,hand_jacp,hand_jacr,hand_dummy_jac])
            jacv_end = np.vstack([jacv,jacdr,foot_jacd,hand_jacv, hand_jacdr, np.zeros_like(hand_dummy_jac)])

            # Formulate QP
            offset = eef_ddot_des - jacv_end @ data.qvel

            Afull = np.block([[M_flat, -B, -foot_jac.T], [np.zeros((4 * num_foot_sites, q_dim + u_dim)), force_cons], [np.eye(var_num)]])
            A = sparse.csc_matrix(Afull)

            q = np.hstack([-1 * offset.T @ W @ jac_end, np.zeros(u_dim + f_dim)])
            
            lb = np.hstack([-gravity, -np.inf * np.ones(4 * num_foot_sites), -np.inf * np.ones(q_dim), ctrl_lower, F_lower_lim])
            ub = np.hstack([-gravity,  np.zeros(4 * num_foot_sites),          np.inf * np.ones(q_dim), ctrl_upper, F_upper_lim])

            if(initialized):
                Hessian = block_diag(jac_end.T @ W @ jac_end, Ru, Rf)
                P = sparse.csc_matrix(Hessian)
                m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
                m.warm_start(x = results.x, y = results.y)
            else:
                Hessian = block_diag(jac_end.T @ W @ jac_end, Ru, Rf)
                P = sparse.csc_matrix(Hessian)
                m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
                initialized = True

            results = m.solve()
            u = results.x[q_dim : q_dim + u_dim]
            data.ctrl = u

            # time.sleep(15)
            # Step Simulation
            mujoco.mj_step(model, data)

            # Sync Viewer
            viewer.sync()
            time.sleep(0.002)  # Sleep to match timestep


if __name__ == '__main__':
    main()
