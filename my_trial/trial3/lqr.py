import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os
import time


filepath = os.path.join(os.path.dirname(__file__), '../trial2/mujoco/model/humanoid/humanoid.xml')
model = mujoco.MjModel.from_xml_path(filepath)
data = mujoco.MjData(model)
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand_on_left_leg")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
mujoco.mj_forward(model, data)

# Force at default height
data.qacc = 0
mujoco.mj_inverse(model, data)
print('force at default height')
print(data.qfrc_inverse)

height_offset = np.linspace(-.001,0.001, 2001)
vertical_forces = []

for offset in height_offset:
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    data.qpos[2] += offset
    mujoco.mj_inverse(model, data)
    vertical_forces.append(data.qfrc_inverse[2])

# Find the height-offset at which the vertical force is smallest.
idx = np.argmin(np.abs(vertical_forces))
best_offset = height_offset[idx]

# # Plot the relationship.
# plt.figure(figsize=(10, 6))
# plt.plot(height_offset * 1000, vertical_forces, linewidth=3)
# # Red vertical line at offset corresponding to smallest vertical force.
# plt.axvline(x=best_offset*1000, color='red', linestyle='--')
# # Green horizontal line at the humanoid's weight.
# weight = model.body_subtreemass[1]*np.linalg.norm(model.opt.gravity)
# plt.axhline(y=weight, color='green', linestyle='--')
# plt.xlabel('Height offset (mm)')
# plt.ylabel('Vertical force (N)')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.title(f'Smallest vertical force '
#           f'found at offset {best_offset*1000:.4f}mm.')
# plt.show()


mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()  # Save the position setpoint.
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()
print('force at modified height')
print(data.qfrc_inverse)

actuator_moment = np.zeros((model.nu, model.nv))
mujoco.mju_sparse2dense(
    actuator_moment,
    data.actuator_moment.reshape(-1),
    data.moment_rownnz,
    data.moment_rowadr,
    data.moment_colind.reshape(-1),
)

ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
print('control setpoint:', ctrl0)

data.ctrl = ctrl0
mujoco.mj_forward(model, data)
print('actuator forces:', data.qfrc_actuator)
print('contact forces', data.qfrc_constraint)

# Set the state and controls to their setpoints.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
data.ctrl = ctrl0
mujoco.mj_forward(model, data)

# Set up LQR gains
nu = model.nu  # Alias for the number of actuators.
R = np.eye(nu)

nv = model.nv  # Shortcut for the number of DoFs.
# Get the Jacobian for the root body (torso) CoM.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
mujoco.mj_forward(model, data)
jac_com = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_com, None, model.body('torso').id)

# Get the Jacobian for the left foot.
jac_foot = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body('foot_left').id)

jac_diff = jac_com - jac_foot
Qbalance = jac_diff.T @ jac_diff

joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
abdomen_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'abdomen' in name
    and not 'z' in name
]
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'left' in name
    and ('hip' in name or 'knee' in name or 'ankle' in name)
    and not 'z' in name
]
balance_dofs = abdomen_dofs + left_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Cost coefficients.
BALANCE_COST        = 1000  # Balancing.
BALANCE_JOINT_COST  = 3     # Joints required for balancing.
OTHER_JOINT_COST    = .3    # Other joints.

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])

# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2*nv, 2*nv))
B = np.zeros((2*nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

dq = np.zeros(model.nv)

iter = 1
model.vis.map.force = 0.01
# Start visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    while viewer.is_running():
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T
        data.ctrl = ctrl0 - K @ dx
        
        if iter > 500:
            data.qfrc_applied[3] = 2 * np.random.randn(1)[0]
            iter = 0
        else:
            iter = iter + 1
        # Compute Inertia Matrix
        mujoco.mj_step(model, data)

        
        # Sync Viewer
        viewer.sync()
        time.sleep(0.002)  # Sleep to match timestep