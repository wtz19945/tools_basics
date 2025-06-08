import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os
import time

filepath = os.path.join(os.path.dirname(__file__), '/home/orl/Tianze/mujoco_work/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(filepath)
data = mujoco.MjData(model)
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)


# Bounds at the joint limits.
bounds = [model.jnt_range[:, 0], model.jnt_range[:, 1]]

# Inital guess is the 'home' keyframe.
x0 = model.key('home').qpos


with mujoco.viewer.launch_passive(model, data) as viewer:
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    # viewer.cam.distance = 4.0
    # viewer.cam.azimuth = 90
    # viewer.cam.elevation = -20

    while viewer.is_running():
        # Compute Inertia Matrix
        mujoco.mj_step(model, data)
        
        # Sync Viewer
        viewer.sync()
        time.sleep(0.002)  # Sleep to match timestep