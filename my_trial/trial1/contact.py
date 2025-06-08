import numpy as np
import mujoco
import mediapy as media
import mujoco.viewer
import time
import matplotlib.pyplot as plt

free_body_MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
    rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
    reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1" mode="trackcom"/>
    <geom name="ground" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid" solimp=".99 .99 .01" solref=".001 1"/>
    <body name="box_and_sphere" pos="0 0 0">
      <freejoint/>
      <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1" solimp=".99 .99 .01"  solref=".001 1"/>
      <geom name="green_sphere" size=".06" pos=".1 .1 .1" rgba="0 1 0 1"/>
      <camera name="fixed" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2"/>
      <camera name="track" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2" mode="track"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(free_body_MJCF)
data = mujoco.MjData(model)
data2 = mujoco.MjData(model)

# Visualization options
options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# Set some visualization scales
model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.03
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3

# Reset data and initialize velocity
mujoco.mj_resetData(model, data)
data.qvel[3:6] = 5 * np.random.randn(3)

# Create another visualization option for the perturbed scene
vopt2 = mujoco.MjvOption()
vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

# Initialize the camera
cam = mujoco.MjvCamera()

# Launch the viewer and simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Create a scene to store the perturbed geometry
    perturbed_scene = mujoco.MjvScene(model, maxgeom=1000)

    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Perturb the data
        data2.qpos = data.qpos.copy()  # Make a copy of the original qpos
        data2.qpos[0] += 10.5  # Perturb position randomly
        data2.qpos[1] += 1

        # Forward the model to update the perturbed state
        mujoco.mj_forward(model, data2)

        # Add perturbed geometry to the scene
        mujoco.mjv_addGeoms(model, data2, vopt2, pert, catmask, perturbed_scene)

        # Update the viewer with the original and perturbed scene
        mujoco.mjv_updateScene(model, data, options, pert, cam, catmask, perturbed_scene)

        # Sync the viewer
        viewer.sync()

        # Optional: Add a small delay for smoother rendering
        time.sleep(0.002)  