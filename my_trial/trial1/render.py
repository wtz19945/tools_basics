import numpy as np
import mujoco
import mediapy as media
import mujoco.viewer
import time
import matplotlib.pyplot as plt

xml = """
<mujoco>
  <option gravity="0 0 10"/>
  
  <statistic center="0 0 0.55" extent="1.1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6"  ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
        height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
        rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="2.5 2.5"
        reflectance="0.2"/> 
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"  directional="false"/>
    <body name="box_and_sphere" euler="0 0 -0">
      <joint name="swing" type="hinge" axis="1 0 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""

# Make model and data
model = mujoco.MjModel.from_xml_string(tippe_top)
data = mujoco.MjData(model)

print('name of geom 1: ', model.geom(1).name)
print('name of body 0: ', model.body(0).name)

print('step time is:', model.opt.timestep)

print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, -9.81)
print('flipped gravity', model.opt.gravity)

print('positions', data.qpos)
print('velocities', data.qvel)

mujoco.mj_resetDataKeyframe(model, data, 0)

timevals = []
angular_velocity = []
stem_height = []

duration = 7
# Make renderer, render and show the pixels
with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running() and data.time < duration:
    mujoco.mj_step(model, data)
    viewer.sync()
    timevals.append(data.time)
    angular_velocity.append(data.qvel[3:6].copy())
    stem_height.append(data.geom_xpos[2,2])
    time.sleep(0.002)

dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(timevals, angular_velocity)
ax[0].set_title('angular velocity')
ax[0].set_ylabel('radians / second')

ax[1].plot(timevals, stem_height)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters')
_ = ax[1].set_title('stem height')
plt.show()