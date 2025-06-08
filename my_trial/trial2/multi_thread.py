
import mujoco
from mujoco import rollout
from mujoco import mjx

import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')

import copy
import time
from multiprocessing import cpu_count
import threading
import numpy as np
import jax
import jax.numpy as jp

import matplotlib
import matplotlib.pyplot as plt
import mediapy as media

from helper_function import *

nthread = cpu_count()

humanoid_path = 'mujoco/model/humanoid/humanoid.xml'
humanoid100_path = 'mujoco/model/humanoid/humanoid100.xml'
hopper_path ='dm_control/dm_control/suite/hopper.xml'

hello()

#@title Benchmarked models
tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size="1 1 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <camera name="distant" pos="0 -.4 .4" xyaxes="1 0 0 0 1 1"/>
    <body name="top" pos="0 0 .02">
      <freejoint name="top"/>
      <site name="top" pos="0 0 0"/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <sensor>
    <gyro name="gyro" site="top"/>
  </sensor>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""

# Create and initialize top model
top_model = mujoco.MjModel.from_xml_string(tippe_top)
top_data = mujoco.MjData(top_model)
# Set to the state to a spinning top (keyframe 0)
mujoco.mj_resetDataKeyframe(top_model, top_data, 0)
top_state = get_state(top_model, top_data)

# Create and initialize humanoid model
humanoid_model = mujoco.MjModel.from_xml_path(humanoid_path)
humanoid_data = mujoco.MjData(humanoid_model)
humanoid_data.qvel[2] = 4 # Make the humanoid jump
humanoid_state = get_state(humanoid_model, humanoid_data)

# Create and initialize humanoid100 model
humanoid100_model = mujoco.MjModel.from_xml_path(humanoid100_path)
humanoid100_data = mujoco.MjData(humanoid100_model)
h100_state = get_state(humanoid100_model, humanoid100_data)

start = time.time()
top_nstep = int(6 / top_model.opt.timestep)
top_state, _ = rollout.rollout(top_model, top_data, top_state, nstep=top_nstep)

humanoid_nstep = int(3 / humanoid_model.opt.timestep)
humanoid_state, _ = rollout.rollout(humanoid_model, humanoid_data,
                                    humanoid_state, nstep=humanoid_nstep)

humanoid100_nstep = int(3 / humanoid100_model.opt.timestep)
h100_state, _ = rollout.rollout(humanoid100_model, humanoid100_data,
                                       h100_state, nstep=humanoid100_nstep)
end = time.time()

start_render = time.time()
top_frames = render_many(top_model, top_data, top_state, framerate=60, shape=(240, 320))
humanoid_frames = render_many(humanoid_model, humanoid_data, humanoid_state, framerate=120, shape=(240, 320))
humanoid100_frames = render_many(humanoid100_model, humanoid100_data, h100_state, framerate=120, shape=(240, 320))

# humanoid and humanoid100 are shown at half speed
media.show_video(np.concatenate((top_frames, humanoid_frames, humanoid100_frames), axis=2), fps=60)
end_render = time.time()

print(f'Rollout took {end-start:.1f} seconds')
print(f'Rendering took {end_render-start_render:.1f} seconds')