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

def hello():
  print('hello world')
def get_state(model, data, nbatch=1):
  full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
  state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
  mujoco.mj_getState(model, data, state, full_physics)
  return np.tile(state, (nbatch, 1))

def xy_grid(nbatch, ncols=10, spacing=0.05):
  nrows = nbatch // ncols
  assert nbatch == nrows * ncols
  xmax = (nrows-1)*spacing/2
  rows = np.linspace(-xmax, xmax, nrows)
  ymax = (ncols-1)*spacing/2
  cols = np.linspace(-ymax, ymax, ncols)
  x, y = np.meshgrid(rows, cols)
  return np.stack((x.flatten(), y.flatten())).T

def benchmark(f, x_list=[None], ntiming=1, f_init=None):
  x_times_list = []
  for x in x_list:
    times = []
    for i in range(ntiming):
      if f_init is not None:
        x_init = f_init(x)

      start = time.perf_counter()
      if f_init is not None:
        f(x, x_init)
      else:
        f(x)
      end = time.perf_counter()
      times.append(end - start)

    x_times_list.append(np.mean(times))
  return np.array(x_times_list)

def render_many(model, data, state, framerate, camera=-1, shape=(480, 640),
                transparent=False, light_pos=None):
  nbatch = state.shape[0]

  if not isinstance(model, mujoco.MjModel):
    model = list(model)

  if isinstance(model, list) and len(model) == 1:
    model = model * nbatch
  elif isinstance(model, list):
    assert len(model) == nbatch
  else:
    model = [model] * nbatch

  # Visual options
  vopt = mujoco.MjvOption()
  vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
  pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
  catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  # Simulate and render.
  frames = []
  with mujoco.Renderer(model[0], *shape) as renderer:
    for i in range(state.shape[1]):
      if len(frames) < i * model[0].opt.timestep * framerate:
        for j in range(state.shape[0]):
          mujoco.mj_setState(model[j], data, state[j, i, :],
                             mujoco.mjtState.mjSTATE_FULLPHYSICS)
          mujoco.mj_forward(model[j], data)

          # Use first model to make the scene, add subsequent models
          if j == 0:
            renderer.update_scene(data, camera, scene_option=vopt)
          else:
            mujoco.mjv_addGeoms(model[j], data, vopt, pert, catmask, renderer.scene)

        # Add light, if requested
        if light_pos is not None:
          light = renderer.scene.lights[renderer.scene.nlight]
          light.ambient = [0, 0, 0]
          light.attenuation = [1, 0, 0]
          light.castshadow = 1
          light.cutoff = 45
          light.diffuse = [0.8, 0.8, 0.8]
          light.dir = [0, 0, -1]
          light.directional = 0
          light.exponent = 10
          light.headlight = 0
          light.specular = [0.3, 0.3, 0.3]
          light.pos = light_pos
          renderer.scene.nlight += 1

        # Render and add the frame.
        pixels = renderer.render()
        frames.append(pixels)
  return frames