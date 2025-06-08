#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
import mujoco.viewer
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

import time
import numpy as np
import itertools
from typing import Callable, NamedTuple, Optional, Union, List
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True, linewidth=100)

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'

print('imported succesfully')

def example1():
    xml = """
    <mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
        <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
    </worldbody>
    </mujoco>
    """

    # Make model, data, and renderer
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    print(jax.devices())

    # Load data to GPU
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    print(mj_data.qpos, type(mj_data.qpos))
    print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

    # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    #     while viewer.is_running():
    #         # MJX simulation step on GPU
    #         mjx_data = mjx.step(mjx_model, mjx_data)

    #         # CPU for visual
    #         updated_data = mjx.get_data(mj_model, mjx_data)
    #         mj_data.qpos[:] = updated_data.qpos
    #         mj_data.qvel[:] = updated_data.qvel
    #         mujoco.mj_forward(mj_model, mj_data)  # Make sure forces etc. are consistent

    #         # mujoco.mj_step(mj_model, mj_data)
    #         # Sync Viewer
    #         viewer.sync()
    #         time.sleep(0.002)  # Sleep to match timestep

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 4096)

    """
    This line initialize the state on a batch:
         mjx_data.replace(qpos=jax.random.uniform(rng, (1,)) : replace the qpos with a 1-d sampled date using rng
         lambda rng: define a lambda function
         
    """
    batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

    """
    This line compiles a batched simulation step function that works efficiently on GPU:
        mjx.step: simulate the system to get the new data
        in_axes=(None,0): None is because the first element of mjx.step (mjx_model) is kept the same
                          0 is because the data is stored in column, so it is paralleled along axis-0
        jax.jit:  This complie the function for faster running             
    """
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    batch = jit_step(mjx_model, batch)

    print(batch.qpos)
    print(batch.qpos.size)

    batched_mj_data = mjx.get_data(mj_model, batch)




def main():
    example1()


if __name__ == '__main__':
    main()

