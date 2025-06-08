#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from typing import Any
import os
from ml_collections import config_dict



import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
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
import g1_env


# Load the model
env_name = 'BH'
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
print('environment loaded')

ckpt_path = epath.Path('/tmp/humanoid_joystick/ckpts')
ckpt_path.mkdir(parents=True, exist_ok=True)


