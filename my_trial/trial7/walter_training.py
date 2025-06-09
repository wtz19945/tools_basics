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
import optax

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
import walter_env


np.set_printoptions(precision=3, suppress=True, linewidth=100)

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'

print('imported succesfully')

# jax.config.update("jax_enable_x64", True)

# Load the model
env_name = 'Walter'
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
reset_fn = jax.jit(env.reset)

print('environment loaded')

ckpt_path = epath.Path('/tmp/humanoid_joystick/ckpts')
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
  # save checkpoints
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f'{current_step}'
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)
  
make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(512, 256, 128), 
)

train = 0
if train == 1:  
    num_timesteps = 50_000_000
else:
    num_timesteps = 0

optimizer = optax.chain(
        optax.adaptive_grad_clip(clipping=0.01),
        optax.adam(learning_rate=3e-4),
)
        
train_fn = functools.partial(
      ppo.train, num_timesteps=num_timesteps, clipping_epsilon=0.2, max_grad_norm = 1, num_evals=20,
      reward_scaling=1, episode_length=1000, normalize_observations=True, num_resets_per_eval = 5,
      action_repeat=1, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=0.005, num_envs=4096, batch_size=256,
      network_factory=make_networks_factory,
      policy_params_fn=policy_params_fn,
      seed=0)

def progress(num_steps, metrics):
    if num_steps > 0:
        print(f'Step: {num_steps} \t Reward: Episode Reward: {metrics["eval/episode_reward"]}')

print('Training started')
make_inference_fn, params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)
print('Training finished')
model_path = './tmp/mjx_brax_policy_walter'
if train == 1:
    model.save_params(model_path, params)
    
    
# Load Model and Define Inference Function
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# compile jax functions
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

x_vel = -0.3  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
ang_vel = -0.0  #@param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
qvel = state.pipeline_state.qvel * 0 
state = state.tree_replace({'pipeline_state.qvel': qvel})
state.info['command'] = the_command
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 1000
render_every = 2
states = []

key = jax.random.key(0)
        
for i in range(n_steps):
    # Stop Command Sampling:
    key, subkey = jax.random.split(key)
    action, _ = inference_fn(state.obs, subkey)
    state = jit_step(state, action)
    state.info['step'] = 0 
    states.append(state.pipeline_state)

# Generate HTML:
html_string = html.render(
    sys=env.sys.tree_replace({'opt.timestep': env._dt}),
    states=states,
    height="100vh",
    colab=False,
)

html_path = os.path.join(
    os.path.dirname(__file__),
    "visualization/visualization_walter.html",
)

with open(html_path, "w") as f:
    f.writelines(html_string)