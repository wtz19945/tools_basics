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

# Start with Defining Learning Environment
class CartPole(PipelineEnv):
    """ Environment for training cartpole """
    def __init__(self,
                 reward_scale = 1.0,
                 ctrl_cost_weight = 0.0,
                 reset_noise_scale = 0.1,
                 **kwargs):
        
        mj_model = mujoco.MjModel.from_xml_path("cart_pole.xml")

        # Standard Parameter for mujoco simulation
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        
        
        # Set up Brax simulation
        sys = mjcf.load_model(mj_model)
        physics_steps_per_control_step = 10
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'
        sys = sys.tree_replace({'opt.timestep': 0.002})
        super().__init__(sys, **kwargs)

        # Set up weights
        self._reward_scale = reward_scale
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
    
    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key, x_key = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Set up the initial conditions
        theta_init = jax.random.uniform(theta_key, (1,), minval=-3.14, maxval=3.14)[0]
        x_init = jax.random.uniform(x_key, (1,), minval=-2.5, maxval=2.5)[0]
        
        qinit = jax.numpy.array([x_init, theta_init])
        qd_init = jax.random.uniform(qd_key, (self.sys.nv,), minval=-1, maxval=1)
        
        # Set up pipeline environment initials
        data = self.pipeline_init(qinit, qd_init)

        # Reset the observations
        obs = self._get_obs(data, jax.numpy.zeros(self.sys.nu))

        # Reset reward 
        reward, done, zero = jax.numpy.zeros(3)
        # Metrics are logging and tracking training
        metrics = {'reward': zero}

        # Return the state
        return State(data, obs, reward, done, metrics)
    
    # This function step forward and return the new State
    def step(self, state: State, action: jax.Array) -> State:
        # Get current state and step one step forward
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Get simulation states
        x_position = data.qpos[0]
        rot_position = data.qpos[1]

        # Compute reward
        # reward = 5 * jax.numpy.cos(rot_position) - 1 * jax.numpy.square(x_position)
        error = jax.numpy.cos(rot_position) - 1
        tracking = jp.exp(-jp.square(error) / 0.5)
        small_x = 0.01 * jp.square(x_position)
        small_velocity = 0.001 * jp.sum(jp.square(data.qpos))
        small_action = 0.001 * jp.sum(jp.square(x_position))
        reward = tracking - small_x - small_velocity - small_action
        # reward = tracking
        reward = jp.clip(reward, 0.0, 10000.0) 
        
        # Terminate if outside the range of the rail and pendulum is past the rail:
        outside_x = jax.numpy.abs(x_position) > 3.0
        done = outside_x
        done = jax.numpy.float32(done)

        # Get new observation
        obs = self._get_obs(data, action)

        # Update metric
        state.metrics.update(reward=reward)

        # Update State
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done
        )
    
    def _get_obs(self, data: mjx.Data, action:  jax.Array) -> jax.numpy.ndarray:
        # Put current state as observation
        return jax.numpy.concatenate([data.qpos, data.qvel, action])


# Register the custom environment
envs.register_environment('cartpole_mjx', CartPole)

# Create environment instance for training and evaluation
env = envs.get_environment('cartpole_mjx')
eval_env = envs.get_environment('cartpole_mjx')

reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

print("jax function created")

key = jax.random.key(0)
key, subkey = jax.random.split(key)

env.reset(subkey)
state = reset_fn(subkey)

# state = step_fn(state, jnp.zeros(env.sys.nu))
state = env.step(state, jp.zeros(env.sys.nu))

# Print progress
def progress(num_steps, metrics):
    if num_steps > 0:
        print(f'Step: {num_steps} \t Reward: Episode Reward: {metrics["eval/episode_reward"]}')

# Make training network config
make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 128, 128),
    value_hidden_layer_sizes=(128, 128, 128),
)

# PPO training config
train_fn = functools.partial(
    ppo.train,
    num_timesteps=100_000_000,  # Increase training time
    num_evals=10,
    episode_length=1000,
    num_envs=2048,
    num_eval_envs=4,
    batch_size=256,  # Larger batch size for more stable updates
    num_minibatches=32,
    unroll_length=20,
    num_updates_per_batch=16,  # More updates per batch for faster learning
    normalize_observations=True,
    discounting=0.98,  # Slightly higher discounting
    learning_rate=.5 * 1e-3,  # Lower learning rate for larger starting angles
    entropy_cost=.01,  # Slightly lower entropy cost for better exploration
    network_factory=make_networks_factory,
    seed=0,
)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, eval_env = eval_env)

# Save the Policy
model_path = './tmp/mjx_brax_policy_cartpole'
model.save_params(model_path, params)

# Load Model and Define Inference Function
params = model.load_params(model_path)


inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)


# Test Policy
filepath = os.path.join(os.path.dirname(__file__), 'cart_pole.xml')
mj_model = mujoco.MjModel.from_xml_path(filepath)
mj_data = mujoco.MjData(mj_model)

mj_data.qpos[1] = -jp.pi / 4
ctrl = jp.zeros(mj_model.nu)
rng = jax.random.PRNGKey(0)

# Visualize and Simulate the system
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():

        rng, act_rng = jax.random.split(rng)

        obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
        ctrl, _ = jit_inference_fn(obs, act_rng)

        mj_data.ctrl = ctrl

        # Simulate the system for a few steps:
        mujoco.mj_step(mj_model, mj_data)

        # Update the viewer
        viewer.sync()

        # Sleep for a bit to visualize the simulation:
        time.sleep(0.002)