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

# Defining Humanoid 
class Humanoid(PipelineEnv):
    def __init__(
            self,
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0,2.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
            **kwargs,

    ):
        filepath = os.path.join(os.path.dirname(__file__), '../trial2/mujoco/mjx/mujoco/mjx/test_data/humanoid/humanoid.xml')
        mj_model = mujoco.MjModel.from_xml_path(filepath)

        # Standard Parameter for mujoco simulation
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        # jax.debug.print("obs: {}", self.sys.qpos0)
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        
        velocity = data.xd.vel[0]

        vel_error = velocity - jp.array([.0, 0.0, 0.0])

        forward_reward = self._forward_reward_weight * jp.exp(-jp.sum(jp.square(vel_error)) / .25)

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])

# Get environment
envs.register_environment('humanoid', Humanoid)
env_name = 'humanoid'
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)

def progress(num_steps, metrics):
    if num_steps > 0:
        print(f'Step: {num_steps} \t Reward: Episode Reward: {metrics["eval/episode_reward"]}')

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 128, 128, 128),
)


# Training function
train = 1
if train == 1:
    num_timesteps = 10_000_000
else:
    num_timesteps = 0
    
train_fn = functools.partial(
    ppo.train, num_timesteps=num_timesteps, num_evals=5, reward_scaling=1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1024,
    batch_size=256, seed=0)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, eval_env = eval_env)

# Save the Policy
model_path = './tmp/mjx_brax_policy_humanoid'
if train == 1:
    model.save_params(model_path, params)

# Load Model and Define Inference Function
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# Test Policy
filepath = os.path.join(os.path.dirname(__file__), '../trial2/mujoco/mjx/mujoco/mjx/test_data/humanoid/humanoid.xml')
mj_model = mujoco.MjModel.from_xml_path(filepath)
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)

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

