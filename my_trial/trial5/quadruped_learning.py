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
np.set_printoptions(precision=3, suppress=True, linewidth=100)

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'

print('imported succesfully')

def get_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1.5,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.8,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-2.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.0002,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.2,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.1,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config

a = get_config()
state_info = {
    'rewards': {k: 0.0 for k in a.rewards.scales.keys()}
}
print(a)
print(state_info)

for k in state_info['rewards']:
    print('xxx:', state_info['rewards'][k])
            

BARKOUR_ROOT_PATH = epath.Path('/home/orl/Tianze/mujoco_work/mujoco_menagerie/google_barkour_vb')

# Define the environment
class BarkourEnv(PipelineEnv):
    """ Define out barkour environment."""
    def __init__(self,
                obs_noise: float = 0.05,
                action_scale: float = 0.3,
                kick_vel: float = 0.05,
                scene_file: str = 'scene_mjx.xml',
                **kwargs):
        
        filepath = os.path.join(os.path.dirname(__file__), BARKOUR_ROOT_PATH/'scene_mjx.xml')
        mj_model = mujoco.MjModel.from_xml_path(filepath)
        
        sys = mjcf.load_model(mj_model)
        self._dt = 0.02 # This environment is 50 fps
        sys = sys.tree_replace({'opt.timestep': 0.004})
      
        # overried menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)
        
        self.reward_config = get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.reward_config.rewards.scales[k[:-6]] = v
        
        self._torso_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso')
        
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        
        # Set home position
        self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        
        # Joint bounds
        self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jp.array([0.52, 2.1, 2.1] * 4)
        
        # Site want to track
        feet_site = [
            'foot_front_left',
            'foot_hind_left',
            'foot_front_right',
            'foot_hind_right',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self._feet_site_id = np.array(feet_site_id)
        
        lower_leg_body = [
            'lower_leg_front_left',
            'lower_leg_hind_left',
            'lower_leg_front_right',
            'lower_leg_hind_right',
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        
        self._foot_radius = 0.0175
        self._nv = sys.nv
    
    # This function sample a random command from joystick
    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd
    
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        # Split the random number generator
        rng, key = jax.random.split(rng)
        
        # Initialize state
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))
        
        # Initialize
        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'last_vel': jp.zeros(12),
            'command': self.sample_command(key),
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()}, # All rewards initialized to 0
            'kick': jp.array([0.0, 0.0]),
            'step': 0,
        }

        obs_history = jp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        
        reward, done = jp.zeros(2)
        metrics = {'total_dist': 0.0}
        
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
            
        state = State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types
        
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)
        
        # Random Perturb the robot
        push_interval = 10                                              # every 10 steps kick/perturb the robot
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi) # Sample a random direction of perturbation
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])       
        kick *= jp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel 
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])        # apply the kick to the robot
        state = state.tree_replace({'pipeline_state.qvel': qvel})       # replace the qvel in the state
        
        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # Observation
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        
        # Get foot contact inbfo
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt
            
        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0  # Check orientation
        done |= jp.any(joint_angles < self.lowers)                          # Check joint limits
        done |= jp.any(joint_angles > self.uppers)                          # Check joint limits
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18         # Check height
        
        # Compute reward
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),  # pytype: disable=attribute-error
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),
        }
        
        # Sum and clip all rewards
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm    # Reset air time to zero if foot is on the ground
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        
        # reset the step counter when done
        state.info['step'] = jp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

            
        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state
   
    def _get_obs(self, pipeline_state: base.State, state_info: dict[str, Any], obs_history: jax.Array,) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        # Why these values?
        obs = jp.concatenate([
            jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
            state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
            pipeline_state.q[7:] - self._default_pose,           # motor angles
            state_info['last_act'],                              # last action
        ])

        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info['rng'], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs

    def _reward_tracking_lin_vel(self, command: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        """Tracking the base linear velocity."""
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(local_vel[:2] - command[:2]))
        lin_vel_reward = jp.exp(-lin_vel_error / self.reward_config.rewards.tracking_sigma)
        return lin_vel_reward
    
    def _reward_tracking_ang_vel(self, command: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        """Tracking the base angular velocity."""
        local_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.sum(jp.square(local_vel[2] - command[2]))
        ang_vel_reward = jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)
        return ang_vel_reward
    
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        """Reward for the base linear velocity in z direction."""
        return jp.square(xd.vel[0][2])
    
    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        """Reward for the base angular velocity in xy direction."""
        return jp.sum(jp.square(xd.ang[0, :2]))  # xy angular velocity
    
    def _reward_orientation(self, x: Transform) -> jax.Array:
        """Reward for the base orientation."""
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))
    
    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Reward for the joint torques."""
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))
    
    def _reward_action_rate(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        """Reward for the action rate."""
        return jp.sum(jp.square(action - last_action))
    
    def _reward_stand_still(self, command: jax.Array, joint_angles: jax.Array) -> jax.Array:
        """Reward for standing still with small vel commands."""
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            math.normalize(command[:2])[1] < 0.1)
        
    def _reward_feet_air_time(self, feet_air_time: jax.Array, first_contact: jax.Array, command: jax.Array) -> jax.Array:
        """Reward for the feet air time."""
        rew_air_time = jp.sum((feet_air_time - 0.1) * first_contact)  # Air time better larger than 0.1
        rew_air_time *= (
            math.normalize(command[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time
    
    def _reward_foot_slip(self, pipeline_state: base.State, contact_filt_cm: jax.Array) -> jax.Array:
        """ Reward for foot slippage """
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)                       
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt_cm.reshape((-1, 1)))
        
    def _reward_termination(self, done: jax.Array, step: int) -> jax.Array:
        """Reward for the termination."""
        return done & (step < 500)
        
        


envs.register_environment('barkour', BarkourEnv)
env_name = 'barkour'
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)


reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

key = jax.random.key(0)
key, subkey = jax.random.split(key)

env.reset(subkey)
state = reset_fn(subkey)

print("stepping")
state.info['command'] = jp.array([0.0, 0.0, 0.0])
# state = step_fn(state, jnp.zeros(env.sys.nu))
state = env.step(state, jp.zeros(env.sys.nu))
print("created")

print("start looping")
start = time.time()
num_steps = 1000
for i in range(num_steps):
    # Stop Command Sampling:
    state = step_fn(state, jp.zeros(env.sys.nu))
print("Avg step time:", (time.time() - start) / 1000)
print("sim done")
time.sleep(111)
    



ckpt_path = epath.Path('/tmp/quadrupred_joystick/ckpts')
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
  # save checkpoints
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f'{current_step}'
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))

train = 0
if train == 1:
    num_timesteps = 100_000_000
else:
    num_timesteps = 0
    
train_fn = functools.partial(
      ppo.train, num_timesteps=num_timesteps, num_evals=10,
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=8192, batch_size=256,
      network_factory=make_networks_factory,
      policy_params_fn=policy_params_fn,
      seed=0)

def progress(num_steps, metrics):
    if num_steps > 0:
        print(f'Step: {num_steps} \t Reward: Episode Reward: {metrics["eval/episode_reward"]}')
        
make_inference_fn, params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)

model_path = './tmp/mjx_brax_policy_barkour'
if train == 1:
    model.save_params(model_path, params)

# Load Model and Define Inference Function
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# compile jax functions
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

x_vel = 0.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
ang_vel = -0.5  #@param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info['command'] = the_command
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
render_every = 2
states = []

key = jax.random.key(0)

for i in range(n_steps):
    # Stop Command Sampling:
    key, subkey = jax.random.split(key)
    action, _ = inference_fn(state.obs, subkey)
    state = jit_step(state, action)
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
    "visualization/visualization.html",
)

with open(html_path, "w") as f:
    f.writelines(html_string)
    

# Here is simulation
# filepath = os.path.join(os.path.dirname(__file__), BARKOUR_ROOT_PATH/'scene_mjx.xml')
# mj_model = mujoco.MjModel.from_xml_path(filepath)
# mj_data = mujoco.MjData(mj_model)
# mujoco.mj_forward(mj_model, mj_data)

# ctrl = jp.zeros(mj_model.nu)
# rng = jax.random.PRNGKey(0)
        
# # Visualize and Simulate the system
# with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
#     while viewer.is_running():

#         rng, act_rng = jax.random.split(rng)

#         obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
#         ctrl, _ = jit_inference_fn(obs, act_rng)

#         mj_data.ctrl = ctrl

#         # Update the viewer
#         viewer.sync()

#         # Sleep for a bit to visualize the simulation:
#         time.sleep(0.002)