from typing import Any, Dict
import os
from absl import app

import flax.serialization
import jax
import jax.numpy as jnp

import flax.struct
import flax.serialization

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco

# Types:
PRNGKey = jax.Array

@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_linear_velocity: float = 1.5
    tracking_angular_velocity: float = 0.8
    # Penalties / Regularization Terms:
    linear_z_velocity: float = -2.0
    angular_xy_velocity: float = -0.05
    orientation: float = -5.0
    torque: float = -2e-4
    action_rate: float = -0.01
    stand_still: float = -0.5
    termination: float = -1.0
    slip: float = 0.0
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25


class Walter(PipelineEnv):

    def __init__(
        self,
        filename: str = 'walter/scene.xml',
        config: RewardConfig = RewardConfig(),
        **kwargs,
    ):
        # Load the MJCF file
        filename = f'models/{filename}'
        self.filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
            filename,
        )
        # Create the mjcf system
        sys = mjcf.load(self.filepath)

        self.step_dt = 0.02
        n_frames = kwargs.pop('n_frames', int(self.step_dt/sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames = n_frames)

        # Setting the initial state
        self._init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos)[7:]
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.limb_idx = jnp.array([0, 1, 4, 5, 8, 9, 12, 13])
        self.wheel_idx = jnp.array([2, 3, 6, 7, 10, 11, 14, 15])
        self.actuated_idx = sys.actuator.q_id
        self.ctrl_lb = jnp.array([-jnp.pi / 4, -jnp.pi / 4, -5.0, -5.0] * 4)
        self.ctrl_ub = jnp.array([jnp.pi / 4, jnp.pi / 4, 5.0, 5.0] * 4)

        # Set the reward config and other parameters:
        self.kernel_sigma = config.kernel_sigma
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        self.reward_config = config_dict

        # Observation parameters:
        self.history_length = 15
        self.num_observations = 40

        # Site/Body IDs:
        self.body_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'body',
        )

        wheel_sites = [
            'bl_wheel_1',
            'bl_wheel_2',
            'br_wheel_1',
            'br_wheel_2',
            'fl_wheel_1',
            'fl_wheel_2',
            'fr_wheel_1',
            'fr_wheel_2',
        ]
        wheel_site_ids = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, wheel_site)
            for wheel_site in wheel_sites
        ]
        self.wheel_site_ids = jnp.array(wheel_site_ids)

        shin_bodies = [
            'bl_shin',
            'br_shin',
            'fl_shin',
            'fr_shin',
        ]
        shin_body_ids = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, shin_body)
            for shin_body in shin_bodies
        ]
        self.shin_body_ids = jnp.array(shin_body_ids)
        self.wheel_radius = 0.0565

        # Control Scales:
        self._action_scale = 0.3
        self._torque_scale = 0.3

    def sample_command(self, rng: PRNGKey) -> jax.Array:
        linear_forward_velocity = [-0.6, 1.5]
        angular_velocity = [-0.7, 0.7]

        _, forward_velocity_key, angular_velocity_key = jax.random.split(rng, 3)

        linear_forward_velocity = jax.random.uniform(
            forward_velocity_key,
            (1,),
            minval=linear_forward_velocity[0],
            maxval=linear_forward_velocity[1],
        )

        angular_velocity = jax.random.uniform(
            angular_velocity_key,
            (1,),
            minval=angular_velocity[0],
            maxval=angular_velocity[1],
        )

        new_cmd = jnp.array([linear_forward_velocity[0], angular_velocity[0]])
        return new_cmd
    
    def reset(self, rng: PRNGKey) -> State:
        # Generate new rng keys:
        key, cmd_key, q_key, qd_key = jax.random.split(rng, 4)

        q = self._init_q + jax.random.uniform(
            key=q_key,
            shape=(self.sys.q_size(),),
            minval=0.0,
            maxval=0.0,
        )

        qd = jax.random.uniform(
            key=qd_key,
            shape=(self.sys.qd_size(),),
            minval=0.0,
            maxval=0.0,
        )

        # Initialize the pipeline state:
        pipeline_state = self.pipeline_init(q=q, qd=qd)

        # Initialize Reward and Termination State:
        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        # State Info: (Used to track values inbetween steps)
        state_info = {
            'rng_key': key,
            'command': self.sample_command(cmd_key),
            'previous_state': {
                'q': jnp.zeros_like(q),
                'qd': jnp.zeros_like(qd),
            },
            'previous_action': jnp.zeros_like(self.default_ctrl),
            'previous_contact': jnp.zeros_like(self.wheel_site_ids, dtype=bool),
            'wheel_air_time': jnp.zeros_like(self.wheel_site_ids),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'step': 0,
        }

        # Metrics: (Reward Dictionary)
        metrics = {'total_distance': 0.0}
        for k in state_info['rewards']:
            metrics[k] =state_info['rewards'][k]

        # Initialize the observation:
        observation_history = jnp.zeros(
            self.history_length * self.num_observations,
        )
        observation = self.get_observation(pipeline_state, state_info, observation_history)

        # Create State:
        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )

        return state


    # Function to define how a simulation step is done
    def step(self, state: State, action: jax.Array) -> State:
        key, subkey, cmd_key = jax.random.split(state.info['rng_key'], 3)

        # Perform a forward physics step
        position_targets = self.default_pose[self.limb_idx] + action[self.limb_idx] * self._action_scale
        torque_targets = action[self.wheel_idx] * self._torque_scale
        motor_targets = jnp.array([
            position_targets[0], position_targets[1], torque_targets[0], torque_targets[1],
            position_targets[2], position_targets[3], torque_targets[2], torque_targets[3],
            position_targets[4], position_targets[5], torque_targets[4], torque_targets[5],
            position_targets[6], position_targets[7], torque_targets[6], torque_targets[7],
        ])

        # Perform a forward physics step
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)

        # Get New Observation:
        observation = self.get_observation(pipeline_state, state.info, state.obs)

        # States:
        x = pipeline_state.x
        xd = pipeline_state.xd
        joint_angles = pipeline_state.q[7:]
        actuated_joints = pipeline_state.q[self.actuated_idx]

        # Wheel Contact:
        wheel_q = pipeline_state.site_xpos[self.wheel_site_ids]
        wheel_z_contact = wheel_q[:, 2] - self.wheel_radius
        contact = wheel_z_contact < 1e-3
        contact_filter_mm = contact | state.info['previous_contact']
        contact_filter_cm = (wheel_z_contact < 3e-2) | state.info['previous_contact']
        first_contact = (state.info['wheel_air_time'] > 0) * contact_filter_mm
        state.info['wheel_air_time'] += self.step_dt

        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self.body_idx - 1]), up) < 0
        done |= jnp.any(actuated_joints < self.ctrl_lb)
        done |= jnp.any(actuated_joints > self.ctrl_ub)
        done |= pipeline_state.x.pos[self.body_idx - 1, 2] < 0.1

        # Reward:
        rewards = {
            'tracking_linear_velocity': self._reward_tracking_linear_velocity(
                state.info['command'], x, xd,
            ),
            'tracking_angular_velocity': self._reward_tracking_angular_velocity(
                state.info['command'], x, xd,
            ),
            'linear_z_velocity': self._reward_linear_z_velocity(xd),
            'angular_xy_velocity': self._reward_angular_xy_velocity(xd),
            'orientation': self._reward_orientation(x),
            'torque': self._reward_torque(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'stand_still': self._reward_stand_still(state.info['command'], joint_angles),
            'termination': jnp.float64(
                self._reward_termination(done, state.info['step'])
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done, state.info['step'])
            ),
            'slip': self._reward_slip(pipeline_state, contact_filter_cm),
        }
        reward = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()), -jnp.inf, jnp.inf)
        
        # State Info:
        state.info['previous_state']['q'] = pipeline_state.q
        state.info['previous_state']['qd'] = pipeline_state.qd
        state.info['previous_action'] = action
        state.info['previous_contact'] = contact
        state.info['rewards'] = rewards
        state.info['wheel_air_time'] *= ~contact_filter_mm
        state.info['step'] += 1
        state.info['rng_key'] = subkey

        # Sample New Command and Reset Step Counter:
        state.info['command'] = jnp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_key),
            state.info['command'],
        )
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 500),
            0,
            state.info['step'],
        )

        # Log Metrics:
        state.metrics['total_distance'] = math.normalize(
            x.pos[self.body_idx - 1],
        )[1]
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
        )
        return state
    
    # Gets the current observation from the environment
    def get_observation(
        self,
        pipeline_state: State,
        state_info: Dict[str, Any],
        observation_history: jax.Array,
    ) -> jnp.ndarray:
        # Observation: [yaw_rate, projected_gravity, command, relative_motor_positions, previous_action]
        inverse_base_rotation = math.quat_inv(
            pipeline_state.x.rot[0],
        )
        local_yaw_rate = math.rotate(
            pipeline_state.xd.ang[0],
            inverse_base_rotation,
        )[2]
        projected_gravity = math.rotate(
            jnp.array([0.0, 0.0, -1.0]),
            inverse_base_rotation,
        )

        observation = jnp.concatenate([
            jnp.array([local_yaw_rate]),
            projected_gravity,
            state_info['command'],
            pipeline_state.q[7:] - self.default_pose,
            state_info['previous_action'],
        ])
        observation = jnp.roll(observation_history, observation.size).at[:observation.size].set(observation)

        return observation
    
    def _reward_linear_z_velocity(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_angular_xy_velocity(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torque(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_tracking_linear_velocity(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (x axis)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[0] - local_vel[0]))
        lin_vel_reward = jnp.exp(
            -lin_vel_error / self.kernel_sigma
        )
        return lin_vel_reward

    def _reward_tracking_angular_velocity(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.kernel_sigma)

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - self.default_pose)) * (
            math.normalize(commands[0])[1] < 0.1
        )

    def _reward_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        wheel_position = pipeline_state.site_xpos[self.wheel_site_ids]
        wheel_offset = wheel_position - jnp.repeat(
            pipeline_state.xpos[self.shin_body_ids],
            repeats=2,
            axis=0,
        )
        offset = base.Transform.create(pos=wheel_offset)
        wheel_indices = jnp.repeat(
            (self.shin_body_ids - 1),
            repeats=2,
            axis=0,
        )
        wheel_velocities = offset.vmap().do(pipeline_state.xd.take(wheel_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(wheel_velocities[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)
        
        
envs.register_environment('walter', Walter)

# Test Enviornment:
def main(argv=None):
    # RNG Key:
    key = jax.random.key(0)

    env = Walter()

    state = jax.jit(env.reset)(key)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(key)

    fwd_ctrl = jnp.array([
        0.0, 0.0, 5.0, 5.0,
    ])
    fwd_ctrl = jnp.tile(fwd_ctrl, 4)

    simulation_steps = 500
    state_history = []
    for i in range(simulation_steps):
        print(f"Step: {i}")
        state = step_fn(state, fwd_ctrl)
        state_history.append(state.pipeline_state)

    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=state_history,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
