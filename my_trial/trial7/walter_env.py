from typing import Any, List, Sequence
import os
from etils import epath

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from ml_collections import config_dict
import mujoco
import time
import mujoco.viewer

import math as pymath
def get_config():
    """Returns reward config for g1 humanoid environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        tracking_lin_vel=5.0,
                        tracking_ang_vel=3.2,
                        lin_vel_z=-2.0,
                        ang_vel_xy=-0.15,
                        base_height=-0.0,
                        orientation=-5.0,
                        torques=-0.0025,
                        action_rate=-0.1,
                        action = -.1,
                        termination=-100.0,
                        pose=-0.0,
                        shin_speed = 0.0,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.5,
                vel_tracking_sigma=0.01,
                shin_vel_tracking_sigma=0.5,
                base_height_target=0.15,
                lin_vel_x=[-0.5, 0.5],
                lin_vel_y=[-0.0, 0.0],
                ang_vel_yaw=[-0.0, 0.0],
            )
        )
        return default_config

    def get_default_noise_config():
        default_config = config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        hip_pos=0.03,  # rad
                        kfe_pos=0.05,
                        ffe_pos=0.08,
                        faa_pos=0.03,
                        joint_vel=1.5,  # rad/s
                        gravity=0.05,
                        linvel=0.1,
                        gyro=0.2,  # angvel.
                    )
                ),
            )
        )
        return default_config
    
    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
            noises =get_default_noise_config(),
            soft_joint_pos_limit_factor = 0.95,
            ctrl_dt=0.02,
            sim_dt=0.002,
        )
    )

    return default_config

class WalterEnv(PipelineEnv):
    """WALTER environment."""
    def __init__(self, 
                obs_noise: float = 0.05,
                action_scale: float = 0.3,
                force_scale:  float = 16,
                wheel_scale: float = 5,
                kick_vel: float = 0.05,
                **kwargs):
        
        ROOT_PATH = epath.Path('/home/orl/Tianze/mujoco_work/tools_basics/my_trial/trial7')
        filepath = os.path.join(os.path.dirname(__file__), ROOT_PATH/'scene.xml')
        mj_model = mujoco.MjModel.from_xml_path(filepath)
        
        self.sys_config = get_config()
        
        sys = mjcf.load_model(mj_model)
        self._dt = self.sys_config.ctrl_dt
        sys = sys.tree_replace({'opt.timestep': self.sys_config.sim_dt})
        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))

        super().__init__(sys, backend='mjx', n_frames=n_frames)
        
        self._mjx_model = mjx.put_model(sys.mj_model)
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.sys_config.rewards.scales[k[:-6]] = v
                
        # System Parameters
        self._action_scale = action_scale
        self._force_scale = force_scale
        self._wheel_scale = wheel_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._soft_joint_pos_limit_factor = self.sys_config.soft_joint_pos_limit_factor
        
        # Joint Position Info
        self._init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        self._default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        
        # Joint Limits TODO: add limits
        self._lowers, self._uppers = sys.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._soft_joint_pos_limit_factor
        
        self._weights = jnp.array([
            1, 0, 0, 0,  # only track the thigh
            1, 0, 0, 0,  
            0,
            1, 0, 0, 0, 
            1, 0, 0, 0, 
        ])
        self._site_id = sys.mj_model.site("torso_site").id
        self._site_id_head = sys.mj_model.site("head_site").id
        
        GRAVITY_SENSOR = "upvector"
        GYRO_SENSOR = "gyro"
        LOCAL_LINVEL_SENSOR = "local_linvel"
        LOCAL_LINVEL_SENSOR_HEAD = "local_linvel_head"
        
        sensor_id = sys.mj_model.sensor(GRAVITY_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._global_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        sensor_id = sys.mj_model.sensor(GYRO_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._gyro_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(LOCAL_LINVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._linvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(LOCAL_LINVEL_SENSOR_HEAD).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._linvel_vector_head_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        self._nv = sys.nv
        self._nu = sys.nu
        self._nq = sys.nq
        
    def reset(self, rng: jax.Array) -> State:
        qpos = self._init_q
        qvel = jnp.zeros(self._nv)
        ctrl = self._default_ctrl
        
        # Sample Random Initial States
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        # qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]) , yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)
        
        # TODO: Useful to set thigh and knee with different initial angles?
        rng, key = jax.random.split(rng)
        shin_new = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        indices = jnp.array([17, 21])
        # qpos = qpos.at[indices].set(shin_new)
        
        # qpos = qpos.at[7:].set(
        #     qpos[7:] + jax.random.uniform(key, (17,), minval=-.2, maxval=.2)
        # )
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0].set(
            jax.random.uniform(key, (), minval=-0.5, maxval=0.5)
        )
        # qvel = qvel.at[5].set(
        #     jax.random.uniform(key, (), minval=-0.5, maxval=0.5)
        # )
        
        pipeline_state = self.pipeline_init(q=qpos, qd=qvel, ctrl = ctrl)
        rng, cmd_rng = jax.random.split(rng)
        state_info = {
            'rng': rng,
            'step': 0,
            'command': self.sample_command(cmd_rng),
            'last_act': jnp.zeros(self._nu),
            "motor_targets": jnp.zeros(self._nu),
            'last_vel': jnp.zeros(17),
            'rewards': {k: 0.0 for k in self.sys_config.rewards.scales.keys()},
            'kick': jnp.array([0.0, 0.0]),
        }
        
        metrics = {}
        for k in self.sys_config.rewards.scales.keys():
            metrics[f"reward/{k}"] = jnp.zeros(())
        
        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done, metrics, state_info) 

    def step(self, state: State, action: jax.Array) -> State:
        """Steps the environment forward one timestep."""
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)
        # Kick the robot
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval = 2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})
        
        # indices = jnp.arange(0, action.size, 4)
        # action = action.at[indices].set(0)
        # indices = jnp.arange(1, action.size, 4)
        # action = action.at[indices].set(0)
        # physics step
        
        # wheel rolling motion should not be centered at the defeault pos
        # TODO: shin might as well
        m_tth = self._default_pose[0:8:4] + action[0:8:4] * self._action_scale
        m_tsh = action[1:8:4] * self._force_scale
        m_tw1 = action[2:8:4] * self._wheel_scale
        m_tw2 = action[3:8:4] * self._wheel_scale
        
        m_hth = self._default_pose[9::4] + action[8::4] * self._action_scale
        m_hsh = action[9::4] * self._force_scale
        m_hw1 = action[10::4] * self._wheel_scale
        m_hw2 = action[11::4] * self._wheel_scale
        
        motor_targets = jnp.array([m_tth[0], m_tsh[0], m_tw1[0], m_tw2[0],
                                  m_tth[1], m_tsh[1], m_tw1[1], m_tw2[1],
                                  m_hth[0], m_hsh[0], m_hw1[0], m_hw2[0],
                                  m_hth[1], m_hsh[1], m_hw1[1], m_hw2[1]])
        
        
        # motor_targets = self._default_pose + action * self._action_scale
        # motor_targets = jnp.clip(motor_targets, self._soft_lowers, self._soft_uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd
        state.info['motor_targets'] = motor_targets
        
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        
        obs = self._get_obs(pipeline_state, state.info)
        # done
        global_vec = pipeline_state.sensordata[self._global_vector_adr].ravel()
        fall_termination = global_vec[-1] < 0.0
        done = fall_termination | jnp.isnan(pipeline_state.q).any() | jnp.isnan(pipeline_state.qd).any() | (x.pos[0][2] < 0.05) 
        done |= (pipeline_state.site_xpos[self._site_id_head].ravel()[2] < 0.05)
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd, pipeline_state)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'base_height': self._reward_base_height(x, pipeline_state),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'action': self._reward_action(action, state.info['last_act']),
            'termination': self._reward_termination(done, state.info['step']),
            'pose': self._reward_pose(pipeline_state.q[7:]),
            'shin_speed': self._reward_shin(pipeline_state.qd[6:]),
        }

        rewards = {
            k: v * self.sys_config.rewards.scales[k] for k, v in rewards.items()
        }

        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info['command'] = jnp.where(
            (state.info['step'] > 1000), self.sample_command(cmd_rng), state.info['command'])
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 1000), 0, state.info['step']
        )
        
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
            
        done = done.astype(reward.dtype)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(self, pipeline_state: base.State, state_info: dict[str, Any]) -> jax.Array:
        gyro = pipeline_state.sensordata[self._gyro_vector_adr]
        linvel = pipeline_state.sensordata[self._linvel_vector_adr]
        gravity = pipeline_state.site_xmat[self._site_id].T @ jnp.array([0, 0, -1])
        
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # TODO: add noise and priviledged state?
        state = jnp.hstack([
            linvel,  # 3
            gyro,  # 3
            gravity,  # 3
            state_info["command"],  # 3
            joint_angles - self._default_pose,  # 17
            joint_vel,  # 16
            state_info["last_act"],  # 16
        ])
        return state

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        
        lin_vel_x = jax.random.uniform(rng1, minval=self.sys_config.rewards.lin_vel_x[0], maxval=self.sys_config.rewards.lin_vel_x[1])
        lin_vel_y = jax.random.uniform(rng2, minval=self.sys_config.rewards.lin_vel_y[0], maxval=self.sys_config.rewards.lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(rng3, minval=self.sys_config.rewards.ang_vel_yaw[0], maxval=self.sys_config.rewards.ang_vel_yaw[1])
        return jnp.where(jax.random.bernoulli(rng4, p=0.1), jnp.array([0.0, 0.0, 0.0]), jnp.array([lin_vel_x, lin_vel_y, ang_vel_yaw]))

    def _reward_tracking_lin_vel(self, command: jax.Array, x: Transform, xd: Motion, pipeline_state: base.State) -> jax.Array:
        """Tracking the base linear velocity."""
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        head_vel = pipeline_state.sensordata[self._linvel_vector_head_adr]
        torso_vel = pipeline_state.sensordata[self._linvel_vector_adr]
        
        lin_vel_error = jnp.sum(jnp.square(torso_vel[:2] - command[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self.sys_config.rewards.vel_tracking_sigma)
        return lin_vel_reward
    
    def _reward_tracking_ang_vel(self, command: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        """Tracking the base angular velocity."""
        local_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.sum(jnp.square(local_vel[2] - command[2]))
        ang_vel_reward = jnp.exp(-ang_vel_error / self.sys_config.rewards.vel_tracking_sigma)
        return ang_vel_reward
    
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        """Reward for the base linear velocity in z direction."""
        return jnp.square(xd.vel[0][2])
    
    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        """Reward for the base angular velocity in xy direction."""
        return jnp.sum(jnp.square(xd.ang[0, :2]))  # xy angular velocity
    
    def _reward_base_height(self, x: Transform, pipeline_state: base.State) -> jax.Array:
        """Reward for the base height."""
        # TODO: remove base height tracking to base velocity tracking
        base_height = x.pos[0][2]
        head_height = pipeline_state.site_xpos[self._site_id_head].ravel()[2]
        base_height_target = self.sys_config.rewards.base_height_target
        base_height_error = jnp.square(base_height - base_height_target) + jnp.square(head_height - base_height_target)  
        return base_height_error
    
    def _reward_orientation(self, x: Transform) -> jax.Array:
        """Reward for the base orientation."""
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))
    
    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Reward for the joint torques."""
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))
    
    def _reward_action_rate(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        """Reward for the action rate."""
        return jnp.sum(jnp.square(action - last_action))

    def _reward_action(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        """Reward for the action."""
        return jnp.sum(jnp.square(action))
    
    def _reward_termination(self, done: jax.Array, step: int) -> jax.Array:
        """Reward for the termination."""
        return done & (step < 500)
    
    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(qpos - self._default_pose) * self._weights)
    
    def _reward_shin(self, qvel: jax.Array) -> jax.Array:
        """Reward for the shin speed."""
        indices = jnp.array([10, 14])
        shin_error = jnp.sum(jnp.square(qvel[indices] - 0.2))
        shin_vel_reward = jnp.exp(-shin_error / self.sys_config.rewards.shin_vel_tracking_sigma)
        return shin_vel_reward
        
envs.register_environment('Walter', WalterEnv)

if __name__ == '__main__':
    a = get_config()
    # Load the model
    env_name = 'Walter'
    env = envs.get_environment(env_name)
    
    print("envrionment created")

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    print("jax function created")

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    env.reset(subkey)
    state = reset_fn(subkey)

    print("stepping")
    state.info['command'] = jnp.array([0.0, 0.0, 0.0])
    # state = step_fn(state, jnp.zeros(env.sys.nu))
    state = env.step(state, jnp.zeros(env.sys.nu))
    state = step_fn(state, jnp.zeros(env.sys.nu))

    print(state.reward)
    print(state.obs)

    ROOT_PATH = epath.Path('/home/orl/Tianze/mujoco_work/tools_basics/my_trial/trial7')
    filepath = os.path.join(os.path.dirname(__file__), ROOT_PATH/'scene.xml')

    # Load the xml file
    model = mujoco.MjModel.from_xml_path(filepath)
    
    # Update timestep:
    model.opt.timestep = 0.002
    visualization_rate = 0.02
    num_steps = int(visualization_rate / model.opt.timestep)

    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    
    # Update the initial state
    mujoco.mj_forward(model, data)
    
    # Visualize and Simulate the system
    cur_t = 0
    period = 10
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)         

            # Update the viewer
            viewer.sync()
            
            cur_t += model.opt.timestep
            if cur_t > period:
                cur_t = 0

            # Sleep for a bit to visualize the simulation:
            time.sleep(.002)