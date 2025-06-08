from typing import Any, List, Sequence, Union, Mapping, Dict
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

from collision import geoms_colliding

np.set_printoptions(precision=3, suppress=True, linewidth=100)
print('imported succesfully')


def get_config():
    """Returns reward config for g1 humanoid environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1.0,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.75,
                        # Penalize the base linear velocity in x direction.
                        lin_vel_z=-0.1,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.15,
                        # Base height reward
                        base_height=0.0,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-2.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.000,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.0,
                        # Encourage no motion at zero command, L2 regularization
                        stand_still=-1.0,
                        # Early termination penalty.
                        termination=-100.0,
                        contact_force=-0.01,
                        collision=-0.1,
                        # Pose related rewards.
                        joint_deviation_knee=-0.1,
                        joint_deviation_hip=-0.25,
                        dof_pos_limits=-1.0,
                        pose=-0.1,
                        # Feet related terms
                        feet_air_time =2.0,
                        feet_slip=-0.25,
                        feet_phase=1.0,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
                base_height_target=0.5,
                max_foot_height=0.15,
                max_contact_force=500.0,
                lin_vel_x=[-1.0, 1.0],
                lin_vel_y=[-1.0, 1.0],
                ang_vel_yaw=[-1.0, 1.0],
            )
        )
        return default_config

    def get_default_noise_config():
        default_config = config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        joint_pos=0.03,  # rad
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


class G1Env(PipelineEnv):
    """ G1 Environment """
    def __init__(self, 
            obs_noise: float = 0.05,
            action_scale: float = 0.5,
            kick_vel: float = 0.05,
            **kwargs):
        
        ROOT_PATH = epath.Path('/home/orl/Tianze/mujoco_work/tools_basics/lessons/mujoco_basics/lesson_4/digit')
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
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._soft_joint_pos_limit_factor = self.sys_config.soft_joint_pos_limit_factor

        # Joint Position Info
        self._init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        self._default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        
        # Joint Limits
        self._lowers, self._uppers = sys.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._soft_joint_pos_limit_factor
        
        # Get all the indices for later use
        waist_indices = []
        waist_joint_names = [
            "waist_yaw",
            "waist_roll",
            "waist_pitch",
        ]
        for joint_name in waist_joint_names:
            waist_indices.append(
                sys.mj_model.joint(f"{joint_name}_joint").qposadr - 7
        )
        self._waist_indices = jnp.array(waist_indices)
        

        arm_indices = []
        arm_joint_names = [
            "shoulder_roll",
            "shoulder_yaw",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
        for side in ["left", "right"]:
            for joint_name in arm_joint_names:
                arm_indices.append(
                    sys.mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
                )
        self._arm_indices = jnp.array(arm_indices)

        hip_indices = []
        hip_joint_names = [
            "hip_roll",
            "hip_yaw",
        ]
        for side in ["left", "right"]:
            for joint_name in hip_joint_names:
                hip_indices.append(
                    sys.mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
            )
        self._hip_indices = jnp.array(hip_indices)

        knee_indices = []
        knee_joint_names = ["knee"]
        for side in ["left", "right"]:
            for joint_name in knee_joint_names:
                knee_indices.append(
                    sys.mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
            )
        self._knee_indices = jnp.array(knee_indices)
        
        # fmt: off
        self._weights = jnp.array([
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # left leg.
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # right leg.
            1.0, 1.0, 1.0,  # waist.
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm.
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm.
        ])
        # fmt: on
        
        self._torso_body_id = sys.mj_model.body('torso_link').id
        self._torso_mass = sys.mj_model.body_subtreemass[self._torso_body_id]
        self._torso_imu_site_id = sys.mj_model.site("imu_in_torso").id
        self._pelvis_imu_site_id = sys.mj_model.site("imu_in_pelvis").id
        
        
        # Register ID
        FEET_SITES = [
            "left_foot",
            "right_foot",
        ]

        HAND_SITES = [
            "left_palm",
            "right_palm",
        ]
                
        LEFT_FEET_GEOMS = [
            "left_foot",
        ]
        RIGHT_FEET_GEOMS = [
            "right_foot",
        ]
        FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
        
        TORSO_GRAVITY_SENSOR = "upvector_torso"
        TORSO_GYRO_SENSOR = "gyro_torso"
        TORSO_LOCAL_LINVEL_SENSOR = "local_linvel_torso"
        
        PELVIS_GRAVITY_SENSOR = "upvector_pelvis"
        PELVIS_GYRO_SENSOR = "gyro_pelvis"
        PELVIS_LOCAL_LINVEL_SENSOR = "local_linvel_pelvis"
        
        TORSO_GLOBAL_LINVEL_SENSOR = "global_linvel_torso"
        PELVIS_GLOBAL_LINVEL_SENSOR = "global_linvel_pelvis"
        TORSO_GLOBAL_ANGVEL_SENSOR = "global_angvel_torso"
        PELVIS_GLOBAL_ANGVEL_SENSOR = "global_angvel_pelvis"
        
        LFORCE_SENSOR = 'left_foot_force'
        RFORCE_SENSOR = 'right_foot_force'
        
        # Site for tracking position
        self._feet_site_id = np.array(
            [sys.mj_model.site(name).id for name in FEET_SITES]
        )
        self._hand_site_id = np.array(
            [sys.mj_model.site(name).id for name in HAND_SITES]
        )  
        self._floor_geom_id = sys.mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [sys.mj_model.geom(name).id for name in FEET_GEOMS]
        )
        
        # Sensor Address for tracking velocities and other properties
        foot_linvel_sensor_adr = []
        for site in FEET_SITES:
            sensor_id = sys.mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = sys.mj_model.sensor_adr[sensor_id]
            sensor_dim = sys.mj_model.sensor_dim[sensor_id]  
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jnp.array(foot_linvel_sensor_adr)

        sensor_id = sys.mj_model.sensor(TORSO_GRAVITY_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._torso_global_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(TORSO_GYRO_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._torso_gyro_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        sensor_id = sys.mj_model.sensor(TORSO_LOCAL_LINVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._torso_linvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(PELVIS_GRAVITY_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._pelvis_global_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(PELVIS_GYRO_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._pelvis_gyro_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        sensor_id = sys.mj_model.sensor(PELVIS_LOCAL_LINVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._pelvis_linvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))


        sensor_id = sys.mj_model.sensor(TORSO_GLOBAL_LINVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._torso_global_linvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(PELVIS_GLOBAL_LINVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._pelvis_global_linvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))

        sensor_id = sys.mj_model.sensor(TORSO_GLOBAL_ANGVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._torso_global_angvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        sensor_id = sys.mj_model.sensor(PELVIS_GLOBAL_ANGVEL_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._pelvis_global_angvel_vector_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        sensor_id = sys.mj_model.sensor(LFORCE_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._lforce_sensor_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        sensor_id = sys.mj_model.sensor(RFORCE_SENSOR).id
        sensor_adr = sys.mj_model.sensor_adr[sensor_id]
        sensor_dim = sys.mj_model.sensor_dim[sensor_id]
        
        self._rforce_sensor_adr = jnp.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        # Self collision
        self._left_hand_geom_id = sys.mj_model.geom("left_hand_collision").id
        self._right_hand_geom_id = sys.mj_model.geom("right_hand_collision").id
        self._left_foot_geom_id = sys.mj_model.geom("left_foot").id
        self._right_foot_geom_id = sys.mj_model.geom("right_foot").id
        self._left_shin_geom_id = sys.mj_model.geom("left_shin").id
        self._right_shin_geom_id = sys.mj_model.geom("right_shin").id
        self._left_thigh_geom_id = sys.mj_model.geom("left_thigh").id
        self._right_thigh_geom_id = sys.mj_model.geom("right_thigh").id
        
        # System Dimensions
        self._nv = sys.nv
        self._nu = sys.nu
        self._nq = sys.nq
        
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        qpos = self._init_q
        qvel = jnp.zeros(self._nv)
        ctrl = self._default_ctrl

        # Sample Random Initial States
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.3, maxval=0.3)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]) , yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)
        
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
        )

        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        # Phase, freq=U(1.0, 1.5)
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        phase_dt = 2 * jnp.pi * self.dt * gait_freq
        phase = jnp.array([0, jnp.pi])
        
        # Set up the initial conditions
        pipeline_state = self.pipeline_init(q=qpos, qd=qvel, ctrl = ctrl)
        
        rng, cmd_rng = jax.random.split(rng)
        
        state_info = {
            'rng': rng,
            'step': 0,
            'command': self.sample_command(cmd_rng),
            'last_act': jnp.zeros(self._nu),
            "motor_targets": jnp.zeros(self._nu),
            'feet_air_time': jnp.zeros(2),
            'last_contact': jnp.zeros(2, dtype=bool),
            'last_vel': jnp.zeros(self._nu),
            'rewards': {k: 0.0 for k in self.sys_config.rewards.scales.keys()},
            'kick': jnp.array([0.0, 0.0]),
            'swing_peak': jnp.zeros(2),
            # Phase related.
            'phase_dt': phase_dt,
            'phase': phase,
        }
        
        metrics = {}
        for k in self.sys_config.rewards.scales.keys():
            metrics[f"reward/{k}"] = jnp.zeros(())

        contact = jnp.array([
            geoms_colliding(pipeline_state, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        
        obs = self._get_obs(pipeline_state, state_info, contact)
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done, metrics, state_info) 
    
    def step(self, state: State, action: jax.Array) -> State:
        """Steps the environment forward one timestep."""
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)
        
        # Kick the robot
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})
        
        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(motor_targets, self._soft_lowers, self._soft_uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd
        state.info['motor_targets'] = motor_targets
        
        # Get the joint angles and velocities
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        
        # Foot Contact
        contact = jnp.array([
            geoms_colliding(pipeline_state, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = pipeline_state.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jnp.maximum(state.info["swing_peak"], p_fz)
        
        # Get observation
        obs = self._get_obs(pipeline_state, state.info, contact)
        
        # done (Fail, self-collision, NaN)
        global_vec = pipeline_state.sensordata[self._torso_global_vector_adr].ravel()
        fall_termination = global_vec[-1] < 0.0
        contact_termination = geoms_colliding(pipeline_state, self._right_foot_geom_id, self._left_foot_geom_id)
        contact_termination |= geoms_colliding(pipeline_state, self._right_foot_geom_id, self._left_shin_geom_id)
        contact_termination |= geoms_colliding(pipeline_state, self._left_foot_geom_id, self._right_shin_geom_id)
        
        done = fall_termination | contact_termination | jnp.isnan(pipeline_state.q).any() | jnp.isnan(pipeline_state.qd).any()

        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], pipeline_state)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], pipeline_state)
            ),
            'lin_vel_z': self._reward_lin_vel_z(pipeline_state),
            'ang_vel_xy': self._reward_ang_vel_xy(pipeline_state),
            'base_height': self._reward_base_height(pipeline_state),
            'orientation': self._reward_orientation(pipeline_state),
            
            # pytype: disable=attribute-error
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'termination': self._reward_termination(done, state.info['step']),
            'contact_force': self._reward_contact_force(pipeline_state),
            'collision': self._reward_collision(pipeline_state),
            'joint_deviation_knee': self._reward_joint_deviation_knee(pipeline_state.q[7:]),
            'joint_deviation_hip': self._reward_joint_deviation_hip(pipeline_state.q[7:], state.info['command']),
            'dof_pos_limits': self._reward_joint_pos_limits(pipeline_state.q[7:]),
            'pose': self._reward_pose(pipeline_state.q[7:]),
            
            # Swing feet related
            'feet_air_time': self._reward_feet_air_time(state.info['feet_air_time'], first_contact, state.info['command']),
            'feet_slip': self._cost_feet_slip(pipeline_state, contact, state.info),
            'feet_phase': self._reward_feet_phase(pipeline_state, state.info['phase'], self.sys_config.rewards.max_foot_height, state.info['command']),
        }

        rewards = {
            k: v * self.sys_config.rewards.scales[k] for k, v in rewards.items()
        }

        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
 
        state.info['command'] = jnp.where(
            (state.info['step'] > 500), self.sample_command(cmd_rng), state.info['command'])
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        state.info['step'] = jnp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )
        
        state.info["swing_peak"] *= ~contact

        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
            
        done = done.astype(reward.dtype)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(self, pipeline_state: base.State, state_info: dict[str, Any], contact: jax.Array) -> Union[jax.Array, Mapping[str, jax.Array]]:
        """Returns the observation."""
        
        gyro = pipeline_state.sensordata[self._pelvis_gyro_vector_adr]
        linvel = pipeline_state.sensordata[self._pelvis_linvel_vector_adr]
        gravity = pipeline_state.site_xmat[self._pelvis_imu_site_id].T @ jnp.array([0, 0, -1])
        
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        
        cos = jnp.cos(state_info["phase"])
        sin = jnp.sin(state_info["phase"])
        phase = jnp.concatenate([cos, sin])
        
        feet_vel = pipeline_state.sensordata[self._foot_linvel_sensor_adr].ravel()
        global_angvel = pipeline_state.sensordata[self._pelvis_global_angvel_vector_adr]
        
        state = jnp.hstack([
            linvel,  # 3
            gyro,  # 3
            gravity,  # 3
            state_info["command"],  # 3
            joint_angles - self._default_pose,  # 29
            joint_vel,  # 29
            state_info["last_act"],  # 29
            phase, # 4
        ])
        
        privileged_state = jnp.hstack([
            state,
            global_angvel,
            pipeline_state.qpos[2],
            pipeline_state.actuator_force,
            contact,
            feet_vel,
            state_info['feet_air_time'],
        ])
        
        return  {
            'state': state,
            'privileged_state': privileged_state,
        }
    
    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        
        lin_vel_x = jax.random.uniform(rng1, minval=self.sys_config.rewards.lin_vel_x[0], maxval=self.sys_config.rewards.lin_vel_x[1])
        lin_vel_y = jax.random.uniform(rng2, minval=self.sys_config.rewards.lin_vel_y[0], maxval=self.sys_config.rewards.lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(rng3, minval=self.sys_config.rewards.ang_vel_yaw[0], maxval=self.sys_config.rewards.ang_vel_yaw[1])
        return jnp.where(jax.random.bernoulli(rng4, p=0.1), jnp.array([0.0, 0.0, 0.0]), jnp.array([lin_vel_x, lin_vel_y, ang_vel_yaw]))

    def _reward_contact_force(self, data: base.State) -> jax.Array:
        l_contact_force = data.sensordata[self._lforce_sensor_adr]
        r_contact_force = data.sensordata[self._rforce_sensor_adr]
        
        cost = jnp.clip(
            jnp.abs(l_contact_force[2])
            - self.sys_config.rewards.max_contact_force,
            min=0.0,
        )
        cost += jnp.clip(
            jnp.abs(r_contact_force[2])
            - self.sys_config.rewards.max_contact_force,
            min=0.0,
        )
        return cost

    def _reward_collision(self, data: base.State) -> jax.Array:
        c = geoms_colliding(data, self._left_hand_geom_id, self._left_thigh_geom_id)
        c |= geoms_colliding(data, self._right_hand_geom_id, self._right_thigh_geom_id)
        return jnp.any(c)
    
    def _reward_tracking_lin_vel(self, command: jax.Array, data: base.State) -> jax.Array:
        """Tracking the base linear velocity."""
        local_vel = data.sensordata[self._pelvis_linvel_vector_adr]
        lin_vel_error = jnp.sum(jnp.square(local_vel[:2] - command[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self.sys_config.rewards.tracking_sigma)
        return lin_vel_reward
    
    def _reward_tracking_ang_vel(self, command: jax.Array, data: base.State) -> jax.Array:
        """Tracking the base angular velocity."""
        local_vel = data.sensordata[self._pelvis_gyro_vector_adr]
        ang_vel_error = jnp.sum(jnp.square(local_vel[2] - command[2]))
        ang_vel_reward = jnp.exp(-ang_vel_error / self.sys_config.rewards.tracking_sigma)
        return ang_vel_reward
    
    def _reward_lin_vel_z(self, data: base.State) -> jax.Array:
        """Reward for the base linear velocity in z direction."""
        return jnp.square(data.sensordata[self._pelvis_global_linvel_vector_adr][2]) + jnp.square(data.sensordata[self._torso_global_linvel_vector_adr][2])
    
    def _reward_ang_vel_xy(self, data: base.State) -> jax.Array:
        """Reward for the base angular velocity in xy direction."""
        return jnp.sum(data.sensordata[self._torso_global_angvel_vector_adr][:2])  # xy angular velocity
    
    def _reward_base_height(self, data: base.State) -> jax.Array:
        """Reward for the base height."""
        base_height = data.qpos[2]
        base_height_target = self.sys_config.rewards.base_height_target
        base_height_error = jnp.square(base_height - base_height_target)   
        return base_height_error
    
    def _reward_orientation(self, data: base.State) -> jax.Array:
        """Reward for the base orientation."""
        torso_zaxis = data.sensordata[self._torso_global_vector_adr]
        return jnp.sum(jnp.square(torso_zaxis - jnp.array([0.073, 0.0, 1.0])))
    
    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Reward for the joint torques."""
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))
    
    def _reward_action_rate(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        """Reward for the action rate."""
        return jnp.sum(jnp.square(action - last_action))
    
    def _reward_stand_still(self, command: jax.Array, joint_angles: jax.Array) -> jax.Array:
        """Reward for standing still with small vel commands."""
        return jnp.sum(jnp.abs(joint_angles - self._default_pose)) * (
            math.normalize(command[:2])[1] < 0.1)
        
    def _reward_termination(self, done: jax.Array, step: int) -> jax.Array:
        """Reward for the termination."""
        # return done & (step < 500)
        return done
    
    def _reward_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
        return jnp.sum(
            jnp.abs(
                qpos[self._knee_indices] - self._default_pose[self._knee_indices]
            )
        )
    
    def _reward_joint_deviation_hip(
        self, qpos: jax.Array, cmd: jax.Array
    ) -> jax.Array:
        error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
        # Allow roll deviation when lateral velocity is high.
        weight = jnp.where(
            cmd[1] > 0.1,
            jnp.array([0.0, 1.0, 0.0, 1.0]),
            jnp.array([1.0, 1.0, 1.0, 1.0]),
        )
        cost = jnp.sum(jnp.abs(error) * weight)
        # cost = jnp.sum(
        #     jnp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
        # )
        # cost *= jnp.abs(cmd[1]) > 0.1
        return cost
    
    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        # return jnp.sum(jnp.square(qpos - self._default_pose) * self._weights)
        return jnp.sum(jnp.square(qpos - self._default_pose))
    
    def _reward_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jnp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jnp.clip(qpos - self._soft_uppers, 0.0, None)
        return jnp.sum(out_of_limits)

    def _reward_feet_air_time(self, feet_air_time: jax.Array, first_contact: jax.Array, command: jax.Array, th_min: float=0.2, th_max:float=0.5) -> jax.Array:
        cmd_norm = jnp.linalg.norm(command)
        air_time = (feet_air_time - th_min) * first_contact
        air_time = jnp.clip(air_time, max=th_max - th_min)
        reward = jnp.sum(air_time)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward
        
    def _cost_feet_slip(self, data: base.State, contact: jax.Array, state_info: dict[str, Any]) -> jax.Array:
        """Penalty for the feet slip."""
        """Maybe because of estimating the foot data is not a good idea???"""
        # body_vel = data.sensordata[self._global_linvel_vector_adr][:2]
        # reward = jnp.sum(jnp.linalg.norm(body_vel, axis=-1) * contact)
        
        body_vel = data.sensordata[self._foot_linvel_sensor_adr][:, :2]
        reward = jnp.sum(jnp.linalg.norm(body_vel, axis=-1) * contact)
        return reward
    
    def _reward_feet_phase(self, data: base.State, phase: jax.Array, foot_height: jax.Array, command: jax.Array) -> jax.Array:
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        def cubic_bezier_interpolation(y_start, y_end, x):
            y_diff = y_end - y_start
            bezier = x**3 + 3 * (x**2 * (1 - x))
            return y_start + y_diff * bezier
        
        x = (phase + jnp.pi) / (2 * jnp.pi)
        # stance = cubic_bezier_interpolation(0, foot_height, 2 * x)
        # swing = cubic_bezier_interpolation(foot_height, 0, 2 * x - 1)
        # rz = jnp.where(x <= 0.5, stance, swing)
        
        rz = self.bezier_phase_ramp(x, 0.2) * foot_height
        error = jnp.sum(jnp.square(foot_z - rz))
        reward = jnp.exp(-error / 0.01)
        return reward
    
    def bezier_phase_ramp(self, x, t_start = 0.5):    
        t_end = 1.0
        t_mid = (t_start + t_end) / 2.0
        
        # Normalize t1 and t2 to [0, 1] for respective intervals
        t1 = (x - t_start) / (t_mid - t_start)
        t2 = (x - t_mid) / (t_end - t_mid)
        
        # Clamp values between 0 and 1
        t1 = jnp.clip(t1, 0, 1)
        t2 = jnp.clip(t2, 0, 1)

        ramp_up = 3*t1**2 - 2*t1**3
        ramp_down = 1 - (3*t2**2 - 2*t2**3)

        # Smooth blending using masks (no ifs)
        y = ramp_up * ((x > t_start) & (x <= t_mid)) + ramp_down * (x > t_mid)
        return y
    
envs.register_environment('G1', G1Env)



if __name__ == '__main__':
    a = get_config()
    # Load the model
    env_name = 'G1'
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

    print(state.reward)
    print(state.obs)
    print("created")
    time.sleep(111)