"""Joystick task for WF_TRON1A wheeled-legged biped.

This environment trains the robot to follow velocity commands using its wheels
for locomotion while maintaining balance with its legs. Unlike walking robots,
the wheeled-legged biped is an inherently unstable system (inverted pendulum)
that must actively balance.

Key design decisions:
- Direct torque control (motor actuators), not position control.
- Actions are raw torques, scaled by action_scale and clipped to actuator limits.
- No gait phase tracking (wheels don't walk).
- Reward encourages balancing upright, tracking velocity commands via wheels,
  and keeping legs in a nominal standing pose.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.wf_tron1a import base as wf_base
from mujoco_playground._src.locomotion.wf_tron1a import wf_tron1a_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=20.0,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking rewards.
              tracking_lin_vel=1.5,
              tracking_ang_vel=0.75,
              # Balance rewards — the most important for this robot.
              upright=5.0,
              ang_vel_xy=-0.05,
              base_height=-1.0,
              lin_vel_z=-0.2,
              # Energy rewards.
              torques=-0.0005,
              action_rate=-0.02,
              energy=-0.0001,
              # Leg pose rewards.
              leg_pose=-0.1,
              dof_pos_limits=-0.5,
              # Other rewards.
              alive=2.0,
              termination=-2.0,
              stand_still=-0.1,
              # Wheel rewards.
              wheel_spin_penalty=-0.0005,
          ),
          tracking_sigma=0.25,
          base_height_target=0.907,
      ),
      push_config=config_dict.create(
          enable=False,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.3, 0.3],
      ang_vel_yaw=[-1.0, 1.0],
      impl="warp",
      naconmax=4 * 4096,
      njmax=40,
  )


class Joystick(wf_base.WFTron1AEnv):
  """Track a joystick command with the wheeled-legged biped."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Joint limits — only for limited joints (legs, not wheels).
    # Build a mask for limited joints.
    self._limited_mask = jp.array([
        bool(self.mj_model.joint(i).limited) for i in range(1, self.mj_model.njnt)
    ])
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Leg joint indices (exclude wheel joints which are unlimited).
    # qpos layout: [0:7] freejoint, [7:15] joint angles
    # Joint order: abad_L(7), hip_L(8), knee_L(9), wheel_L(10),
    #              abad_R(11), hip_R(12), knee_R(13), wheel_R(14)
    self._leg_joint_indices = jp.array([0, 1, 2, 4, 5, 6])  # indices into qpos[7:]
    self._wheel_joint_indices = jp.array([3, 7])  # indices into qpos[7:]

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._wheel_site_ids = np.array(
        [self._mj_model.site(name).id for name in consts.WHEEL_SITES]
    )

    self._floor_geom_id = self._mj_model.geom("floor").id
    self._wheel_L_contact_sensor = self._mj_model.sensor("wheel_L_floor_found").id
    self._wheel_R_contact_sensor = self._mj_model.sensor("wheel_R_floor_found").id

    # Actuator torque limits for action scaling.
    self._ctrl_range = jp.array(self._mj_model.actuator_ctrlrange)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomize x, y position.
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

    # Randomize yaw.
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Slightly randomize leg joint angles (not wheels).
    rng, key = jax.random.split(rng)
    leg_noise = jax.random.uniform(key, (6,), minval=-0.1, maxval=0.1)
    qpos = qpos.at[7].set(qpos[7] + leg_noise[0])   # abad_L
    qpos = qpos.at[8].set(qpos[8] + leg_noise[1])   # hip_L
    qpos = qpos.at[9].set(qpos[9] + leg_noise[2])   # knee_L
    qpos = qpos.at[11].set(qpos[11] + leg_noise[3])  # abad_R
    qpos = qpos.at[12].set(qpos[12] + leg_noise[4])  # hip_R
    qpos = qpos.at[13].set(qpos[13] + leg_noise[5])  # knee_R

    # Small initial velocity perturbation.
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.3, maxval=0.3)
    )

    data = mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax, njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Random push.
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    # action in [-1,1], scale to torque. Clip to actuator limits.
    ctrl = action * self._config.action_scale
    ctrl = jp.clip(ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Update info.
    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action

    # Resample command every 500 steps.
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Terminate if upvector z < 0 (fallen over).
    up_z = self.get_gravity(data)[-1]
    fall_termination = up_z < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    # Joint angles (all 8 joints including wheels).
    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,                                    # 3
        noisy_gyro,                                      # 3
        noisy_gravity,                                   # 3
        info["command"],                                 # 3
        noisy_joint_angles - self._default_pose,         # 8
        noisy_joint_vel,                                 # 8
        info["last_act"],                                # 8
    ])  # Total: 39

    # Privileged state includes noise-free observations.
    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,                                            # 3
        accelerometer,                                   # 3
        gravity,                                         # 3
        linvel,                                          # 3
        global_angvel,                                   # 3
        joint_angles - self._default_pose,               # 8
        joint_vel,                                       # 8
        root_height,                                     # 1
        data.actuator_force,                             # 8
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    linvel = self.get_local_linvel(data)
    angvel = self.get_gyro(data)
    gravity = self.get_gravity(data)

    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], linvel
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], angvel
        ),
        # Balance rewards.
        "upright": self._reward_upright(gravity),
        "ang_vel_xy": self._cost_ang_vel_xy(angvel),
        "base_height": self._cost_base_height(data),
        "lin_vel_z": self._cost_lin_vel_z(linvel),
        # Energy rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        # Leg pose rewards.
        "leg_pose": self._cost_leg_pose(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        # Wheel rewards.
        "wheel_spin_penalty": self._cost_wheel_spin(data.qvel[6:]),
    }

  # --- Tracking rewards ---

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, local_linvel: jax.Array
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_linvel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, local_angvel: jax.Array
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - local_angvel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # --- Balance rewards ---

  def _reward_upright(self, torso_zaxis: jax.Array) -> jax.Array:
    # Reward for being upright. upvector z=1 when perfectly upright.
    # torso_zaxis[-1] is the z component of the up vector.
    return torso_zaxis[-1]

  def _cost_ang_vel_xy(self, local_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(local_angvel[:2]))

  def _cost_base_height(self, data: mjx.Data) -> jax.Array:
    base_height = data.qpos[2]
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  def _cost_lin_vel_z(self, local_linvel: jax.Array) -> jax.Array:
    return jp.square(local_linvel[2])

  # --- Energy rewards ---

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel * qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  # --- Leg pose rewards ---

  def _cost_leg_pose(self, qpos: jax.Array) -> jax.Array:
    # Penalize leg joints deviating from default standing pose.
    # Only leg joints (not wheels).
    leg_angles = qpos[self._leg_joint_indices]
    default_leg = self._default_pose[self._leg_joint_indices]
    return jp.sum(jp.square(leg_angles - default_leg))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Only check limited joints (wheels are unlimited).
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    # Zero out penalties for unlimited joints (wheels).
    out_of_limits = out_of_limits * self._limited_mask
    return jp.sum(out_of_limits)

  # --- Other rewards ---

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    leg_angles = qpos[self._leg_joint_indices]
    default_leg = self._default_pose[self._leg_joint_indices]
    return jp.sum(jp.abs(leg_angles - default_leg)) * (cmd_norm < 0.1)

  # --- Wheel rewards ---

  def _cost_wheel_spin(self, qvel: jax.Array) -> jax.Array:
    # Penalize excessive wheel speeds. qvel[6:] has indices 3=wheel_L, 7=wheel_R.
    wheel_vel_L = qvel[3]
    wheel_vel_R = qvel[7]
    return wheel_vel_L**2 + wheel_vel_R**2

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # 10% chance of zero command (stand still).
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )
