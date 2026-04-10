# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Rotate-z with leap hand and touch sensors."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.leap_hand import base as leap_hand_base
from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts

_NUM_TOUCH = len(consts.TOUCH_SENSOR_NAMES)  # 20
_NUM_PALM_TOUCH = 8
_STATE_DIM = 32 + _NUM_TOUCH + _NUM_TOUCH  # 72: legacy constant, not used in reset()

TOUCH_SENSOR_NOISE_PCT_LEVELS = (0, 10, 15, 20, 25, 50)


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.6,
      action_repeat=1,
      episode_length=500,
      early_termination=True,
      history_len=1,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              angvel=1.0,
              linvel=0.0,
              pose=0.0,
              torques=0.0,
              energy=0.0,
              termination=-100.0,
              action_rate=0.0,
          ),
      ),
      impl='warp',
      naconmax=30 * 8192,
      njmax=160,
  )


class CubeRotateZAxisTouch(leap_hand_base.LeapHandEnv):
  """Rotate a cube around the z-axis with touch sensor observations."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      fixed_mask: Optional[jax.Array] = None,
      touch_sensor_noise_prob: float = 0.0,
      finger_noise_prob: float = 0.0,  # Per-step dropout for finger/thumb sensors (indices 8-19)
      non_palm_touch_threshold: float = 10.0,
  ):
    self._fixed_mask = fixed_mask
    self._touch_sensor_noise_prob = touch_sensor_noise_prob
    self._finger_noise_prob = finger_noise_prob
    self._non_palm_touch_threshold = non_palm_touch_threshold
    super().__init__(
        xml_path=consts.CUBE_TOUCH_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

    # Static numpy array of active sensor indices (shape known at construction time).
    if fixed_mask is not None:
      self._active_touch_indices = np.where(np.array(fixed_mask))[0]
    else:
      self._active_touch_indices = np.arange(_NUM_TOUCH)

    # Instance-level state dim: 32 (joints+act) + K (active touch channels).
    self._state_dim = 32 + len(self._active_touch_indices)

    # Per-active-sensor noise probability vector (shape: (K,)).
    # Full 20-sensor array: palm (0-7) get touch_sensor_noise_prob, finger/thumb (8-19) get finger_noise_prob.
    # Indexed by _active_touch_indices so shape matches active_touch in _get_obs.
    _full_noise_probs = np.concatenate([
        np.full(_NUM_PALM_TOUCH, touch_sensor_noise_prob),   # palm sensors 0-7
        np.full(12, max(touch_sensor_noise_prob, finger_noise_prob)),  # finger/thumb 8-19
    ])
    self._per_active_sensor_noise_probs = jp.array(
        _full_noise_probs[self._active_touch_indices]
    )

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id

    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)
    self._default_pose = self._init_q[self._hand_qids]
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize hand qpos and qvel.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize cube qpos and qvel.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )
    start_quat = leap_hand_base.uniform_quat(quat_rng)
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=q_hand,
        mocap_pos=jp.array([-100.0, -100.0, -100.0]),
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_cube_angvel": jp.zeros(3),
        "episode_touch_mask": self.generate_episode_touch_mask(
            info={"rng": rng},
            fixed_mask=self._fixed_mask,
        ),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(self._config.history_len * self._state_dim)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    obs = self._get_obs(data, state.info, state.obs["state"])
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_cube_angvel"] = self.get_cube_angvel(data)
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_cube_position(data)[2] < -0.05
    return fall_termination

  def _get_touch_sensors_thresholded(self, data: mjx.Data) -> jax.Array:
    """Get touch sensor data with stricter thresholding on non-palm sensors."""
    touch = jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, name)
        for name in consts.TOUCH_SENSOR_NAMES
    ])
    palm_touch = (touch[:_NUM_PALM_TOUCH] > 0.0).astype(jp.float32)
    non_palm_touch = (touch[_NUM_PALM_TOUCH:] > self._non_palm_touch_threshold).astype(
        jp.float32
    )
    return jp.concatenate([palm_touch, non_palm_touch])

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
  ) -> Dict[str, jax.Array]:
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    # Touch sensors: select only the K active channels by static index.
    touch_data = self._get_touch_sensors_thresholded(data)
    active_touch = touch_data[self._active_touch_indices]  # K dims, static shape

    # Per-step i.i.d. Bernoulli dropout: zero each active reading with prob p.
    # Uses per-sensor probabilities: palm (0-7) at touch_sensor_noise_prob,
    # finger/thumb (8-19) at finger_noise_prob (pre-indexed to active sensors).
    if self._touch_sensor_noise_prob > 0.0 or self._finger_noise_prob > 0.0:
      info["rng"], noise_rng = jax.random.split(info["rng"])
      keep = (1.0 - jax.random.bernoulli(
          noise_rng, self._per_active_sensor_noise_probs, active_touch.shape
      ).astype(jp.float32))
      active_touch = active_touch * keep

    state = jp.concatenate([
        noisy_joint_angles,  # 16
        info["last_act"],  # 16
        active_touch,  # K
    ])  # 32 + K
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_quat = self.get_cube_orientation(data)
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    fingertip_positions = self.get_fingertip_positions(data)
    joint_torques = data.actuator_force

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        joint_torques,
        fingertip_positions,
        cube_pos_error,
        cube_quat,
        cube_angvel,
        cube_linvel,
    ])

    return {
        "state": obs_history,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    return {
        "angvel": self._reward_angvel(cube_angvel, cube_pos_error),
        "linvel": self._cost_linvel(cube_linvel),
        "termination": done,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "pose": self._cost_pose(data.qpos[self._hand_qids]),
        "torques": self._cost_torques(data.actuator_force),
        "energy": self._cost_energy(
            data.qvel[self._hand_dqids], data.actuator_force
        ),
    }

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
    return jp.linalg.norm(cube_linvel, ord=1, axis=-1)

  def _reward_angvel(
      self, cube_angvel: jax.Array, cube_pos_error: jax.Array
  ) -> jax.Array:
    del cube_pos_error
    return cube_angvel @ jp.array([0.0, 0.0, 1.0])

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_angles - self._default_pose))


def domain_randomize(
    model: mjx.Model, rng: jax.Array, dr_config: Optional[Dict[str, Any]] = None
):
  """Applies domain randomization with optional per-parameter ranges.

  Keys in dr_config should map to {"min": float, "max": float}.
  If a key is absent, the built-in defaults are used.
  """
  dr_config = dr_config or {}

  def _range(key: str, default_min: float, default_max: float) -> tuple[float, float]:
    cfg = dr_config.get(key)
    if cfg is None:
      return default_min, default_max
    if not isinstance(cfg, dict):
      raise ValueError(f"DR range for '{key}' must be an object with min/max.")
    minval = float(cfg.get("min", default_min))
    maxval = float(cfg.get("max", default_max))
    if minval > maxval:
      raise ValueError(
          f"Invalid DR range for '{key}': min ({minval}) > max ({maxval})."
      )
    return minval, maxval

  def _optional_range(key: str) -> Optional[tuple[float, float]]:
    cfg = dr_config.get(key)
    if cfg is None:
      return None
    if not isinstance(cfg, dict):
      raise ValueError(f"DR range for '{key}' must be an object with min/max.")
    if "min" not in cfg or "max" not in cfg:
      raise ValueError(f"DR range for '{key}' must contain 'min' and 'max'.")
    minval = float(cfg["min"])
    maxval = float(cfg["max"])
    if minval > maxval:
      raise ValueError(
          f"Invalid DR range for '{key}': min ({minval}) > max ({maxval})."
      )
    return minval, maxval

  fingertip_friction_min, fingertip_friction_max = _range(
      "geom_friction_fingertips", 0.5, 1.0
  )
  cube_inertia_scale_min, cube_inertia_scale_max = _range(
      "cube_inertia_scale", 0.8, 1.2
  )
  cube_ipos_add_min, cube_ipos_add_max = _range("cube_ipos_add", -5e-3, 5e-3)
  hand_qpos0_add_min, hand_qpos0_add_max = _range("hand_qpos0_add", -0.05, 0.05)
  hand_frictionloss_scale_min, hand_frictionloss_scale_max = _range(
      "hand_frictionloss_scale", 0.5, 2.0
  )
  hand_armature_scale_min, hand_armature_scale_max = _range(
      "hand_armature_scale", 1.0, 1.05
  )
  hand_mass_scale_min, hand_mass_scale_max = _range("hand_mass_scale", 0.9, 1.1)
  actuator_kp_scale_min, actuator_kp_scale_max = _range(
      "actuator_kp_scale", 0.8, 1.2
  )
  hand_damping_scale_min, hand_damping_scale_max = _range(
      "hand_damping_scale", 0.8, 1.2
  )

  hand_geom_friction_sliding = _optional_range("hand_geom_friction_sliding")
  hand_geom_friction_torsional = _optional_range("hand_geom_friction_torsional")
  hand_geom_friction_rolling = _optional_range("hand_geom_friction_rolling")

  mj_model = CubeRotateZAxisTouch().mj_model
  cube_geom_id = mj_model.geom("cube").id
  cube_body_id = mj_model.body("cube").id
  hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
  hand_body_names = [
      "palm",
      "if_bs",
      "if_px",
      "if_md",
      "if_ds",
      "mf_bs",
      "mf_px",
      "mf_md",
      "mf_ds",
      "rf_bs",
      "rf_px",
      "rf_md",
      "rf_ds",
      "th_mp",
      "th_bs",
      "th_px",
      "th_ds",
  ]
  hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
  hand_geom_ids = np.where(np.isin(mj_model.geom_bodyid, hand_body_ids))[0]
  fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
  fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

  @jax.vmap
  def rand(rng):
    geom_friction = model.geom_friction

    # Optional override: set Leap hand geom friction triplet for all hand geoms.
    # MuJoCo geom_friction = [sliding, torsional, rolling].
    if hand_geom_friction_sliding is not None:
      rng, key = jax.random.split(rng)
      mu = jax.random.uniform(
          key, (1,), minval=hand_geom_friction_sliding[0], maxval=hand_geom_friction_sliding[1]
      )
      geom_friction = geom_friction.at[hand_geom_ids, 0].set(mu)
    if hand_geom_friction_torsional is not None:
      rng, key = jax.random.split(rng)
      mu = jax.random.uniform(
          key,
          (1,),
          minval=hand_geom_friction_torsional[0],
          maxval=hand_geom_friction_torsional[1],
      )
      geom_friction = geom_friction.at[hand_geom_ids, 1].set(mu)
    if hand_geom_friction_rolling is not None:
      rng, key = jax.random.split(rng)
      mu = jax.random.uniform(
          key, (1,), minval=hand_geom_friction_rolling[0], maxval=hand_geom_friction_rolling[1]
      )
      geom_friction = geom_friction.at[hand_geom_ids, 2].set(mu)

    rng, key = jax.random.split(rng)
    fingertip_friction = jax.random.uniform(
        key,
        (1,),
        minval=fingertip_friction_min,
        maxval=fingertip_friction_max,
    )
    geom_friction = geom_friction.at[fingertip_geom_ids, 0].set(fingertip_friction)

    rng, key1, key2 = jax.random.split(rng, 3)
    dmass = jax.random.uniform(
        key1, minval=cube_inertia_scale_min, maxval=cube_inertia_scale_max
    )
    body_inertia = model.body_inertia.at[cube_body_id].set(
        model.body_inertia[cube_body_id] * dmass
    )
    dpos = jax.random.uniform(
        key2, (3,), minval=cube_ipos_add_min, maxval=cube_ipos_add_max
    )
    body_ipos = model.body_ipos.at[cube_body_id].set(
        model.body_ipos[cube_body_id] + dpos
    )

    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
        qpos0[hand_qids]
        + jax.random.uniform(
            key,
            shape=(16,),
            minval=hand_qpos0_add_min,
            maxval=hand_qpos0_add_max,
        )
    )

    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key,
        shape=(16,),
        minval=hand_frictionloss_scale_min,
        maxval=hand_frictionloss_scale_max,
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key,
        shape=(16,),
        minval=hand_armature_scale_min,
        maxval=hand_armature_scale_max,
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key,
        shape=(len(hand_body_ids),),
        minval=hand_mass_scale_min,
        maxval=hand_mass_scale_max,
    )
    body_mass = model.body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass
    )

    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key,
        (model.nu,),
        minval=actuator_kp_scale_min,
        maxval=actuator_kp_scale_max,
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    rng, key = jax.random.split(rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
        key,
        (16,),
        minval=hand_damping_scale_min,
        maxval=hand_damping_scale_max,
    )
    dof_damping = model.dof_damping.at[hand_qids].set(kd)

    return (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_friction,
      body_mass,
      body_inertia,
      body_ipos,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "body_ipos": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "body_ipos": body_ipos,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes


def create_mask_class(
    class_name: str,
    fixed_mask: jax.Array,
    touch_sensor_noise_prob: float = 0.0,
    finger_noise_prob: float = 0.0,
):
  """Create a CubeRotateZAxisTouch subclass with a baked-in fixed mask."""

  def __init__(
      self,
      config=default_config(),
      config_overrides=None,
      **kwargs,
  ):
    CubeRotateZAxisTouch.__init__(
        self,
        config=config,
        config_overrides=config_overrides,
        fixed_mask=fixed_mask,
        touch_sensor_noise_prob=touch_sensor_noise_prob,
        finger_noise_prob=finger_noise_prob,
    )

  noise_str = (f" and {touch_sensor_noise_prob * 100:.0f}% sensor dropout."
               if touch_sensor_noise_prob > 0.0 else ".")
  finger_str = (f" and {finger_noise_prob * 100:.0f}% finger sensor dropout."
                if finger_noise_prob > 0.0 else "")
  cls = type(class_name, (CubeRotateZAxisTouch,), {
      "__init__": __init__,
      "__doc__": f"Rotate z-axis with touch sensors using fixed mask{noise_str}{finger_str}",
      "__module__": __name__,
  })
  return cls


def _create_touch_mask_classes(mask_path: str):
  """Load masks from JSON and create CubeRotateZAxisTouchMask1..N classes,
  plus noise-level variants CubeRotateZAxisTouchMask{N}Noise{PCT}."""
  all_masks, mask_names = leap_hand_base.load_fixed_masks(mask_path)
  globals_dict = globals()
  for idx, mask in enumerate(all_masks):
    mask_n = idx + 1
    # Base mask class (no noise).
    class_name = f"CubeRotateZAxisTouchMask{mask_n}"
    globals_dict[class_name] = create_mask_class(class_name, mask, 0.0)

    # Noise-level variants.
    for pct in TOUCH_SENSOR_NOISE_PCT_LEVELS:
      if pct == 0:
        continue
      noise_class_name = f"CubeRotateZAxisTouchMask{mask_n}Noise{pct}"
      globals_dict[noise_class_name] = create_mask_class(
          noise_class_name, mask, pct / 100.0
      )

    # FingerNoise variant: palm (0-4) 0% noise, finger/thumb (5-19) 95% noise.
    finger_noise_class_name = f"CubeRotateZAxisTouchMask{mask_n}FingerNoise"
    globals_dict[finger_noise_class_name] = create_mask_class(
        finger_noise_class_name, mask,
        touch_sensor_noise_prob=0.0,
        finger_noise_prob=0.9,
    )
