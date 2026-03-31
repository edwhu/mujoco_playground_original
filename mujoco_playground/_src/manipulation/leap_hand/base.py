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
"""Base classes for leap hand."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts


def load_fixed_masks(mask_path):
  """Load fixed masks from sensor_masks.json and convert to JAX arrays.

  Supports two JSON formats:
    1. Compact: {"mask1": "010101...", ...} — bitstrings
    2. Hierarchical: {"2_sensors": {"tips": [0,0,1,...], ...}, ...}

  Returns:
    Tuple of (all_masks, mask_names) where all_masks is a list of JAX
    boolean arrays and mask_names is a list of corresponding names.
  """
  import json

  with open(mask_path, 'r') as f:
    masks_data = json.load(f)

  def _flatten_masks(data):
    masks = []
    names = []
    for category, category_masks in data.items():
      if category == "metadata":
        continue
      for mask_name, mask_values in category_masks.items():
        masks.append(jp.array(mask_values, dtype=jp.bool_))
        names.append(f"{category}_{mask_name}")
    return masks, names

  # Detect compact bitstring format
  is_compact = isinstance(masks_data, dict) and len(masks_data) > 0 and all(
      isinstance(v, str) and all(ch in "01" for ch in v)
      for v in masks_data.values()
  )

  if is_compact:
    converted = {"imported": {}}
    for name, bitstring in masks_data.items():
      converted["imported"][name] = [1 if ch == '1' else 0 for ch in bitstring]
    all_masks, mask_names = _flatten_masks(converted)
  else:
    all_masks, mask_names = _flatten_masks(masks_data)

  return all_masks, mask_names


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "leap_hand"
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "xmls" / "reorientation_cube_textures"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


class LeapHandEnv(mjx_env.MjxEnv):
  """Base class for LEAP hand environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 10

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  # Sensor readings.

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "palm_position")

  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_position")

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_orientation")

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

  def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angacc")

  def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_upvector")

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_orientation")

  def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_upvector")

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
    ])

  # Touch sensor methods.

  def get_touch_sensors(self, data: mjx.Data) -> jax.Array:
    """Get binarized touch sensor data (20 dims)."""
    touch = jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, name)
        for name in consts.TOUCH_SENSOR_NAMES
    ])
    return (touch > 0.0).astype(jp.float32)

  def generate_episode_touch_mask(
      self,
      info: dict,
      fixed_mask=None,
  ) -> jax.Array:
    """Generate touch mask for an episode.

    Returns a 20-dim boolean mask. If fixed_mask is provided, uses that;
    otherwise returns all-ones (all sensors active).
    """
    if fixed_mask is not None:
      return fixed_mask
    return jp.ones(len(consts.TOUCH_SENSOR_NAMES), dtype=jp.bool_)

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])
