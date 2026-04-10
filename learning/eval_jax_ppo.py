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
"""Load a PPO checkpoint, run Brax Evaluator metrics, and optionally render videos.

Example: ``python learning/eval_jax_ppo.py --mask_path=scripts/sensor_masks.json --env_name=LeapCubeRotateZAxisTouchMask16FingerNoiseFingerThreshold10 --load_checkpoint_path=logs/LeapCubeRotateZAxisTouchMask14FingerNoise-20260406-125952-rz-newxml-fingernoise-8sensor_203/checkpoints/ --num_eval_envs=128 --num_videos=1``
"""

import datetime
import functools
import json
import os
import warnings

os.environ["MUJOCO_GL"] = "egl"
OS_ENV_PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"
if OS_ENV_PREALLOCATE not in os.environ:
  os.environ[OS_ENV_PREALLOCATE] = "false"

from absl import app
from absl import flags
from absl import logging
from brax.training import acting
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jp
import mediapy as media
from etils import epath
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo.train import _maybe_wrap_env

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path",
    None,
    "Path to checkpoint directory (numeric step folder) or parent checkpoints dir.",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the run output directory name")
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization (must match training)"
)
_MASK_PATH = flags.DEFINE_string(
    "mask_path", None, "Path to sensor masks JSON file for touch environments"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of parallel envs for Brax Evaluator"
)
_DETERMINISTIC_EVAL = flags.DEFINE_boolean(
    "deterministic_eval",
    False,
    "Deterministic policy during Brax eval (matches brax PPO train flag)",
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of parallel rollout videos to render"
)
_SKIP_VIDEO = flags.DEFINE_boolean(
    "skip_video", False, "Only run Brax eval metrics, skip rendering"
)
_SKIP_BRAX_EVAL = flags.DEFINE_boolean(
    "skip_brax_eval", False, "Skip Brax Evaluator (e.g. only render videos)"
)
_REWARD_SCALING = flags.DEFINE_float(
    "reward_scaling", 0.1, "Override RL reward_scaling in config (usually unused for eval)"
)
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Override episode length")
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Override action repeat")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Override obs normalization flag"
)
_LOGDIR = flags.DEFINE_string("logdir", None, "Directory for logs and outputs")
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir", None,
    "Directory for caching compiled Warp kernels.",
)


def get_rl_config(env_name: str):
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  if env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  if env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)
  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def resolve_checkpoint_path(user_path: str) -> epath.Path:
  ckpt_path = epath.Path(user_path).resolve()
  if ckpt_path.is_dir():
    subdirs = [c for c in ckpt_path.glob("*") if c.is_dir()]
    if not subdirs:
      raise ValueError(
          f"No checkpoint step subdirectories under {ckpt_path}"
      )
    subdirs.sort(key=lambda x: int(x.name))
    return subdirs[-1]
  return ckpt_path


def metrics_to_jsonable(metrics: dict) -> dict:
  out = {}
  for k, v in metrics.items():
    if hasattr(v, "item"):
      out[k] = v.item() if getattr(v, "shape", ()) == () else v.tolist()
    elif isinstance(v, dict):
      out[k] = metrics_to_jsonable(v)
    else:
      out[k] = float(v) if isinstance(v, (float, int)) else str(v)
  return out


def main(argv):
  del argv

  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top
    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  if _LOAD_CHECKPOINT_PATH.value is None:
    raise app.UsageError("--load_checkpoint_path is required.")

  if _MASK_PATH.value is not None:
    mujoco_playground.manipulation.register_rotation_environments_with_masks(
        _MASK_PATH.value
    )

  env_cfg = registry.get_default_config(_ENV_NAME.value)
  ppo_params = get_rl_config(_ENV_NAME.value)
  ppo_params.num_timesteps = 0

  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value

  env_cfg_overrides = {"impl": _IMPL.value}
  if _VISION.value:
    env_cfg_overrides["vision"] = True
    env_cfg_overrides["vision_config.nworld"] = ppo_params.num_envs
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )

  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"eval-{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Output directory: {logdir}")

  restore_path = resolve_checkpoint_path(_LOAD_CHECKPOINT_PATH.value)
  print(f"Loading checkpoint via ppo.train (num_timesteps=0) from: {restore_path}")

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  randomization_fn = None
  if _DOMAIN_RANDOMIZATION.value:
    randomization_fn = registry.get_domain_randomizer(_ENV_NAME.value)

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = randomization_fn

  num_eval_envs = int(ppo_params.get("num_eval_envs", 128))
  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]
  ckpt_path = logdir / "checkpoints"
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      vision=_VISION.value,
  )

  def progress(_num_steps, _metrics):
    pass

  eval_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    eval_env_overrides["vision_config.nworld"] = num_eval_envs
  eval_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=eval_env_overrides,
  )

  make_inference_fn, policy_params, _ = train_fn(
      environment=env,
      progress_fn=progress,
      policy_params_fn=lambda *args: None,
      eval_env=eval_env,
  )

  eval_key = jax.random.PRNGKey(_SEED.value)
  wrapped_eval = _maybe_wrap_env(
      eval_env,
      wrap_env=True,
      num_envs=num_eval_envs,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.action_repeat,
      device_count=1,
      key_env=eval_key,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      randomization_fn=randomization_fn,
  )

  evaluator = acting.Evaluator(
      wrapped_eval,
      functools.partial(
          make_inference_fn, deterministic=_DETERMINISTIC_EVAL.value
      ),
      num_eval_envs=num_eval_envs,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.action_repeat,
      key=jax.random.fold_in(eval_key, 1),
  )

  metrics = {}
  if not _SKIP_BRAX_EVAL.value:
    metrics = evaluator.run_evaluation(policy_params, {})
    print("Brax eval metrics:")
    for k in sorted(metrics.keys()):
      print(f"  {k}: {metrics[k]}")
    metrics_path = logdir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
      json.dump(metrics_to_jsonable(metrics), fp, indent=2)
    print(f"Saved metrics to {metrics_path}")

  if _SKIP_VIDEO.value:
    return

  print("Rendering rollout videos...")
  inference_fn = jax.jit(make_inference_fn(policy_params, deterministic=True))

  infer_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    infer_env_overrides["vision_config.nworld"] = _NUM_VIDEOS.value
  infer_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=infer_env_overrides,
  )
  wrapped_infer_env = wrapper.wrap_for_brax_training(
      infer_env,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.get("action_repeat", 1),
  )

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(wrapped_infer_env.reset)(rng)

  empty_data = reset_states.data.__class__(
      **{k: None for k in reset_states.data.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = reset_states.__class__(
      **{k: None for k in reset_states.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = empty_traj.replace(data=empty_data)

  def step(carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
    act = jax.vmap(inference_fn)(state.obs, act_keys)[0]
    state = wrapped_infer_env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
        "reward": state.reward,
    })
    return (state, rng), traj_data

  @jax.jit
  def do_rollout(state, rng):
    _, traj = jax.lax.scan(
        step, (state, rng), None, length=ppo_params.episode_length
    )
    return traj

  traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
  episode_returns = list(
      jax.device_get(jp.ravel(jp.sum(traj_stacked.reward, axis=0)))
  )
  traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(ppo_params.episode_length)
    ]

  render_every = 1
  fps = 1.0 / infer_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = infer_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    ret = float(episode_returns[i])
    ret_label = f"{round(ret, 2):.2f}"
    base = f"rollout{i}_ret{ret_label}"
    media.write_video(logdir / f"{base}.mp4", frames, fps=fps)
    with open(logdir / f"{base}_return.txt", "w", encoding="utf-8") as fp:
      fp.write(f"{ret}\n")
    print(
        f"Rollout video saved as '{logdir}/{base}.mp4',"
        f" episode return={ret:.4f} (also {logdir}/{base}_return.txt)."
    )


def run():
  app.run(main)


if __name__ == "__main__":
  run()
