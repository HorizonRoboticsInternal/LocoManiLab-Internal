"""Gymnasium access to native G1 environment observations and actions.

The action Box is the underlying ``env.action_spec``. Whole-body controllers
currently expose a dummy action while their selected backend receives commands
through its own interface. This wrapper does not translate policy tokens or
reference poses and does not implement the LocoManiLab policy-server contract.
"""

import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.utils.env_utils import create_env
from robocasa.utils.robot_utils import DEFAULT_CAMERA, DEFAULT_ROBOT, validate_g1_robots


def _observation_space(observation):
    """Describe the actual numerical/image values without remapping their keys."""
    entries = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if array.dtype.kind == "f":
            low, high = -np.inf, np.inf
        elif array.dtype.kind in "iu":
            bounds = np.iinfo(array.dtype)
            low, high = bounds.min, bounds.max
        elif array.dtype.kind == "b":
            low, high = 0, 1
        else:
            raise TypeError(f"Unsupported native observation {key!r}: {array.dtype}")
        entries[key] = spaces.Box(low=low, high=high, shape=array.shape, dtype=array.dtype)
    return spaces.Dict(entries)


class RoboCasaGymEnv(gym.Env):
    """Keep native G1 state/image keys, action bounds, rewards and termination.

    Task text is returned in ``info`` alongside the native environment info.
    Scene selection and splits
    are forwarded to ``create_env``; callers should use its supported split
    names (None, pretrain, target, all).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        split=None,
        robots=DEFAULT_ROBOT,
        render_mode=None,
        **kwargs,
    ):
        validate_g1_robots(robots)
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.camera_names = (
            [DEFAULT_CAMERA] if camera_names is None
            else [camera_names] if isinstance(camera_names, str)
            else list(camera_names)
        )
        self.env_name = env_name
        self.env = create_env(
            env_name=env_name,
            robots=robots,
            render_onscreen=False,
            render_offscreen=enable_render,
            split=split,
            camera_names=self.camera_names,
            camera_widths=640 if camera_widths is None else camera_widths,
            camera_heights=480 if camera_heights is None else camera_heights,
            **kwargs,
        )
        self.enable_render = enable_render
        self.render_cache = None
        observation = self.env.reset()
        self.observation_space = _observation_space(observation)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )
        self._cache_render(observation)

    def _cache_render(self, observation):
        if self.enable_render and self.camera_names:
            image = observation.get(f"{self.camera_names[0]}_image")
            if image is not None:
                self.render_cache = np.array(image[::-1], copy=True)

    def _info(self, info=None):
        result = dict(info or {})
        result["language"] = self.env.get_ep_meta().get("lang", "")
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        if options:
            raise ValueError("Pass scene options when constructing the environment")
        observation = self.env.reset()
        self._cache_render(observation)
        return observation, self._info()

    def step(self, action):
        action = np.asarray(action)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected a native G1 action of shape {self.action_space.shape}, got {action.shape}"
            )
        observation, reward, done, info = self.env.step(action)
        self._cache_render(observation)
        return observation, reward, bool(done), False, self._info(info)

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("No camera frame is available; enable camera observations first")
        return self.render_cache

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "env"), name)


def create_robocasa_gym_env_class(env):
    """Retain the existing ``robocasa/<TaskName>`` Gymnasium IDs."""
    def initialize(self, **kwargs):
        RoboCasaGymEnv.__init__(self, env_name=env, **kwargs)

    env_class_type = type(env, (RoboCasaGymEnv,), {"__init__": initialize})
    setattr(sys.modules[__name__], env, env_class_type)
    env_id = f"robocasa/{env}"
    if env_id not in registry:
        register(id=env_id, entry_point=f"robocasa.wrappers.gym_wrapper:{env}")


for _env in REGISTERED_ENVS:
    create_robocasa_gym_env_class(_env)
