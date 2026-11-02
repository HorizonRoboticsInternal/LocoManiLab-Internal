"""Native G1 Gymnasium contracts, using fake environments without scene assets."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robocasa.wrappers import gym_wrapper
from robocasa.utils import env_utils
from robocasa.g1_host.controller_registry import register_g1_controller_factories
from robosuite.controllers.composite.composite_controller import (
    REGISTERED_COMPOSITE_CONTROLLERS_DICT,
)


class _NativeEnv:
    action_spec = (np.array([-2.0, 0.0]), np.array([3.0, 1.0]))

    def __init__(self):
        self.reset_count = 0
        self.step_actions = []
        self.closed = False
        self.observation = {
            "robot0_joint_pos": np.array([0.25, -0.5], dtype=np.float32),
            "ego_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            "contact": np.array(True),
            "counter": np.array([1], dtype=np.int32),
        }

    def reset(self):
        self.reset_count += 1
        return self.observation

    def step(self, action):
        self.step_actions.append(action)
        return self.observation, 2.5, True, {"native_detail": "preserved"}

    def get_ep_meta(self):
        return {"lang": "move the cup"}

    def _check_success(self):
        raise AssertionError("Wrapper repeated a stateful task success check")

    def close(self):
        self.closed = True


@pytest.fixture
def native(monkeypatch):
    env = _NativeEnv()
    calls = []

    def create_env(**kwargs):
        calls.append(kwargs)
        return env

    monkeypatch.setattr(gym_wrapper, "create_env", create_env)
    return SimpleNamespace(env=env, calls=calls)


def test_gym_wrapper_forwards_g1_camera_split_and_scene_selection(native):
    wrapper = gym_wrapper.RoboCasaGymEnv(
        env_name="ExampleG1Task", camera_names=["ego"],
        camera_widths=24, camera_heights=18, split="target", scene_ids=401,
    )
    assert native.calls == [{
        "env_name": "ExampleG1Task", "robots": "G1FloatingBody",
        "render_onscreen": False, "render_offscreen": True,
        "split": "target", "camera_names": ["ego"],
        "camera_widths": 24, "camera_heights": 18, "scene_ids": 401,
    }]
    assert native.env.reset_count == 1
    assert wrapper.action_space.shape == (2,)
    np.testing.assert_array_equal(wrapper.action_space.low, [-2.0, 0.0])
    np.testing.assert_array_equal(wrapper.action_space.high, [3.0, 1.0])
    assert set(wrapper.observation_space.spaces) == set(native.env.observation)
    for key, value in native.env.observation.items():
        assert wrapper.observation_space[key].shape == value.shape
        assert wrapper.observation_space[key].dtype == value.dtype
    assert wrapper.observation_space.contains(native.env.observation)


def test_gym_wrapper_default_camera_is_g1_head_camera(native):
    gym_wrapper.RoboCasaGymEnv(env_name="ExampleG1Task")
    assert native.calls[0]["camera_names"] == ["robot0_head_camera"]


def test_gym_wrapper_treats_a_single_camera_name_as_one_camera(native):
    wrapper = gym_wrapper.RoboCasaGymEnv(env_name="ExampleG1Task", camera_names="ego")
    assert native.calls[0]["camera_names"] == ["ego"]
    np.testing.assert_array_equal(wrapper.render(), native.env.observation["ego_image"][::-1])


def test_gym_without_rendering_does_not_request_an_offscreen_renderer(native):
    wrapper = gym_wrapper.RoboCasaGymEnv(
        env_name="ExampleG1Task", camera_names="ego", enable_render=False,
    )
    assert native.calls[0]["render_offscreen"] is False
    with pytest.raises(RuntimeError, match="camera frame"):
        wrapper.render()


def test_gym_reset_keeps_raw_keys_and_reseeds_without_changing_scene(native):
    wrapper = gym_wrapper.RoboCasaGymEnv(env_name="ExampleG1Task", camera_names=["ego"])
    observation, info = wrapper.reset(seed=123)
    assert observation is native.env.observation
    assert info == {"language": "move the cup"}
    assert native.env.reset_count == 2
    assert len(native.calls) == 1
    assert native.env.rng.integers(1000) == np.random.default_rng(123).integers(1000)
    with pytest.raises(ValueError, match="constructing"):
        wrapper.reset(options={"scene_ids": 402})
    assert native.env.reset_count == 2


def test_gym_step_delegates_native_actions_rewards_and_done_once(native):
    wrapper = gym_wrapper.RoboCasaGymEnv(env_name="ExampleG1Task", camera_names=["ego"])
    action = np.array([0.2, 0.8], dtype=np.float32)
    observation, reward, terminated, truncated, info = wrapper.step(action)
    assert observation is native.env.observation
    np.testing.assert_array_equal(native.env.step_actions[0], action)
    assert reward == 2.5
    assert terminated is True and truncated is False
    assert info == {"native_detail": "preserved", "language": "move the cup"}
    with pytest.raises(ValueError, match="native G1 action"):
        wrapper.step(np.zeros(64))
    assert len(native.env.step_actions) == 1
    wrapper.close()
    assert native.env.closed is True


def test_gym_render_uses_camera_frame_without_mutating_raw_observation(native):
    wrapper = gym_wrapper.RoboCasaGymEnv(env_name="ExampleG1Task", camera_names=["ego"])
    image = native.env.observation["ego_image"]
    expected = image.copy()
    np.testing.assert_array_equal(wrapper.render(), expected[::-1])
    wrapper.render()[0] = 0
    np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize("robots", ["PandaOmron", "GR1", ["G1", "G1FloatingBody"]])
def test_gym_rejects_other_robots_before_environment_construction(native, robots):
    with pytest.raises(ValueError, match="one G1 robot"):
        gym_wrapper.RoboCasaGymEnv(env_name="ExampleG1Task", robots=robots)
    assert native.calls == []


def test_create_env_defaults_to_g1_before_loading_a_controller(monkeypatch):
    seen = []

    def stop_at_config(*, controller, robot):
        seen.append(robot)
        raise RuntimeError("stopped before controller/model loading")

    monkeypatch.setattr(env_utils, "load_composite_controller_config", stop_at_config)
    with pytest.raises(RuntimeError, match="stopped before"):
        env_utils.create_env("EmptyEnv")
    assert seen == ["G1FloatingBody"]


def test_create_env_rejects_other_robots_before_loading_a_controller(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Unsupported robot caused controller loading")

    monkeypatch.setattr(env_utils, "load_composite_controller_config", forbidden)
    with pytest.raises(ValueError, match="one G1 robot"):
        env_utils.create_env("EmptyEnv", robots="PandaOmron")


@pytest.mark.parametrize("name, module_name, class_name", [
    ("WHOLE_BODY_G1", "robocasa.g1_host.wbt", "WholeBodyG1Composite"),
    ("G1_BRIDGE", "robocasa.g1_host.bridge", "G1BridgeComposite"),
])
def test_g1_controller_factories_delegate_only_when_selected(
    monkeypatch, name, module_name, class_name,
):
    # Stand in for the optional runtime. Registering and selecting a controller
    # must preserve its constructor arguments, without requiring policy weights.
    received = []

    class SelectedController:
        def __init__(self, *args, **kwargs):
            received.append((args, kwargs))

    module = ModuleType(module_name)
    setattr(module, class_name, SelectedController)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delitem(REGISTERED_COMPOSITE_CONTROLLERS_DICT, name, raising=False)
    register_g1_controller_factories()
    assert received == []
    result = REGISTERED_COMPOSITE_CONTROLLERS_DICT[name]("sim", robot_model="robot", grippers={})
    assert isinstance(result, SelectedController)
    assert received == [(("sim",), {"robot_model": "robot", "grippers": {}})]

    monkeypatch.setitem(REGISTERED_COMPOSITE_CONTROLLERS_DICT, name, SelectedController)
    register_g1_controller_factories()
    assert REGISTERED_COMPOSITE_CONTROLLERS_DICT[name] is SelectedController
