"""Focused tests for the TeleLab conveyor speed and task layout."""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

from robocasa.environments.telelab.telelab_conveyor import TeleLabConveyor
from robocasa.environments.telelab.telelab_env import TeleLabEnv
from robocasa.environments.telelab.conveyor_box_to_desk import (
    TeleLabConveyorPlaceBoxOnDesk,
)
from robocasa.models.fixtures.conveyor_belt import ConveyorBeltFixture
from robosuite.utils.mjcf_utils import string_to_array


def _make_unloaded_env(**kwargs):
    with patch.object(TeleLabEnv, "__init__", return_value=None):
        return TeleLabConveyor(**kwargs)


@pytest.mark.parametrize(
    "speed_range",
    [
        (-1.0,),
        (-1.0, 0.0, 1.0),
        (0.1, 1.0),
        (-1.0, -0.1),
        (1.0, -1.0),
        (-1.6, 1.0),
        (-1.0, 1.6),
    ],
)
def test_conveyor_speed_range_validation(speed_range):
    with pytest.raises(ValueError):
        _make_unloaded_env(conveyor_speed_range=speed_range)


def test_set_conveyor_speed_clamps_to_configured_range():
    env = _make_unloaded_env(conveyor_speed_range=(-0.4, 0.6))
    env._belt_actuator_names = ["belt_drive_00", "belt_drive_01"]
    commands = {}

    def set_scene_actuator(name, value):
        commands[name] = value
        return value

    env.set_scene_actuator = set_scene_actuator

    assert env.set_conveyor_speed(0.9) == pytest.approx(0.6)
    assert commands == {"belt_drive_00": 0.6, "belt_drive_01": 0.6}
    assert env.set_conveyor_speed(-0.9) == pytest.approx(-0.4)


def test_conveyor_keypress_uses_configured_step_and_limits(capsys):
    env = _make_unloaded_env(
        conveyor_speed=0.0,
        conveyor_speed_step=0.3,
        conveyor_speed_range=(-0.5, 0.5),
    )
    env._belt_actuator_names = ["belt_drive_00"]
    env.set_scene_actuator = lambda _name, value: value

    assert env.handle_scene_keypress(ord("o"))
    assert env.conveyor_speed == pytest.approx(0.3)
    assert env.handle_scene_keypress(ord("O"))
    assert env.conveyor_speed == pytest.approx(0.5)
    assert env.handle_scene_keypress(ord("P"))
    assert env.conveyor_speed == pytest.approx(0.0)
    assert not env.handle_scene_keypress(ord("X"))
    assert "conveyor speed" in capsys.readouterr().out


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_conveyor_rejects_nonfinite_speeds(value):
    env = _make_unloaded_env()
    env._belt_actuator_names = ["belt_drive_00"]
    env.set_scene_actuator = lambda _name, speed: speed

    with pytest.raises(ValueError):
        env.set_conveyor_speed(value)


def test_scaled_end_rollers_stay_inside_frame():
    scale = np.asarray((2.5 / 4.3, 0.7 / 1.2, 0.7 / 0.788))
    fixture = ConveyorBeltFixture(scale=scale)
    body = fixture.worldbody.find("./body[@name='conveyor_belt_single']")
    frame = body.find("./geom[@name='conveyor_frame']")
    frame_half_size = string_to_array(frame.get("size"))

    for name in ("belt_end_left", "belt_end_right"):
        roller = body.find(f"./geom[@name='{name}']")
        roller_pos = string_to_array(roller.get("pos"))
        roller_size = string_to_array(roller.get("size"))

        assert roller_size[1] == pytest.approx(0.52 * scale[1])
        assert abs(roller_pos[0]) + roller_size[0] <= frame_half_size[0]
        assert roller_size[1] <= frame_half_size[1]


@pytest.mark.parametrize("env_type", [TeleLabConveyor, TeleLabConveyorPlaceBoxOnDesk])
def test_default_l_layout_puts_robot_inside_facing_conveyor(env_type):
    with patch.object(TeleLabEnv, "__init__", return_value=None):
        env = env_type()

    conveyor_left = env.conveyor_pos[0] - 2.15 * env.conveyor_scale[0]
    conveyor_near = env.conveyor_pos[1] - 0.60 * env.conveyor_scale[1]
    desk_left = env.desk_pos[0] + (0.032 - 0.422) * env.desk_scale[2]
    desk_right = env.desk_pos[0] + (0.032 + 0.422) * env.desk_scale[2]
    desk_far = env.desk_pos[1] + (0.075 + 0.795) * env.desk_scale[0]

    assert desk_left - conveyor_left == pytest.approx(env.DESK_SHIFT_X)
    assert conveyor_near - desk_far == pytest.approx(env.L_CORNER_GAP_Y)
    assert env.robot_base_pos[0] > desk_right
    assert env.robot_base_pos[1] < conveyor_near
    assert env.robot_base_ori[2] == pytest.approx(math.pi / 2)


def test_place_box_task_tightens_robot_and_desk_layout():
    with patch.object(TeleLabEnv, "__init__", return_value=None):
        task = TeleLabConveyorPlaceBoxOnDesk()

    conveyor_near = task.conveyor_pos[1] - 0.60 * task.conveyor_scale[1]
    desk_far = task.desk_pos[1] + (0.075 + 0.795) * task.desk_scale[0]
    assert conveyor_near - task.robot_base_pos[1] == pytest.approx(0.20)
    assert conveyor_near - desk_far == pytest.approx(0.02)
    assert task.desk_pos[0] + (0.032 - 0.422) * task.desk_scale[2] - (
        task.conveyor_pos[0] - 2.15 * task.conveyor_scale[0]
    ) == pytest.approx(0.05)


def test_task_boxes_are_shifted_as_a_group_to_belt_left_end():
    task = TeleLabConveyorPlaceBoxOnDesk
    positions = np.asarray([local_x for _, local_x in task.BOX_SPECS])

    np.testing.assert_allclose(np.diff(positions), np.full(2, 0.69 * task.CONVEYOR_SCALE[0]))
    assert task.BOX_POSITION_OFFSET[0] == pytest.approx(-0.18)

    box_1_bbox_left = (
        -0.0069783 - 0.0985911
    ) * task.BOX_SCALE_OVERRIDES["box_1"]
    belt_left = -2.15 * task.CONVEYOR_SCALE[0]
    box_1_left = positions[0] + task.BOX_POSITION_OFFSET[0] + box_1_bbox_left
    assert 0.01 < box_1_left - belt_left < 0.03


def test_task_boxes_are_light_and_rotated_about_z():
    task = TeleLabConveyorPlaceBoxOnDesk

    assert task.BOX_SCALE_OVERRIDES == {"box_1": 0.45, "box_2": 0.45}
    assert task.BOX_MASS_OVERRIDES == {
        "box_1": 0.15,
        "box_2": 0.15,
        "box_3": 0.08,
    }
    assert all(
        0 < mass <= task.BOX_MASS_LIMIT_KG
        for mass in task.BOX_MASS_OVERRIDES.values()
    )
    assert set(task.BOX_QUAT_OVERRIDES) == {"box_1", "box_2", "box_3"}
    assert task.BOX_QUAT_OVERRIDES["box_1"] == pytest.approx(
        (0.0, 0.0, 0.70710678, 0.70710678)
    )
    assert task.BOX_QUAT_OVERRIDES["box_2"] == pytest.approx(
        (0.5, 0.5, 0.5, 0.5)
    )
    assert task.BOX_QUAT_OVERRIDES["box_3"] == pytest.approx(
        (0.0, 0.0, 0.70710678, 0.70710678)
    )
    for quat in task.BOX_QUAT_OVERRIDES.values():
        assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_box_mass_overrides_reach_compiled_bodies():
    env = TeleLabConveyorPlaceBoxOnDesk(
        robots="G1",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    try:
        compiled_masses = {
            name: env.sim.model.body_mass[
                env.sim.model.body_name2id(env.objects[name].root_body)
            ]
            for name in env.SOURCE_BOX_NAMES
        }
        assert compiled_masses == pytest.approx(env.BOX_MASS_OVERRIDES)
    finally:
        env.close()


def test_left_and_right_task_boxes_have_extra_negative_y_offset():
    task = TeleLabConveyorPlaceBoxOnDesk

    base_offset = np.asarray(task.BOX_POSITION_OFFSET)
    for box_name in ("box_1", "box_3"):
        override = np.asarray(task.BOX_POSITION_OFFSET_OVERRIDES[box_name])
        assert override[0] == pytest.approx(base_offset[0])
        assert override[1] == pytest.approx(base_offset[1] - 0.05)
        assert override[2] == pytest.approx(base_offset[2])
    assert "box_2" not in task.BOX_POSITION_OFFSET_OVERRIDES
    assert base_offset[1] == pytest.approx(-0.15)


def test_box_to_desk_uses_reduced_conveyor_speed():
    with patch.object(TeleLabEnv, "__init__", return_value=None):
        env = TeleLabConveyorPlaceBoxOnDesk()

    assert env.CONVEYOR_SPEED == pytest.approx(0.1)
    assert env.conveyor_speed == pytest.approx(0.0)

    env.control_freq = 20
    env.sim = Mock()
    env.set_conveyor_speed = Mock()
    env._box_on_desk = Mock(return_value=False)
    with patch.object(TeleLabConveyor, "_reset_internal"):
        env._reset_internal()

    env.set_conveyor_speed.assert_called_once_with(0.0)


def test_box_to_desk_starts_conveyor_on_keyboard_e_without_consuming_controller_key():
    with patch.object(TeleLabEnv, "__init__", return_value=None):
        env = TeleLabConveyorPlaceBoxOnDesk()

    env.conveyor_speed = 0.0
    env.set_conveyor_speed = Mock()

    assert not env.handle_scene_keypress(ord("e"))
    env.set_conveyor_speed.assert_called_once_with(env.CONVEYOR_SPEED)
