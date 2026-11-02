"""Regression tests for Holodeck task-specific reset state."""

from types import SimpleNamespace

import numpy as np
import pytest

import robocasa.environments.holodeck.tasks.move_cup as move_cup_module
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.articulation import (
    HolodeckOpenDrawer,
)
from robocasa.environments.holodeck.tasks.move_cup import HolodeckMoveCupToCenter
from robocasa.environments.holodeck.tasks.laptop import (
    HolodeckCloseLaptop,
    HolodeckOpenLaptop,
)
from robocasa.environments.holodeck.tasks.mug_lid import HolodeckOpenMugLid
from robocasa.environments.holodeck.tasks.multi_hop.navigation import (
    HolodeckIceCubeThenPlaceCup,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.tasks.push_chair_to_desk import (
    HolodeckPushChairToDesk,
)
from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.pick_book_from_dresser import (
    HolodeckPickBookFromDresser,
)
from robocasa.environments.holodeck.tasks.pick_cup_from_desk import (
    HolodeckPickCupFromDesk,
)
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation


class _FakeData:
    def __init__(self):
        self.qpos = {}
        self.qvel = {}

    def set_joint_qpos(self, name, value):
        self.qpos[name] = np.asarray(value).copy()

    def get_joint_qpos(self, name):
        return self.qpos.get(name, np.zeros(7, dtype=float)).copy()

    def set_joint_qvel(self, name, value):
        self.qvel[name] = np.asarray(value).copy()


def test_holodeck_drawer_reset_restores_calibrated_joint(monkeypatch):
    monkeypatch.setattr(RoomOpenArticulation, "_reset_internal", lambda self: None)
    data = _FakeData()
    task = object.__new__(HolodeckOpenDrawer)
    task.sim = SimpleNamespace(data=data, forward=lambda: None)
    task.target_joint = "drawer_joint"
    task.target_semantic_joint = SimpleNamespace(closed_qpos=0.0, open_qpos=0.2)
    task._get_articulation_task_spec = lambda: None
    task.CLOSED_QPOS = 0.0
    task.OPEN_INIT_FRACTION = 0.20
    task.BEHAVIOR = "open"

    task._reset_internal()

    assert data.qpos["drawer_joint"] == pytest.approx(0.0)
    assert data.qvel["drawer_joint"] == pytest.approx(0.0)


def test_holodeck_cup_reset_restores_task_pose(monkeypatch):
    monkeypatch.setattr(HolodeckEnv, "_reset_internal", lambda self: None)
    data = _FakeData()
    task = object.__new__(HolodeckMoveCupToCenter)
    task.sim = SimpleNamespace(data=data, forward=lambda: None)
    task.objects = {"cup": SimpleNamespace(joints=("cup_joint",))}
    task.task_spec = {
        "cup_pose": {"pos": [1.0, 2.0, 0.7], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    task._initial_cup_base_xy = lambda: np.array([1.5, 2.5])
    task._cup_was_grasped = True
    task._success_hold_steps = 4

    task._reset_internal()

    assert data.qpos["cup_joint"] == pytest.approx(
        [1.5, 2.5, 0.7, 1.0, 0.0, 0.0, 0.0]
    )
    assert data.qvel["cup_joint"] == pytest.approx(np.zeros(6))
    assert task._cup_was_grasped is False
    assert task._success_hold_steps == 0


def test_holodeck_book_lift_success_requires_height_and_resets_below_threshold(
    monkeypatch,
):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = object.__new__(HolodeckPickBookFromDresser)
    state = {"z": 1.0}
    task._book_initial_z = 1.0
    task._success_hold = 0
    task._book_root_z = lambda: state["z"]

    task._post_action(None)
    assert task._success_hold == 0

    state["z"] = 1.31
    task._post_action(None)
    assert task._success_hold == 1

    state["z"] = 1.29
    task._post_action(None)
    assert task._success_hold == 0


def test_holodeck_cup_lift_success_requires_only_height(monkeypatch):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = object.__new__(HolodeckPickCupFromDesk)
    state = {"z": 1.0}
    task._cup_initial_z = 1.0
    task._success_hold = 0
    task._cup_root_z = lambda: state["z"]

    state["z"] = 1.31
    task._post_action(None)
    assert task._success_hold == 1

    state["z"] = 1.29
    task._post_action(None)
    assert task._success_hold == 0


def test_ice_cup_composite_uses_the_cup_region_as_its_first_stage():
    task = object.__new__(HolodeckIceCubeThenPlaceCup)
    task._ice_in_cup_region = lambda: False
    assert task._ice_inserted_after_release() is False

    task._ice_in_cup_region = lambda: True
    assert task._ice_inserted_after_release() is True


def test_ice_cup_composite_validates_only_after_its_layout_is_placed(monkeypatch):
    task = object.__new__(HolodeckIceCubeThenPlaceCup)
    validations = []
    monkeypatch.setattr(
        HolodeckMultiHopComposite,
        "_validate_initial_composite_state",
        lambda _self: validations.append(True),
    )

    task._ice_layout_ready = False
    task._validate_initial_composite_state()
    assert validations == []

    task._ice_layout_ready = True
    task._validate_initial_composite_state()
    assert validations == [True]


def test_holodeck_move_cup_uses_yaml_robot_pose(monkeypatch):
    monkeypatch.setattr(HolodeckEnv, "_setup_task_fixtures", lambda self: None)
    robot_pose = {"pos": [4.9452, 4.8, 0.0], "yaw": 0.0}
    monkeypatch.setattr(
        move_cup_module,
        "get_holodeck_task_spec",
        lambda *_args, **_kwargs: {
            "desk_body": "desk",
            "robot_pose": robot_pose,
            "cup_scale": 1.52,
            "cup_pose": {"pos": [5.45, 4.8, 0.7773]},
            "target_base_xy": [5.67, 4.8],
            "target_tolerance": [0.15, 0.30],
            "tabletop_bounds_xy": [[5.35, 3.77], [6.0, 5.82]],
            "support_z": 0.70,
            "support_z_tolerance": 0.04,
            "upright_local_axis": [0.0, 0.0, 1.0],
            "min_upright_cosine": 0.90,
            "max_linear_speed": 0.08,
            "max_angular_speed": 0.30,
        },
    )
    task = object.__new__(HolodeckMoveCupToCenter)
    task.scene_name = "train_117"
    task.fixture_refs = {}
    task.semantic_scene = SimpleNamespace(fixtures={})
    task.fixtures = {"desk": SimpleNamespace(body_name="desk")}
    task.SHOW_TARGET_REGION = False
    task.get_fixture = lambda **_kwargs: [SimpleNamespace(body_name="desk")]
    assigned = []
    task.set_robot_base_pose = lambda pos, yaw: assigned.append((pos, yaw))

    task._setup_task_fixtures()

    assert len(assigned) == 1
    assigned_pos, assigned_yaw = assigned[0]
    assert assigned_pos == pytest.approx(robot_pose["pos"])
    assert assigned_yaw == pytest.approx(robot_pose["yaw"])


@pytest.mark.parametrize(
    ("robot_model_name", "expected"),
    (
        ("G1FloatingBody", True),
        ("G1", False),
        ("G1FixedLowerBody", False),
    ),
)
def test_holodeck_move_cup_desk_clearance_is_calibrated_for_floating_g1_only(
    robot_model_name, expected
):
    task = object.__new__(HolodeckMoveCupToCenter)
    robot_model = type(robot_model_name, (), {})()
    task.robots = (SimpleNamespace(robot_model=robot_model),)

    assert task._uses_g1_collision_model() is expected


def test_holodeck_laptop_uses_yaml_robot_pose(monkeypatch):
    monkeypatch.setattr(
        RoomOpenArticulation, "_setup_task_fixtures", lambda self: None
    )
    robot_pose = {"pos": [4.9452, 4.0661376, 0.0], "yaw": 0.0}
    monkeypatch.setattr(
        "robocasa.environments.holodeck.tasks.laptop.get_holodeck_task_spec",
        lambda *_args, **kwargs: {
            "target_body": "laptop",
            "target_joint": "laptop_lid",
            "robot_pose": robot_pose,
            "closed_qpos": 1.5708,
            "open_qpos": 0.0,
        },
    )
    task = object.__new__(HolodeckOpenLaptop)
    task.scene_name = "train_117"
    task.fixture_refs = {}
    task._semantic_candidates = lambda: [SimpleNamespace(body_name="laptop")]
    assigned = []
    task.set_robot_base_pose = lambda pos, yaw: assigned.append((pos, yaw))

    task._setup_task_fixtures()

    assert len(assigned) == 1
    assigned_pos, assigned_yaw = assigned[0]
    assert assigned_pos == pytest.approx(robot_pose["pos"])
    assert assigned_yaw == pytest.approx(robot_pose["yaw"])


def test_holodeck_mug_lid_uses_the_exact_xml_table_body_when_unclassified(
    monkeypatch,
):
    monkeypatch.setattr(HolodeckEnv, "_setup_task_fixtures", lambda self: None)
    robot_pose = {"pos": [5.25, 1.45, 0.0], "yaw": 0.0}
    monkeypatch.setattr(
        "robocasa.environments.holodeck.tasks.mug_lid.get_holodeck_task_spec",
        lambda *_args, **_kwargs: {
            "table_body": "unclassified_objaverse_desk",
            "robot_pose": robot_pose,
            "object_scale": 1.0,
            "mug_pose": {"pos": [5.75, 1.4, 0.82]},
            "lid_pose": {"pos": [5.75, 1.4, 0.92]},
        },
    )
    task = object.__new__(HolodeckOpenMugLid)
    task.scene_name = "train_1664"
    task._laptop_layout = None
    task.fixture_refs = {}
    task.get_fixture = lambda **_kwargs: []
    task.find_scene_body = lambda name: (
        object() if name == "unclassified_objaverse_desk" else None
    )
    assigned = []
    task.set_robot_base_pose = lambda pos, yaw: assigned.append((pos, yaw))

    task._setup_task_fixtures()

    assert task.table_fixture is None
    assert task._is_scene_valid()
    assert assigned[0][0] == pytest.approx(robot_pose["pos"])
    assert assigned[0][1] == pytest.approx(robot_pose["yaw"])


def test_close_laptop_uses_a_noise_free_robot_spawn_for_exact_standoff():
    assert HolodeckCloseLaptop.NOISE_FREE_ROBOT_SPAWN is True
    assert HolodeckOpenLaptop.NOISE_FREE_ROBOT_SPAWN is True


def test_holodeck_chair_reset_restores_task_pose(monkeypatch):
    monkeypatch.setattr(HolodeckEnv, "_reset_internal", lambda self: None)
    data = _FakeData()
    task = object.__new__(HolodeckPushChairToDesk)
    task.sim = SimpleNamespace(data=data, forward=lambda: None)
    task.chair_joint = "chair_joint"
    task.task_spec = {
        "chair_start_pose": {
            "pos": [3.0, 4.0, 0.8],
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
    }
    task._success_hold_steps = 5
    task._position_debug_steps = 8

    task._reset_internal()

    assert data.qpos["chair_joint"] == pytest.approx(
        [3.0, 4.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    )
    assert data.qvel["chair_joint"] == pytest.approx(np.zeros(6))
    assert task._success_hold_steps == 0
    assert task._position_debug_steps == 0


def test_holodeck_chair_success_uses_target_region_only():
    task = object.__new__(HolodeckPushChairToDesk)
    task.chair_body = "chair"
    task.target_xy = np.array([3.0, 4.0])
    task.target_tolerance = np.array([0.2, 0.3])
    task.sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda _name: 0),
        data=SimpleNamespace(
            body_xpos=np.array([[3.15, 4.25, 0.5]]),
        ),
    )

    assert task._chair_in_target_region()

    task.sim.data.body_xpos[0, 0] = 3.21
    assert not task._chair_in_target_region()


def test_holodeck_pull_chair_uses_one_second_hold():
    for task_cls in (HolodeckPushChairToDesk, HolodeckPullChairFromDesk):
        task = object.__new__(task_cls)
        task.control_freq = 20
        assert task.HOLD_SECONDS == pytest.approx(1.0)
        assert task.hold_steps == 20
