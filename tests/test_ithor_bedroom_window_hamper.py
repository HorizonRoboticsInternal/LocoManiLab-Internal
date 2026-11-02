"""Ordered window/hamper contract without constructing a MuJoCo environment."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_bedroom.tasks.bedside_window_hamper import (
    IThorBedroomApproachBedsideWindowThenOpenHamperLid as WindowHamper,
    _BedsideWindowThenHamperMixin,
)
from robocasa.environments.ithor_bedroom.tasks.hamper_lid import (
    IThorBedroomOpenHamperLid,
)
from robocasa.environments.ithor_bedroom.tasks.navigation import (
    IThorBedroomApproachBedsideWindow,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)


class _Terminal:
    def _post_action(self, action):
        return None

    def _reset_internal(self):
        self.fraction = 0.30


class Route(_BedsideWindowThenHamperMixin, _Terminal):
    PRINT_STAGE_MESSAGES = False

    def __init__(self, frequency=20):
        self.control_freq = frequency
        self.HOLD_STEPS = 5
        self._get_articulation_task_spec = lambda: None
        RoomOpenArticulation._configure_articulation_task(self)
        self.window_fixture = "window"
        self.target_fixture = "hamper"
        self.near = set()
        self.sim = None
        self.target_semantic_joint = SimpleNamespace(fraction=lambda _: self.fraction)
        self._reset_internal()

    def robot_fixture_collision_distance(self, fixture, max_distance):
        return 0.25 if fixture in self.near else max_distance

    def _raw_success(self):
        return self.fraction >= 0.80

    def step(self, n=1):
        for _ in range(n):
            self._post_action(None)

    def visit(self, fixture):
        self.near = {fixture}
        self.step(self.APPROACH_HOLD_STEPS)


@pytest.mark.parametrize("frequency", [10, 20])
def test_correct_order_final_duration_and_pure_observation(frequency):
    task = Route(frequency)
    assert not task._check_success()
    task.visit("window")
    assert task._window_hamper_stage == 1
    task.visit("hamper")
    assert task._window_hamper_stage == 2
    assert not task._check_success()
    task.fraction = 0.80
    task.step(frequency - 1)
    assert not task._check_success()
    task.step()
    snapshot = task.__dict__.copy()
    assert all(task._check_success() for _ in range(20))
    assert task.__dict__ == snapshot
    task.near.clear()
    task.step()
    assert not task._check_success() and task._opening_hold == 0
    task.near = {"hamper"}
    task.step(frequency)
    assert task._check_success()
    task.fraction = 0.79
    task.step()
    assert not task._check_success()


def test_hamper_first_and_early_opening_do_not_skip_navigation():
    task = Route()
    task.visit("hamper")
    assert task._window_hamper_stage == 0
    task.fraction = 0.80
    task.visit("window")
    assert task._window_hamper_stage == 0
    task.fraction = 0.30
    task.visit("window")
    assert task._window_hamper_stage == 1
    task.fraction = 0.80
    task.visit("hamper")
    task.step(task.HOLD_STEPS)
    assert task._window_hamper_stage == 1 and not task._check_success()
    task.fraction = 0.30
    task.visit("hamper")
    task.fraction = 0.80
    task.step(task.HOLD_STEPS)
    assert task._check_success()


@pytest.mark.parametrize("fixture, prior", [("window", None), ("hamper", "window")])
def test_navigation_holds_reset_on_departure_or_lid_change(fixture, prior):
    task = Route()
    if prior:
        task.visit(prior)
    initial_stage = task._window_hamper_stage
    task.near = {fixture}
    task.step(4)
    task.near.clear()
    task.step()
    assert task._navigation_hold == 0
    task.near = {fixture}
    task.step(4)
    task.fraction += 0.10
    task.step()
    assert task._navigation_hold == 0
    task.fraction = 0.30
    task.step(4)
    assert task._window_hamper_stage == initial_stage
    task.step()
    assert task._window_hamper_stage == initial_stage + 1


def test_reset_on_same_instance_clears_completed_sequence():
    task = Route()
    task.visit("window")
    task.visit("hamper")
    task.fraction = 0.80
    task.step(task.HOLD_STEPS)
    assert task._check_success()
    task._reset_internal()
    assert task._window_hamper_stage == 0
    assert task._navigation_hold == task._opening_hold == 0
    assert not task._check_success()
    task.fraction = 0.80
    task.step(task.HOLD_STEPS)
    assert not task._check_success()


def test_atomic_calibrations_are_copied_and_catalog_is_consistent():
    scene = "FloorPlan323"
    atomic_hamper = deepcopy(IThorBedroomOpenHamperLid.SCENE_SPECS[scene])
    atomic_window = deepcopy(IThorBedroomApproachBedsideWindow.SCENE_SPECS[scene])
    spec = WindowHamper.SCENE_SPECS[scene]
    assert WindowHamper.SUPPORTED_SCENE_IDS == (323,)
    for key, value in atomic_hamper.items():
        if key != "lang":
            assert spec[key] == value
    assert spec["navigation"] == atomic_window
    assert (
        spec["navigation"] is not IThorBedroomApproachBedsideWindow.SCENE_SPECS[scene]
    )
    instance = object.__new__(WindowHamper)
    instance.scene_name = scene
    private = instance.scene_task_spec()
    private["spawn_root_pos"][0] += 1
    private["navigation"]["start_pose"]["xy"][0] += 1
    assert IThorBedroomOpenHamperLid.SCENE_SPECS[scene] == atomic_hamper
    assert IThorBedroomApproachBedsideWindow.SCENE_SPECS[scene] == atomic_window
    assert instance.scene_task_spec() == spec
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    assert [e.cls for e in entries if e.name == WindowHamper.__name__] == [WindowHamper]
    assert validate_task_catalog(entries) == []
