"""Ordered navigation/open/close contracts for the living-room drawer cycle."""

from types import SimpleNamespace

import pytest
import robocasa.environments.ithor_living_room as room
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_open_close_drawer import (
    IThorLivingRoomWalkAndOpenCloseDrawer as Cycle,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
    get_ithor_livingroom_task_specs,
)


def make_task(monkeypatch):
    task = object.__new__(Cycle)
    task.scene_name = 'FloorPlan209'
    task.control_freq = 20
    task.sim = object()
    task.target_fixture = object()
    task.PRINT_STAGE_MESSAGES = False
    task._atomic_tasks = task._get_atomic_tasks()
    task._reset_composite_progress()
    task._success_hold = 0
    task._configure_articulation_task()
    state = {'distance': 1.0, 'fraction': .2}
    task.robot_fixture_collision_distance = lambda *args, **kwargs: state['distance']
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state['fraction'])
    monkeypatch.setattr(IThorLivingRoomEnv, '_post_action', lambda self, action: (0., False, {}))
    return task, state


def step(task, count):
    for _ in range(count):
        task._post_action(None)


def test_scene_specs_registration_and_exact_drawer_targets():
    assert REGISTERED_ENVS[Cycle.__name__] is Cycle
    assert Cycle.__name__ in room.__all__
    specs = get_ithor_livingroom_task_specs(Cycle.SPEC_TASK)
    assert set(specs) == {f'FloorPlan{s}' for s in Cycle.SUPPORTED_SCENE_IDS}
    for scene, spec in specs.items():
        old = get_ithor_livingroom_task_spec(scene, 'walk_and_open_drawer')
        for key in ('robot_pose', 'target_fixture', 'target_joint'):
            assert spec[key] == old[key]
        assert spec['init_open_fraction'] == .2
        assert spec['success_open_fraction'] == .8
        assert spec['success_close_fraction'] == .2
        assert spec['open_hold_seconds'] == spec['hold_seconds'] == 1.
        assert spec['minimum_start_distance'] >= spec['approach_distance'] + .15
        assert 'lang' not in spec


def test_cycle_thresholds_hold_interruption_and_reset(monkeypatch):
    task, state = make_task(monkeypatch)
    task._validate_initial_composite_state()
    assert not task._check_success()
    state['distance'] = .351
    step(task, 5)
    assert task.stage_index == 0
    state['distance'] = .35
    step(task, 5)
    assert task.stage_index == 1
    state['fraction'] = .799
    step(task, 20)
    assert task.stage_index == 1
    state['fraction'] = .8
    step(task, 19)
    assert task.stage_index == 1
    step(task, 1)
    assert task.stage_index == 2 and not task._check_success()
    state['fraction'] = .2
    step(task, 19)
    assert not task._check_success()
    state['fraction'] = .201
    step(task, 1)
    assert task._stage_hold_steps == 0
    state['fraction'] = .2
    step(task, 20)
    assert task._check_success()
    task._reset_task_progress()
    assert task.stage_index == 0 and not task._check_success()
    assert not task._open_stage_armed


def test_opening_before_arrival_cannot_skip_open_stage(monkeypatch):
    task, state = make_task(monkeypatch)
    state['fraction'] = .8
    step(task, 20)
    assert task.stage_index == 0
    state['distance'] = .35
    step(task, 5)
    step(task, 20)
    assert task.stage_index == 1
    state['fraction'] = .2
    step(task, 20)
    assert task.stage_index == 1 and not task._check_success()
    state['fraction'] = .8
    step(task, 20)
    assert task.stage_index == 2
    state['fraction'] = .2
    step(task, 20)
    assert task._check_success()


def test_navigation_reports_after_five_consecutive_near_steps(monkeypatch, capsys):
    task, state = make_task(monkeypatch)
    task.PRINT_STAGE_MESSAGES = True

    def distance(fixture, max_distance):
        assert fixture is task.target_fixture
        return state['distance']

    task.robot_fixture_collision_distance = distance
    state['distance'] = .35
    step(task, 4)
    assert task.stage_index == 0
    state['distance'] = .351
    step(task, 1)
    assert task._stage_hold_steps == 0
    state['distance'] = .35
    step(task, 5)
    assert task.stage_index == 1
    assert task.current_atomic_task.name == 'open_drawer'
    assert 'stage 1/3, next stage: open_drawer' in capsys.readouterr().out


def test_reset_rejects_spawn_too_close(monkeypatch):
    task, state = make_task(monkeypatch)
    state['distance'] = .5
    with pytest.raises(RuntimeError, match='start more than'):
        task._validate_initial_composite_state()


@pytest.mark.parametrize('name', [
    'IThorLivingRoomOpenDrawer',
    'IThorLivingRoomCloseDrawer',
    'IThorLivingRoomWalkAndOpenDrawer',
    'IThorLivingRoomWalkAndCloseDrawer',
    'IThorLivingRoomWalkAndOpenCloseDrawer',
    'IThorLivingRoomWalkAndOpenLowDrawer',
])
def test_drawer_tasks_exclude_scene_212(name):
    cls = REGISTERED_ENVS[name]
    assert 212 not in cls.SUPPORTED_SCENE_IDS
    assert name in room.__all__
    specs = get_ithor_livingroom_task_specs(cls.SPEC_TASK)
    assert set(specs) == {f'FloorPlan{s}' for s in cls.SUPPORTED_SCENE_IDS}
    task = object.__new__(cls)
    task.scene_name = 'FloorPlan212'
    with pytest.raises(KeyError, match='FloorPlan212'):
        task._get_articulation_task_spec()
