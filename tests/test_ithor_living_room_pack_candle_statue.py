"""Contracts for the two-item living-room packing task."""

from itertools import product
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import robocasa.environments.ithor_living_room as room
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.composite.pack_candle_statue import (
    IThorLivingRoomPackCandleStatueIntoBasket as Pack,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_specs,
)


def test_two_item_specs_assets_metadata_and_registration(monkeypatch):
    assert REGISTERED_ENVS[Pack.__name__] is Pack
    assert Pack.__name__ in room.__all__
    assert 'IThorLivingRoomPackCandleStatueCupIntoBasket' not in REGISTERED_ENVS
    assert not get_ithor_livingroom_task_specs('pack_candle_statue_cup_into_basket')
    specs = get_ithor_livingroom_task_specs(Pack.SPEC_TASK)
    assert set(specs) == {f'FloorPlan{s}' for s in Pack.SUPPORTED_SCENE_IDS}
    monkeypatch.setattr(IThorLivingRoomEnv, 'get_ep_meta', lambda self: {})
    for scene, spec in specs.items():
        task = object.__new__(Pack)
        task.scene_name = scene
        assert set(spec['object_poses']) == {'candle', 'statue'}
        assert {cfg['name'] for cfg in task._get_obj_cfgs()} == {'basket', 'candle'}
        assert task.ITEM_NAMES == ('candle', 'statue')
        meta = task.get_ep_meta()
        assert 'cup' not in meta['lang']
        assert len(meta['composite_stages']) == 2
        assert meta['composite_ordering'] == 'any'


@pytest.mark.parametrize('first,second', [('candle', 'statue'), ('statue', 'candle')])
def test_both_items_required_in_either_order_with_two_stage_messages(
    monkeypatch, first, second,
):
    task = object.__new__(Pack)
    task.HOLD_SECONDS = 1.0
    task.control_freq = 20
    monkeypatch.setattr(IThorLivingRoomEnv, '_setup_scene', lambda self: None)
    monkeypatch.setattr(IThorLivingRoomEnv, '_post_action', lambda self, action: None)
    task._setup_scene()
    packed = set()
    task._item_is_packed = lambda name: name in packed
    reports = []
    task._report_stage_complete = lambda *args: reports.append(args)
    assert not task._check_success()
    packed.add(first)
    for _ in range(20):
        task._post_action(None)
    assert not task._check_success()
    assert reports == [(1, 2, f'put_{second}_in_basket')]
    packed.add(second)
    for _ in range(19):
        task._post_action(None)
    assert not task._check_success()
    packed.remove(first)
    task._post_action(None)
    assert task._success_hold == 0
    packed.add(first)
    for _ in range(20):
        task._post_action(None)
    assert task._check_success()
    assert reports == [(1, 2, f'put_{second}_in_basket'), (2, 2, None)]


@pytest.fixture
def packing_task():
    task = object.__new__(Pack)
    task.scene_name = "FloorPlan202"
    spec = task._get_task_spec()
    marker = ET.parse(task._asset_path(task.BASKET_XML)).find(
        ".//geom[@name='reg_bbox']"
    )
    center = np.fromstring(marker.get("pos"), sep=" ")
    half = np.fromstring(marker.get("size"), sep=" ")
    corners = (
        center + np.array(list(product((-1, 1), repeat=3))) * half
    ) * spec["basket_scale"]
    task.objects = {
        "basket": SimpleNamespace(get_bbox_points=lambda: corners),
        "statue": object(),
    }
    task.obj_body_id = {"basket": 0, "statue": 1}
    task.sim = SimpleNamespace(data=SimpleNamespace(
        body_xpos=np.array([[1., 2., 0.75], [1., 2., 0.85]]),
        body_xmat=np.tile(np.eye(3).ravel(), (2, 1)),
    ))
    task.success_xy_tolerance = spec["success_xy_tolerance"]
    task.HOLD_STEPS = 20
    task._success_hold = 0
    task._packed_items = set()
    # Candle can rest on the statue without touching the basket directly.
    task.check_contact = lambda obj, basket: obj is task.objects["statue"]
    task._report_stage_complete = lambda *args, **kwargs: None
    return task


def _place_candle_center(task, local_center, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    task.sim.data.body_xmat[0] = rotation.ravel()
    world_center = task.sim.data.body_xpos[0] + rotation @ local_center
    offsets = np.array(list(product((-0.03, 0.03), repeat=3)))
    task.object_bbox_points = lambda name: world_center + offsets


@pytest.mark.parametrize("yaw", [0., np.pi / 2.])
def test_off_center_candle_without_basket_contact_completes(packing_task, monkeypatch, yaw):
    task = packing_task
    # 20 cm exceeds the old 18 cm cutoff but fits the scaled basket.
    _place_candle_center(task, np.array([0.20, 0., 0.10]), yaw)
    monkeypatch.setattr(IThorLivingRoomEnv, "_post_action", lambda self, action: None)
    for _ in range(19):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()


@pytest.mark.parametrize("center", [(0.30, 0., 0.10), (0., 0., 0.40), (0., 0., -0.05)])
def test_candle_outside_above_or_below_basket_fails(packing_task, center):
    _place_candle_center(packing_task, np.array(center), np.pi / 2.)
    assert not packing_task._item_is_packed("candle")


def test_statue_keeps_existing_contact_requirement(packing_task):
    assert packing_task._item_is_packed("statue")
    packing_task.check_contact = lambda *args: False
    assert not packing_task._item_is_packed("statue")
