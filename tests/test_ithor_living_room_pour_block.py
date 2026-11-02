"""Focused contracts for replacing cup inversion with pouring the push-task block."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import robocasa.environments.ithor_living_room as room
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.atomic.pour_block_from_basket import (
    IThorLivingRoomPourBlockFromBasket as PourBlock,
)
from robocasa.environments.ithor_living_room.tasks.atomic.push_block_to_target_region import (
    IThorLivingRoomPushBlockToTargetRegion as PushBlock,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_specs,
)


def test_registration_specs_and_block_match_push_task():
    assert REGISTERED_ENVS['IThorLivingRoomPourBlockFromBasket'] is PourBlock
    assert 'IThorLivingRoomPourBlockFromBasket' in room.__all__
    assert 'IThorLivingRoomInvertCup' not in REGISTERED_ENVS
    assert not get_ithor_livingroom_task_specs('invert_cup')
    specs = get_ithor_livingroom_task_specs(PourBlock.SPEC_TASK)
    assert set(specs) == {f'FloorPlan{s}' for s in PourBlock.SUPPORTED_SCENE_IDS}
    assert PourBlock.TARGET_XML == PushBlock.TARGET_XML
    for scene in specs:
        pour, push = object.__new__(PourBlock), object.__new__(PushBlock)
        pour.scene_name = push.scene_name = scene
        assert pour._get_object_adapter() == push._get_object_adapter()
        assert pour._get_obj_cfgs()[1] == push._get_obj_cfgs()[0]
        assert 'lang' not in pour._get_task_spec()
        assert specs[scene]['hold_seconds'] == 1.0


@pytest.mark.parametrize('angle', [0.0, 1.1, np.pi])
def test_complete_exit_rejects_partial_overlap_in_rotated_basket(angle):
    task = object.__new__(PourBlock)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    center = np.array([1.0, 2.0, 0.8])
    task.obj_body_id = {'basket': 0}
    task.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=[center], body_xmat=[rotation]))
    signs = np.array(list(product((-1., 1.), repeat=3)))
    basket = signs * [.164, .123, .10]
    state = {'block': signs * .045}
    task.object_bbox_points = lambda name: (basket if name == 'basket' else state['block']) @ rotation.T + center
    assert task._block_fully_inside_basket()
    assert not task._raw_success()
    state['block'] = signs * .045 + [.19, 0, 0]
    assert not task._raw_success()  # Block center is outside, but one side overlaps.
    state['block'] = signs * .045 + [.22, 0, 0]
    assert task._raw_success()  # No tilt, hand-contact, or grasp prerequisite.
    state['block'] = signs * .045 + [0, 0, .16]
    assert task._raw_success()


def test_exit_must_hold_for_one_second_and_interruption_clears_counter(monkeypatch):
    task = object.__new__(PourBlock)
    task.HOLD_SECONDS = 1.0
    task.control_freq = 20
    monkeypatch.setattr(IThorLivingRoomEnv, '_setup_scene', lambda self: None)
    monkeypatch.setattr(IThorLivingRoomEnv, '_post_action', lambda self, action: None)
    state = {'outside': False}
    task._block_fully_outside_basket = lambda: state['outside']
    task._setup_scene()
    assert not task._check_success()
    state['outside'] = True
    for _ in range(19):
        task._post_action(None)
    assert not task._check_success()
    state['outside'] = False
    task._post_action(None)
    assert task._success_hold_steps == 0
    state['outside'] = True
    for _ in range(20):
        task._post_action(None)
    assert task._check_success()
    task._setup_scene()
    assert not task._check_success()
