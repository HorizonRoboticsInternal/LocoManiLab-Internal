"""Composite progress must restart after normal and in-place resets."""

import pytest

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_kitchen import (
    IThorKitchenAppleBananaPlateSwap,
    IThorKitchenAppleCabinetPnP,
    IThorKitchenApproachOpenFridge,
    IThorKitchenApproachOpenOven,
)


@pytest.mark.parametrize(
    "task_cls",
    (
        IThorKitchenApproachOpenFridge,
        IThorKitchenApproachOpenOven,
        IThorKitchenAppleCabinetPnP,
    ),
)
@pytest.mark.parametrize("reset_method", ("_reset_internal", "_reset_task_progress"))
def test_stage_output_restarts_after_reset(task_cls, reset_method, monkeypatch, capsys):
    monkeypatch.setattr(RoomOpenArticulation, "_reset_internal", lambda self: None)
    monkeypatch.setattr(
        RoomOpenArticulation, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = object.__new__(task_cls)
    task._robot_is_near_target = lambda: True
    task._raw_success = lambda: True
    task.g1_hand_grasping_object = lambda name: True
    task._apple_inside_cabinet = lambda: True
    task.object_released = lambda name: True
    task._task_success_condition = lambda: True

    for _ in range(2):
        task._success_hold = 20
        getattr(task, reset_method)()
        assert task._success_hold == 0
        if task_cls is IThorKitchenAppleCabinetPnP:
            assert task._task_success_hold_steps == 0
        task._post_action(None)
        task._post_action(None)
        output = capsys.readouterr().out
        total = len(task.COMPOSITE_STAGES)
        for stage in range(1, total + 1):
            assert output.count(f"stage {stage}/{total}") == 1


def test_plate_swap_in_place_reset_clears_success_timer():
    task = object.__new__(IThorKitchenAppleBananaPlateSwap)
    task.control_freq = 20
    task.success_hold_seconds = 1.0
    task._success_hold_steps = 20
    assert task._check_success()

    task._reset_task_progress()

    assert task._success_hold_steps == 0
    assert not task._check_success()
