"""Scene 422's upper cabinet is a separate, exact-joint atomic task."""

from types import SimpleNamespace

import pytest

from robocasa.environments.ithor_bathroom import tasks
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_bathroom.tasks.storage_articulation import (
    IThorBathroomOpenUpperCabinetDoor as Task,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)
from robocasa.models.scenes.thor_semantics import SemanticJoint


def test_upper_cabinet_registration_is_separate_from_existing_tasks():
    assert Task.SUPPORTED_SCENE_IDS == (422,)
    assert Task in tasks.ATOMIC_TASK_CLASSES
    assert Task.__name__ in tasks.__all__
    assert validate_task_catalog(discover_task_catalog((BATHROOM_PACKAGE,))) == []
    spec = Task.SCENE_SPECS["FloorPlan422"]
    assert spec["fixture"] == "FloorPlan422:Cabinet_a6b630aa_2"
    assert spec["joint"] == ("cabinet_408a24f2d1bddd4eabb5091faf214ac9_1_1_0_joint_0")
    assert spec["joint_type"] == "hinge"
    assert spec["init_open_fraction"] == 0.20
    assert spec["success_open_fraction"] == 0.80
    for cls in (
        tasks.IThorBathroomOpenCabinetDoor,
        tasks.IThorBathroomCloseCabinetDoor,
    ):
        assert cls.SUPPORTED_SCENE_IDS == (405, 407, 418)
        assert "FloorPlan422" not in cls.SCENE_SPECS


@pytest.mark.parametrize("control_freq", (10, 20))
def test_exact_upper_door_threshold_hold_and_interruption(monkeypatch, control_freq):
    task = object.__new__(Task)
    task.scene_name = "FloorPlan422"
    spec = task.scene_task_spec()
    task.OPEN_INIT_FRACTION = spec["init_open_fraction"]
    task.SUCCESS_OPEN_FRACTION = spec["success_open_fraction"]
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    task._success_hold = 0
    joint = SemanticJoint(
        stable_id=f"{spec['fixture']}:joint:door",
        name=spec["joint"],
        role="door",
        range=(-1.5708, 0.0),
        closed_qpos=0.0,
        open_qpos=-1.5708,
    )
    qpos = {joint.name: task._initial_joint_qpos(joint), "other_door": -1.5708}
    task.target_semantic_joint = joint
    task.sim = SimpleNamespace(data=SimpleNamespace(get_joint_qpos=qpos.__getitem__))
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda self, action: None)

    assert joint.fraction(task.sim) == pytest.approx(0.20)
    assert not task._raw_success()  # Opening another door cannot satisfy the task.
    qpos[joint.name] = 0.799 * joint.open_qpos
    task._post_action(None)
    assert not task._raw_success()
    assert task._success_hold == 0

    qpos[joint.name] = 0.80 * joint.open_qpos
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert task._raw_success()
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()

    qpos[joint.name] = 0.79 * joint.open_qpos
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()
