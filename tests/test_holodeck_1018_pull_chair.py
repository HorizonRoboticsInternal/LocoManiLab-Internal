"""Task-local chair replacement shared by 1018 pull-chair workflows."""

from pathlib import Path

import yaml

from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.multi_hop.navigation import (
    HolodeckApproachDeskThenPullChair,
    HolodeckApproachTrashCanThenApproachDeskThenPullChair,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckApproachDeskPullChairThenOpenLaptop,
    HolodeckPullChairThenApproachTrashCanThenPickBox,
)


SPECS_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa/models/assets/scenes/holodeck_task_specs.yaml"
)


def test_1018_pull_and_composites_use_1664_sliding_chair_without_changing_push():
    scenes = yaml.safe_load(SPECS_PATH.read_text())["scenes"]
    pull = scenes["train_1018"]["pull_chair_from_desk"]
    reference = scenes["train_1664"]["pull_chair_from_desk"]
    baked_chair = scenes["train_1018"]["push_chair_to_desk"]["chair_body"]

    assert pull["fixture_mode"] == reference["fixture_mode"] == "spawn_asset"
    for key in ("chair_asset", "chair_body", "chair_joint"):
        assert pull[key] == reference[key]
    assert pull["remove_scene_bodies"] == [baked_chair]
    assert baked_chair not in pull["chair_body"]
    assert pull["chair_start_pose"]["pos"][:2] == [7.4, 0.72]
    assert pull["target_xy"] == [6.0, 0.72]
    assert HolodeckPullChairFromDesk.PUSH_SPEC_TASK == "pull_chair_from_desk"

    # In 1018 none of these consumers selects a separate chair spec. Their
    # pull stage therefore shares the atomic task's asset and body removal.
    for task_cls in (
        HolodeckApproachDeskThenPullChair,
        HolodeckApproachTrashCanThenApproachDeskThenPullChair,
        HolodeckApproachDeskPullChairThenOpenLaptop,
        HolodeckPullChairThenApproachTrashCanThenPickBox,
    ):
        assert 1018 in task_cls.SUPPORTED_SCENE_IDS
        assert any(
            stage.task_cls is HolodeckPullChairFromDesk
            for stage in task_cls.ATOMIC_TASKS
        )
    assert 1018 not in HolodeckApproachDeskThenPullChair.COMPOSITE_SCENE_IDS
    assert 1018 not in HolodeckApproachDeskPullChairThenOpenLaptop.COMPOSITE_SCENE_IDS
    assert HolodeckApproachDeskPullChairThenOpenLaptop.CHAIR_SPEC_TASK == "pull_chair_from_desk"
