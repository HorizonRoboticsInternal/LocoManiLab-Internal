"""Discard an apple into a nearby trash can, optionally after opening a drawer."""

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_composite.base import IThorAtomicTask
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.articulation import (
    IThorKitchenOpenDrawer,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.coffee_drink_to_trash import (
    IThorKitchenCoffeeDrinkToTrash,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_step_manipulation import (
    _OrderedKitchenWorkflowMixin,
)
from robocasa.environments.ithor_kitchen.tasks.composite.open_drawer_take_apple import (
    _CalibrateCollidingExactPoseMixin,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_specs
from robocasa.models.scenes.thor_semantics import SemanticRegion


class _AppleTrashWorkflowMixin(
    _OrderedKitchenWorkflowMixin, SupportContactPlacementMixin
):
    SUPPORTED_SCENE_IDS = (7, 13, 24, 25)
    COMPOSITE_KIND = "discard"
    APPLE_XML = "objaverse/apple/apple_0/model.xml"
    REQUIRED_SPEC_FIELDS = (
        "robot_pose",
        "apple_pos",
        "source_fixture",
        "trash_fixture",
        "trash_region",
        "trash_body_pos",
        "hold_seconds",
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        super().__init__(
            *args,
            scene_ids=self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids,
            **kwargs,
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_atomic_tasks(self):
        return (IThorAtomicTask("put_apple_in_trash", "_apple_in_trash"),)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.trash_fixture = self.get_fixture(id=spec["trash_fixture"])
        self.source_fixture = self.get_fixture(id=spec["source_fixture"])
        if self.trash_fixture is None or self.source_fixture is None:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing source/trash"
            )
        self.trash_region = self.trash_fixture.regions[spec["trash_region"]]
        IThorKitchenCoffeeDrinkToTrash._set_trash_body_pos(self, spec["trash_body_pos"])
        self.fixture_refs.update(
            trash_can=self.trash_fixture, apple_source=self.source_fixture
        )
        self.region_refs["trash_receptacle"] = self.trash_region

    def _get_obj_cfgs(self):
        return [dict(name="apple", info=dict(mjcf_path=self.APPLE_XML))]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name=f"Manual{self.ATOMIC_SKILL}Sampler")

    def _semantic_placements_are_valid(self):
        return True

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        pos = np.asarray(spec["apple_pos"], dtype=float)
        support_pos = pos.copy()
        # FloorPlan24's calibration records the apple root height, which is
        # above its actual contact point on the dining table.
        support_pos[2] = float(spec.get("apple_support_z", pos[2]))
        support = SemanticRegion(
            stable_id=f"{self.source_fixture.stable_id}:region:apple_start",
            fixture_id=self.source_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([0.12, 0.12, 0.02]),
            _world_center=support_pos,
            _world_yaw=0.0,
        )
        if not self.place_object_on_support_contact(
            "apple",
            support,
            self.source_fixture,
            xy=pos[:2],
            inward_fractions=(0.0,),
            probe_above=0.03,
            probe_below=0.04,
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: apple has no support"
            )

    def _apple_in_trash(self):
        return bool(
            self.object_in_region(
                "apple",
                self.trash_region,
                rim_tol=IThorKitchenCoffeeDrinkToTrash.TRASH_Z_TOL,
                xy_tol=IThorKitchenCoffeeDrinkToTrash.TRASH_XY_TOL,
            )
            and self.object_released("apple")
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            locomotion_required=False,
            trash_fixture=self.trash_fixture.stable_id,
        )
        return meta


class IThorKitchenPickAppleAndTurnToTrash(_AppleTrashWorkflowMixin, IThorKitchenEnv):
    SPEC_TASK = "pick_apple_and_turn_to_trash"
    ATOMIC_SKILL = "PickAppleAndTurnToTrash"
    TASK_LANG = (
        "pick up the apple from the table, turn around in place, "
        "and drop the apple into the trash can behind you"
    )
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)


class IThorKitchenOpenDrawerAndTurnAppleToTrash(
    _CalibrateCollidingExactPoseMixin,
    _AppleTrashWorkflowMixin,
    IThorKitchenOpenDrawer,
):
    SPEC_TASK = "open_drawer_and_turn_apple_to_trash"
    COMPOSITE_KIND = "open_drawer_then_discard"
    ATOMIC_SKILL = "OpenDrawerAndTurnAppleToTrash"
    TASK_LANG = (
        "open the drawer, take out the apple, turn around in place, "
        "and drop the apple into the trash can behind you"
    )
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    REQUIRED_SPEC_FIELDS = _AppleTrashWorkflowMixin.REQUIRED_SPEC_FIELDS + (
        "target_fixture",
        "target_joint",
        "init_open_fraction",
        "success_open_fraction",
    )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "open_drawer",
                "_drawer_is_open",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            ),
        ) + super()._get_atomic_tasks()

    def _drawer_is_open(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            >= float(self._get_task_spec()["success_open_fraction"])
        )


__all__ = [
    "IThorKitchenPickAppleAndTurnToTrash",
    "IThorKitchenOpenDrawerAndTurnAppleToTrash",
]
