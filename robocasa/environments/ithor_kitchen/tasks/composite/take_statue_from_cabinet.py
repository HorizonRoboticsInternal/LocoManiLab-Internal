"""Take the cabinet statue to a calibrated region, then close the door."""

import numpy as np

from robocasa.environments.ithor_composite.base import IThorAtomicTask
from robocasa.environments.ithor_kitchen.tasks.composite.multi_step_manipulation import (
    IThorKitchenApproachCabinetAndStoreStatue,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_specs
from robocasa.models.scenes.thor_semantics import SemanticRegion


class IThorKitchenTakeStatueFromCabinet(
    IThorKitchenApproachCabinetAndStoreStatue,
):
    """Start at an open cabinet; place its statue on the table before closing."""

    ATOMIC_SKILL = "TakeStatueFromCabinet"
    COMPOSITE_KIND = "place_close"
    SPEC_TASK = "take_statue_from_cabinet"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    REQUIRED_SPEC_FIELDS = (
        "hold_seconds",
        "robot_pose",
        "target_fixture",
        "target_joint",
        "init_open_fraction",
        "success_close_fraction",
        "cabinet_receptacle_region",
        "statue_pos",
        "statue_quat",
        "table_fixture",
        "success_region",
    )
    CABINET_REGION_RIM_TOL = 0.04

    def _get_task_lang(self):
        return (
            "take the statue out of the open cabinet, place it on the table, "
            "then close the cabinet door"
        )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "put_statue_on_table",
                "_statue_on_table",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            ),
            IThorAtomicTask(
                "close_cabinet",
                "_cabinet_closed_with_statue_on_table",
                task_cls="IThorKitchenCloseCabinet",
                on_enter="_arm_active_stage_condition",
            ),
        )

    def _finalize_articulation_robot_pose(self, distance):
        del distance
        self.sim.forward()
        if self.count_body_collisions() == 0:
            return True
        return self._calibrate_robot_free_joint_height()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.table_fixture = self.get_fixture(id=spec["table_fixture"])
        if self.table_fixture is None:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing table fixture"
            )
        target = spec["success_region"]
        self.success_region = SemanticRegion(
            stable_id=f"{self.table_fixture.stable_id}:region:success:statue_target",
            fixture_id=self.table_fixture.stable_id,
            kind="interior",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.asarray(target["half_size"], dtype=float),
            _world_center=np.asarray(target["center"], dtype=float),
            _world_yaw=float(target["yaw"]),
        )
        self.fixture_refs["table"] = self.table_fixture
        self.region_refs["success"] = self.success_region

    def _before_object_placement(self):
        # This task uses a scene-calibrated cabinet pose. The exported receptacle
        # lower face is slightly above the real collidable shelf, so probing from
        # the semantic volume can reject a physically valid in-cabinet pose.
        super()._before_object_placement()
        if not self._statue_in_cabinet():
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: statue outside cabinet"
            )

    def _statue_in_cabinet(self):
        return bool(
            self.object_in_region(
                "statue",
                self.cabinet_receptacle_region,
                rim_tol=self.CABINET_REGION_RIM_TOL,
            )
            and self._fixture_contact("statue", self.cabinet_receptacle_fixture)
            and self.object_released("statue")
        )

    def _statue_in_success_region(self):
        return bool(
            self.object_in_region(
                "statue", self.success_region, rim_tol=0.0, xy_tol=0.0
            )
            and self._fixture_contact("statue", self.table_fixture)
            and self.object_released("statue")
        )

    def _statue_on_table(self):
        return self._statue_in_success_region()

    def _cabinet_closed_with_statue_on_table(self):
        return bool(
            self._close_active_stage_condition() and self._statue_in_success_region()
        )

    def _check_success(self):
        return bool(
            super()._check_success()
            and self._statue_in_success_region()
            and self._articulation_is_closed()
        )


__all__ = ["IThorKitchenTakeStatueFromCabinet"]
