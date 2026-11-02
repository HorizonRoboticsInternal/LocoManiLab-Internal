"""Navigate to a scene-specific low drawer, then crouch and open it."""

from dataclasses import replace

import numpy as np

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.articulation import (
    IThorLivingRoomOpenDrawer,
)


class IThorLivingRoomWalkAndOpenLowDrawer(
    IThorMultiStageCompositeMixin, IThorLivingRoomOpenDrawer,
):
    SUPPORTED_SCENE_IDS = (209, 210, 226)
    SPEC_TASK = "walk_and_open_low_drawer"
    ATOMIC_SKILL = "ApproachAndOpenLowDrawer"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    COMPOSITE_KIND = "navigate_then_open_low_drawer"
    COMPOSITE_ORDERING = "sequential"
    INITIAL_POSE_JITTER = False
    ATOMIC_TASKS = (
        IThorAtomicTask("navigate_to_low_drawer", "_robot_is_near_drawer", hold_steps=5),
        IThorAtomicTask(
            "crouch_and_open_low_drawer", "_open_after_arrival",
            on_enter="_start_open_stage",
        ),
    )

    def _setup_scene(self):
        super()._setup_scene()
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        drawer_body = self.sim.model.body_id2name(
            int(self.sim.model.jnt_bodyid[joint_id])
        )
        # Restrict navigation distance to the selected drawer and its handle.
        self._navigation_drawer = replace(
            self.target_fixture, body_names=[drawer_body],
        )
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)

    def _reset_composite_progress(self):
        super()._reset_composite_progress()
        self._open_stage_armed = False

    def _validate_initial_composite_state(self):
        super()._validate_initial_composite_state()
        self._drawer_start_distance = self.robot_fixture_collision_distance(
            self._navigation_drawer, max_distance=4.0,
        )
        minimum = float(self._get_task_spec()["minimum_start_distance"])
        if (
            not np.isfinite(self._drawer_start_distance)
            or self._drawer_start_distance <= minimum
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot must start more than "
                f"{minimum:.2f} m from the drawer"
            )

    def _robot_is_near_drawer(self):
        distance = self.robot_fixture_collision_distance(
            self._navigation_drawer, max_distance=1.5,
        )
        return bool(
            np.isfinite(distance)
            and distance <= float(self._get_task_spec()["approach_distance"])
        )

    def _start_open_stage(self):
        self._open_stage_armed = not self._raw_success()

    def _raw_success(self):
        # Exact 0.16 m / 0.20 m travel can round just below 0.8.
        fraction = self.target_semantic_joint.fraction(self.sim)
        return bool(fraction >= self.SUCCESS_OPEN_FRACTION - 1e-8)

    def _open_after_arrival(self):
        opened = self._raw_success()
        if not opened:
            self._open_stage_armed = True
        return bool(self._open_stage_armed and opened)

    def _required_stage_hold_steps(self, task):
        if task.hold_steps is not None:
            return int(task.hold_steps)
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        if self.scene_name == "FloorPlan226":
            lang = (
                "walk to the coffee table, then crouch down and open the "
                "second drawer from the top on its front side"
            )
        elif self.scene_name == "FloorPlan210":
            lang = (
                "walk to the lower-right drawer of the dresser, then crouch "
                "down and open it"
            )
        else:
            lang = (
                "walk to the lower-left drawer of the dresser, then crouch "
                "down and open it"
            )
        meta.update(
            lang=lang,
            composite_stages=[task.name for task in self._atomic_tasks],
            composite_ordering=self.COMPOSITE_ORDERING,
            start_distance=float(self._drawer_start_distance),
            approach_distance=float(self._get_task_spec()["approach_distance"]),
            init_open_fraction=self.OPEN_INIT_FRACTION,
            success_open_fraction=self.SUCCESS_OPEN_FRACTION,
            success_definition=(
                "reach the named lower drawer within 0.35 m for 5 control steps, "
                "then open the same joint to at least 80% for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomWalkAndOpenLowDrawer"]
