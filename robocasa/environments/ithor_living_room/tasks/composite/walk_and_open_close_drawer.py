"""Navigate to one living-room drawer, open it, then close it."""

import numpy as np

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.articulation import (
    IThorLivingRoomOpenDrawer,
)


class IThorLivingRoomWalkAndOpenCloseDrawer(
    IThorMultiStageCompositeMixin, IThorLivingRoomOpenDrawer,
):
    """Complete navigation, 80% opening, and 20% closing in order."""

    SPEC_TASK = "walk_and_open_close_drawer"
    ATOMIC_SKILL = "ApproachAndOpenCloseDrawer"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_KIND = "navigate_then_open_close_drawer"
    COMPOSITE_ORDERING = "sequential"
    INITIAL_POSE_JITTER = False
    ATOMIC_TASKS = (
        IThorAtomicTask("navigate_to_drawer", "_robot_is_near_drawer", hold_steps=5),
        IThorAtomicTask(
            "open_drawer", "_open_after_arrival", on_enter="_start_open_stage",
        ),
        IThorAtomicTask("close_drawer", "_drawer_is_closed"),
    )

    def _configure_articulation_task(self):
        super()._configure_articulation_task()
        spec = self._get_articulation_task_spec()
        self.SUCCESS_CLOSE_FRACTION = float(spec["success_close_fraction"])
        if not 0.0 <= self.SUCCESS_CLOSE_FRACTION < self.SUCCESS_OPEN_FRACTION:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid close fraction")

    def _reset_composite_progress(self):
        super()._reset_composite_progress()
        self._open_stage_armed = False

    def _validate_initial_composite_state(self):
        super()._validate_initial_composite_state()
        spec = self._get_articulation_task_spec()
        self._drawer_start_distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=4.0,
        )
        minimum = float(spec["minimum_start_distance"])
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
            self.target_fixture, max_distance=1.5,
        )
        return bool(
            np.isfinite(distance)
            and distance <= float(self._get_articulation_task_spec()["approach_distance"])
        )

    def _start_open_stage(self):
        # An opening completed before navigation must be undone and repeated.
        self._open_stage_armed = not self._raw_success()

    def _open_after_arrival(self):
        opened = self._raw_success()
        if not opened:
            self._open_stage_armed = True
        return bool(self._open_stage_armed and opened)

    def _drawer_is_closed(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim) <= self.SUCCESS_CLOSE_FRACTION
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[0]:
            return int(task.hold_steps)
        spec = self._get_articulation_task_spec()
        seconds = (
            spec["open_hold_seconds"]
            if task is self._atomic_tasks[1]
            else spec["hold_seconds"]
        )
        return max(1, int(round(float(seconds) * self.control_freq)))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        spec = self._get_articulation_task_spec()
        meta.update(
            lang="walk to the drawer, open it to at least 80%, then close it to at most 20%",
            composite_stages=[task.name for task in self._atomic_tasks],
            composite_ordering=self.COMPOSITE_ORDERING,
            start_distance=float(self._drawer_start_distance),
            approach_distance=float(spec["approach_distance"]),
            success_open_fraction=self.SUCCESS_OPEN_FRACTION,
            success_close_fraction=self.SUCCESS_CLOSE_FRACTION,
            success_definition=(
                "reach the named drawer for 5 control steps, then open the same "
                f"joint to at least 80% for {spec['open_hold_seconds']:g} seconds, "
                f"then close it to at most 20% for {spec['hold_seconds']:g} seconds"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomWalkAndOpenCloseDrawer"]
