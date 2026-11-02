"""Navigate to a low drawer, then crouch and close it."""

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_open_low_drawer import (
    IThorLivingRoomWalkAndOpenLowDrawer,
)


class IThorLivingRoomWalkAndCloseLowDrawer(IThorLivingRoomWalkAndOpenLowDrawer):
    SPEC_TASK = "walk_and_close_low_drawer"
    ATOMIC_SKILL = "ApproachAndCloseLowDrawer"
    COMPOSITE_KIND = "navigate_then_close_low_drawer"
    BEHAVIOR = "close"
    ATOMIC_TASKS = (
        IThorAtomicTask("navigate_to_low_drawer", "_robot_is_near_drawer", hold_steps=5),
        IThorAtomicTask(
            "crouch_and_close_low_drawer", "_close_after_arrival",
            on_enter="_start_close_stage",
        ),
    )

    def _reset_composite_progress(self):
        IThorMultiStageCompositeMixin._reset_composite_progress(self)
        self._close_stage_armed = False

    def _raw_success(self):
        fraction = self.target_semantic_joint.fraction(self.sim)
        return bool(fraction <= self.SUCCESS_CLOSE_FRACTION + 1e-8)

    def _start_close_stage(self):
        self._close_stage_armed = not self._raw_success()

    def _close_after_arrival(self):
        closed = self._raw_success()
        if not closed:
            self._close_stage_armed = True
        return bool(self._close_stage_armed and closed)

    def get_ep_meta(self):
        meta = IThorMultiStageCompositeMixin.get_ep_meta(self)
        if self.scene_name == "FloorPlan226":
            lang = (
                "walk to the coffee table, then crouch down and close the "
                "second drawer from the top on its front side"
            )
        elif self.scene_name == "FloorPlan210":
            lang = "walk to the lower-right drawer of the dresser, then crouch down and close it"
        else:
            lang = "walk to the lower-left drawer of the dresser, then crouch down and close it"
        meta.update(
            lang=lang,
            composite_stages=[task.name for task in self._atomic_tasks],
            composite_ordering=self.COMPOSITE_ORDERING,
            start_distance=float(self._drawer_start_distance),
            approach_distance=float(self._get_task_spec()["approach_distance"]),
            init_open_fraction=self.OPEN_INIT_FRACTION,
            success_close_fraction=self.SUCCESS_CLOSE_FRACTION,
            success_definition=(
                "reach the named low drawer within 0.35 m for 5 control steps, "
                "then close the same joint to at most 20% for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomWalkAndCloseLowDrawer"]
