"""Navigate to the target fixture, open it, and close it."""

from robocasa.environments.ithor_composite.base import IThorAtomicTask
from robocasa.environments.ithor_kitchen.tasks.atomic.articulation import (
    IThorKitchenOpenCabinet,
    IThorKitchenOpenDrawer,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_step_manipulation import (
    _OrderedArticulationWorkflowMixin,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_specs


class _ApproachOpenCloseMixin(_OrderedArticulationWorkflowMixin):
    COMPOSITE_KIND = "navigate_open_close"
    REQUIRED_SPEC_FIELDS = (
        "robot_pose", "navigation_goal_pose", "navigation_xy_tolerance",
        "navigation_yaw_tolerance", "target_fixture", "target_joint",
        "init_open_fraction", "success_open_fraction", "success_close_fraction",
        "open_hold_seconds", "hold_seconds",
    )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask("navigate_to_" + self.TARGET_NAME, "_at_operating_pose"),
            IThorAtomicTask(
                "open_" + self.TARGET_NAME, "_open_after_arrival",
                on_enter="_arm_active_stage_condition",
            ),
            IThorAtomicTask(
                "close_" + self.TARGET_NAME, "_close_active_stage_condition",
                on_enter="_arm_active_stage_condition",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[1]:
            return max(1, int(round(
                float(self._get_task_spec()["open_hold_seconds"]) * self.control_freq
            )))
        return super()._required_stage_hold_steps(task)

    def _at_operating_pose(self):
        # Arrival is proximity to the actual fixture, not a fixed base pose.
        return self._robot_near_articulation()

    def _open_after_arrival(self):
        return self._active_stage_condition(self._articulation_is_open())


class IThorKitchenApproachDrawerAndOpenClose(
    _ApproachOpenCloseMixin, IThorKitchenOpenDrawer,
):
    SPEC_TASK = "approach_drawer_and_open_close"
    ATOMIC_SKILL = "ApproachDrawerAndOpenClose"
    TARGET_NAME = "drawer"
    TASK_LANG = "walk to the drawer, open it, then close the same drawer"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)


class IThorKitchenApproachCabinetAndOpenClose(
    _ApproachOpenCloseMixin, IThorKitchenOpenCabinet,
):
    SPEC_TASK = "approach_cabinet_and_open_close"
    ATOMIC_SKILL = "ApproachCabinetAndOpenClose"
    TARGET_NAME = "cabinet"
    TASK_LANG = "walk to the cabinet, open its door, then close the same door"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)


__all__ = [
    "IThorKitchenApproachDrawerAndOpenClose",
    "IThorKitchenApproachCabinetAndOpenClose",
]
