"""iTHOR kitchen: walk to the fridge from a calibrated pose and open it."""

from robocasa.environments.ithor_kitchen.tasks.atomic.articulation import (
    IThorKitchenOpenFridge,
)
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)


class IThorKitchenApproachOpenFridge(
    CompositeStageReporterMixin,
    IThorKitchenOpenFridge,
):
    SPEC_TASK = "approach_open_fridge"
    ATOMIC_SKILL = "ApproachOpenFridge"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "navigation_articulation"

    SUCCESS_DISTANCE = 0.75
    COMPOSITE_STAGES = ("approach_fridge", "open_fridge")

    def _reset_task_progress(self):
        self._success_hold = 0
        self._approach_reported = False
        self._open_reported = False

    def _setup_scene(self):
        super()._setup_scene()
        spec = self._get_task_spec()
        self.SUCCESS_DISTANCE = float(
            spec.get("success_distance", self.SUCCESS_DISTANCE)
        )
        self.robot_start_xy = self._robot_base_xy().copy()
        self._reset_task_progress()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _robot_is_near_target(self):
        return bool(
            self.robot_fixture_collision_distance(self.target_fixture)
            <= self.SUCCESS_DISTANCE
        )

    def _raw_success(self):
        fraction = self.target_semantic_joint.fraction(self.sim)
        return bool(
            self.SUCCESS_OPEN_FRACTION <= fraction
            and self._robot_is_near_target()
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        if not self._approach_reported and self._robot_is_near_target():
            self._approach_reported = True
            self._report_stage_complete(1, 2, next_stage="open_fridge")
        if not self._open_reported and self._raw_success():
            self._open_reported = True
            self._report_stage_complete(2, 2)
        return ret

    def _check_success(self):
        return self._raw_success()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "walk to the fridge from far away and open the fridge door"
        ep_meta["composite_stages"] = list(self.COMPOSITE_STAGES)
        ep_meta["target_open_fraction_least"] = self.SUCCESS_OPEN_FRACTION
        if hasattr(self, "robot_start_xy"):
            ep_meta["robot_start_xy"] = [float(v) for v in self.robot_start_xy]
        return ep_meta


__all__ = ["IThorKitchenApproachOpenFridge"]
