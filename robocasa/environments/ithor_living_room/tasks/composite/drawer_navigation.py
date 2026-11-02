"""Navigation gate shared by drawer composite tasks."""

import numpy as np

from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)


class DrawerNavigationGateMixin(CompositeStageReporterMixin):
    """Require reaching the exact drawer before changing its target state."""

    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    COMPOSITE_ORDERING = "sequential"
    APPROACH_HOLD_STEPS = 5

    def _setup_scene(self):
        super()._setup_scene()
        spec = self._get_articulation_task_spec()
        self._drawer_start_distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=4.0
        )
        minimum = float(spec["minimum_start_distance"])
        if not (
            np.isfinite(self._drawer_start_distance)
            and self._drawer_start_distance > minimum
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot must start more than "
                f"{minimum:.2f} m from the drawer"
            )
        self._route_stage = 0
        self._approach_hold = 0
        self._operation_hold = 0

    def _robot_is_near_drawer(self):
        distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=1.5
        )
        return bool(
            np.isfinite(distance)
            and distance
            <= float(self._get_articulation_task_spec()["approach_distance"])
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        near = self._robot_is_near_drawer()
        if self._route_stage == 0:
            self._approach_hold = self._approach_hold + 1 if near else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._route_stage = 1
                self._report_stage_complete(
                    1,
                    2,
                    next_stage=f"{self.BEHAVIOR}_drawer",
                )
            self._operation_hold = 0
            return ret
        valid = self._raw_success()
        self._operation_hold = self._operation_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return self._route_stage == 1 and self._operation_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang=(
                "walk to the drawer and close it"
                if self.BEHAVIOR == "close"
                else "walk to the drawer and open it"
            ),
            start_distance=float(self._drawer_start_distance),
            approach_distance=float(
                self._get_articulation_task_spec()["approach_distance"]
            ),
            composite_stages=[
                "walk to the named drawer",
                f"{self.BEHAVIOR} the same drawer",
            ],
        )
        return ep_meta


__all__ = ["DrawerNavigationGateMixin"]
