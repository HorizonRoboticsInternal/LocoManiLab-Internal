"""Scene 323 window visit followed by the existing laundry-hamper action."""

from copy import deepcopy

import numpy as np

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin

from .articulation_base import BedroomArticulationMixin
from .hamper_lid import IThorBedroomOpenHamperLid, _FloorHamperMixin
from .navigation import IThorBedroomApproachBedsideWindow


def _window_hamper_specs():
    scene = "FloorPlan323"
    spec = deepcopy(IThorBedroomOpenHamperLid.SCENE_SPECS[scene])
    spec["navigation"] = deepcopy(IThorBedroomApproachBedsideWindow.SCENE_SPECS[scene])
    spec["robot_root_height"] = spec["navigation"]["robot_root_height"]
    spec["lang"] = (
        "walk to the window beside the bed, then walk to the laundry hamper "
        "and open its lid"
    )
    return {scene: spec}


class _BedsideWindowThenHamperMixin(CompositeStageReporterMixin):
    TASK_FAMILY = "composite"
    ATOMIC_SKILL = "NavigateThenNavigateThenArticulate"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_KIND = "window_then_hamper_open"
    COMPOSITE_ORDERING = (
        "approach_bedside_window",
        "approach_hamper",
        "open_hamper_lid",
    )
    APPROACH_HOLD_STEPS = 5
    HOLD_SECONDS = 1.0
    SUCCESS_DISTANCE = IThorBedroomApproachBedsideWindow.SUCCESS_DISTANCE
    INITIAL_JOINT_FRACTION_TOLERANCE = 0.03
    INITIAL_ROBOT_JITTER_RADIUS = 0.0

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        navigation = self.scene_task_spec()["navigation"]
        self.window_fixture = self.get_fixture(id=navigation["fixture"])
        if (
            self.window_fixture is None
            or self.window_fixture.stable_id != navigation["fixture"]
        ):
            raise LookupError(f"{self.scene_name}: missing exact bedside window")
        # Scene 323's atomic window route preserves every scene fixture.
        if navigation["remove_fixtures"]:
            raise ValueError("the window-hamper route requires an unobstructed scene")
        self.fixture_refs["bedside_window"] = self.window_fixture

    def _setup_scene(self):
        # Reuse the atomic hamper's placement, floor support and base weld first.
        super()._setup_scene()
        navigation = self.scene_task_spec()["navigation"]
        pose = navigation["start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if hasattr(self, "_robot_free_joint_z"):
            del self._robot_free_joint_z
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and IThorBedroomApproachBedsideWindow._calibrate_robot_free_joint_height(
                self
            )
        ):
            raise RuntimeError(f"{self.scene_name}: invalid window route start pose")
        joint = self._get_robot_free_joint()
        joint_id = self.sim.model.joint_name2id(joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.forward()
        if self.count_body_collisions():
            raise RuntimeError(f"{self.scene_name}: window route start collides")
        minimum = float(navigation["minimum_start_distance"])
        distance = self.robot_fixture_collision_distance(
            self.window_fixture, max_distance=max(5.0, minimum + 1.0)
        )
        if not np.isfinite(distance) or distance < minimum:
            raise RuntimeError(
                f"{self.scene_name}: window start distance {distance:.3f} m "
                f"is below {minimum:.3f} m"
            )
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._window_start_distance = float(distance)
        self._reset_task_progress()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        self._window_hamper_stage = 0
        self._navigation_hold = 0
        self._opening_hold = 0
        self._lid_initial_fraction = float(
            self.target_semantic_joint.fraction(self.sim)
        )

    def _near_route_fixture(self, fixture):
        distance = self.robot_fixture_collision_distance(fixture, max_distance=1.0)
        return bool(np.isfinite(distance) and distance <= self.SUCCESS_DISTANCE)

    def _lid_at_reset(self):
        fraction = float(self.target_semantic_joint.fraction(self.sim))
        return bool(
            abs(fraction - self._lid_initial_fraction)
            <= self.INITIAL_JOINT_FRACTION_TOLERANCE
            and not self._raw_success()
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        stage = self._window_hamper_stage
        if stage < 2:
            fixture = self.window_fixture if stage == 0 else self.target_fixture
            ready = self._near_route_fixture(fixture) and self._lid_at_reset()
            self._navigation_hold = self._navigation_hold + 1 if ready else 0
            self._opening_hold = 0
            if self._navigation_hold >= self.APPROACH_HOLD_STEPS:
                self._window_hamper_stage += 1
                self._navigation_hold = 0
                next_stage = self.COMPOSITE_ORDERING[self._window_hamper_stage]
                self._report_stage_complete(stage + 1, 3, next_stage=next_stage)
            return ret
        ready = self._near_route_fixture(self.target_fixture) and self._raw_success()
        self._opening_hold = self._opening_hold + 1 if ready else 0
        if self._opening_hold == self.HOLD_STEPS:
            self._report_stage_complete(3, 3)
        return ret

    def _check_success(self):
        return bool(
            self._window_hamper_stage == 2 and self._opening_hold >= self.HOLD_STEPS
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.scene_task_spec()["lang"],
            stage_order=list(self.COMPOSITE_ORDERING),
            window_fixture=self.window_fixture.stable_id,
            hamper_fixture=self.target_fixture.stable_id,
            start_distance=self._window_start_distance,
            minimum_start_distance=self.scene_task_spec()["navigation"][
                "minimum_start_distance"
            ],
            approach_distance=self.SUCCESS_DISTANCE,
            approach_hold_steps=self.APPROACH_HOLD_STEPS,
            final_hold_seconds=self.HOLD_SECONDS,
            final_hold_steps=self.HOLD_STEPS,
            initial_open_fraction=self._lid_initial_fraction,
            success_fraction_threshold=self.SUCCESS_OPEN_FRACTION,
        )
        return meta


class IThorBedroomApproachBedsideWindowThenOpenHamperLid(
    _BedsideWindowThenHamperMixin,
    _FloorHamperMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    """Visit the bedside window, approach the hamper, then open its exact lid."""

    SPEC_TASK = "hamper_lid"
    BEHAVIOR = "open"
    SUPPORTED_SCENE_IDS = (323,)
    SCENE_SPECS = _window_hamper_specs()


__all__ = ["IThorBedroomApproachBedsideWindowThenOpenHamperLid"]
