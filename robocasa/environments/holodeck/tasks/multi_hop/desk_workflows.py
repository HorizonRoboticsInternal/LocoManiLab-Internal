"""Direct walk-to-desk composite workflows for calibrated Holodeck scenes."""

from __future__ import annotations

import numpy as np

from robocasa.environments.holodeck.tasks.approach_desk import HolodeckApproachDesk
from robocasa.environments.holodeck.tasks.articulation import HolodeckCloseDrawer
from robocasa.environments.holodeck.tasks.mug_lid import HolodeckOpenMugLid
from robocasa.environments.holodeck.tasks.move_cup import HolodeckMoveCupToCenter
from robocasa.environments.holodeck.tasks.pick_cup_from_desk import (
    HolodeckPickCupFromDesk,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckApproachDeskWorkflowMixin:
    """Share the calibrated remote spawn and ordered desk approach stage."""

    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    NAVIGATION_SPEC_TASK = "approach_desk"
    COMPOSITE_ROBOT_POSE_KEY = "desk_composite_robot_pose"
    MIN_APPROACH_DISPLACEMENT = 0.15
    STAGE_HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.desk_fixture = self.get_fixture(id=self.navigation_spec["desk_body"])
        if self.desk_fixture is None:
            return
        self.fixture_refs["approach_desk"] = self.desk_fixture
        pose = self.navigation_spec.get(
            self.COMPOSITE_ROBOT_POSE_KEY,
            self.navigation_spec.get(
                "desk_composite_robot_pose", self.navigation_spec["robot_pose"]
            ),
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.desk_fixture is not None)

    def _setup_scene(self):
        super()._setup_scene()
        distance = self.robot_fixture_collision_distance(
            self.desk_fixture,
            max_distance=self.success_radius + 1.0,
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: desk composite spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _begin_desk_stage(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _desk_reached(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy) >= self.MIN_APPROACH_DISPLACEMENT:
            self._desk_stage_moved = True
        return bool(
            self._desk_stage_moved
            and self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.success_radius + 0.25,
            )
            <= self.success_radius
        )

    def _required_stage_hold_steps(self, task):
        inherited = super()._required_stage_hold_steps(task)
        return max(
            inherited,
            max(1, int(round(self.STAGE_HOLD_SECONDS * self.control_freq))),
        )

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)


class HolodeckApproachDeskThenPickCup(
    _HolodeckApproachDeskWorkflowMixin,
    HolodeckPickCupFromDesk,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, then pick up the calibrated glass cup."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenPickCup"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_ROBOT_POSE_KEY = "pick_cup_composite_robot_pose"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            on_enter="_begin_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "pick_cup",
            "_cup_lifted",
            task_cls=HolodeckPickCupFromDesk,
            on_enter="_begin_cup_lift_stage",
            metadata={"skill": "lift", "object": "glass_cup", "min_lift_m": 0.30},
        ),
    )

    def _begin_cup_lift_stage(self):
        self._cup_lift_start_z = self._cup_root_z()

    def _cup_lifted(self):
        return bool(
            self._cup_root_z()
            > self._cup_lift_start_z + HolodeckPickCupFromDesk.MIN_LIFT_DELTA
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then pick up the glass cup",
            target_fixture=self.desk_fixture.stable_id,
            success_definition=(
                "first move at least 0.15 m from the remote spawn and remain near "
                "the calibrated desk, then lift the cup more than 0.30 m for 1 second"
            ),
        )
        return meta


class HolodeckApproachDeskThenMoveCupToCenter(
    _HolodeckApproachDeskWorkflowMixin,
    HolodeckMoveCupToCenter,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, then move the glass cup into its center region."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenMoveCupToCenter"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_SPEC_TASK = "approach_desk_then_move_cup_to_center"
    COMPOSITE_SPEC_SCENES = ("train_1018",)
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            on_enter="_begin_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "move_cup_to_center",
            "_cup_in_target_state",
            task_cls=HolodeckMoveCupToCenter,
            metadata={"skill": "pick_place", "object": "glass_cup", "target": "desk_center"},
        ),
    )

    def _uses_composite_spec(self):
        return self.scene_name in self.COMPOSITE_SPEC_SCENES

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.CALIBRATION_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.CALIBRATION_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        if self.task_spec["desk_body"] != self.navigation_spec["desk_body"]:
            raise RuntimeError(
                f"{self.scene_name}: cup and navigation desks must match"
            )

    def _initial_layout_valid(self):
        """Validate the cup layout without requiring the remote spawn standoff.

        The atomic move-cup task starts 0.45 m from the cup. This composite
        intentionally starts at the desk navigation pose and reaches that
        operating position only after its first stage.
        """
        source = self._initial_cup_base_xy()
        target = self._target_base_xy()
        tolerance = self._target_tolerance()
        outside = np.maximum(np.abs(source - target) - tolerance, 0.0)
        return bool(
            np.linalg.norm(outside) >= self.INITIAL_MIN_TARGET_CLEARANCE
            and self.count_body_collisions() == 0
            and abs(self._cup_bottom_z() - float(self.task_spec["support_z"]))
            <= float(self.task_spec["support_z_tolerance"])
            and self._cup_is_upright()
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then move the glass cup to the center",
            target_fixture=self.desk_fixture.stable_id,
            success_definition=(
                "first reach the calibrated desk in order, then keep the glass cup "
                "inside the calibrated center target region and in contact with the "
                "desk for 1 second"
            ),
        )
        return meta


class HolodeckApproachDeskThenCloseDrawer(
    _HolodeckApproachDeskWorkflowMixin,
    HolodeckCloseDrawer,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, then close the specified dresser drawer."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenCloseDrawer"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1197)
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            on_enter="_begin_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "close_drawer",
            "_drawer_closed",
            task_cls=HolodeckCloseDrawer,
            metadata={"skill": "articulation", "target": "drawer"},
        ),
    )

    def _drawer_closed(self):
        return bool(RoomOpenArticulation._raw_success(self))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then close the specified dresser drawer",
            target_fixture=self.desk_fixture.stable_id,
            target_joint=self.target_joint,
            success_definition=(
                "first reach the calibrated desk in order, then keep the specified "
                "drawer at its calibrated closed fraction for 1 second"
            ),
        )
        return meta


class HolodeckApproachDeskThenOpenJar(
    _HolodeckApproachDeskWorkflowMixin,
    HolodeckOpenMugLid,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, then open the calibrated tabletop jar."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenOpenJar"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_SPEC_TASK = "approach_desk_then_open_jar"
    COMPOSITE_SPEC_SCENES = ("train_1018",)
    COMPOSITE_ROBOT_POSE_KEY = "open_jar_composite_robot_pose"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            on_enter="_begin_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "open_jar",
            "_jar_opened",
            task_cls=HolodeckOpenMugLid,
            metadata={"skill": "pick", "object": "jar_lid", "target": "jar"},
        ),
    )

    def _uses_composite_spec(self):
        return self.scene_name in self.COMPOSITE_SPEC_SCENES

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
            scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
            if scene_bodies:
                removed = self.remove_scene_bodies(scene_bodies)
                missing = set(scene_bodies) - removed
                if missing:
                    raise RuntimeError(
                        f"{self.scene_name}: configured jar scene bodies are missing: "
                        f"{sorted(missing)}"
                    )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        if self.task_spec["table_body"] != self.navigation_spec["desk_body"]:
            raise RuntimeError(
                f"{self.scene_name}: jar and navigation desks must match"
            )
        close_spec = get_holodeck_task_spec(
            self.scene_name,
            "close_mug_lid",
            required=("mug_pose", "closed_lid_pose", "close_distance"),
        )
        self.jar_closed_offset = np.asarray(
            close_spec["closed_lid_pose"]["pos"], dtype=float
        ) - np.asarray(close_spec["mug_pose"]["pos"], dtype=float)
        closed_pose = dict(self.task_spec["lid_pose"])
        closed_pose["pos"] = (
            np.asarray(self.task_spec["mug_pose"]["pos"], dtype=float)
            + self.jar_closed_offset
        ).tolist()
        self.jar_closed_pose = closed_pose
        self.jar_close_distance = float(close_spec["close_distance"])

    def _jar_opened(self):
        """Require the lid to depart from its sealed pose after navigation."""
        mug_pos = self.sim.data.get_body_xpos(self.objects["mug"].root_body)
        lid_pos = self.sim.data.get_body_xpos(self.objects["mug_lid"].root_body)
        closed_offset = self.jar_closed_offset
        return bool(
            np.linalg.norm(np.asarray(lid_pos, dtype=float) - mug_pos - closed_offset)
            > self.jar_close_distance
            and not self._objects_in_contact("mug_lid", "mug")
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._place(self.objects["mug_lid"], self.jar_closed_pose)
        self.sim.forward()
        # The jar mixin places the jar and lid after the composite parent has
        # reset. Re-enter the first stage only after that calibrated layout is
        # present, so reset cannot accidentally validate a stale pose.
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then open the jar",
            target_fixture=self.desk_fixture.stable_id,
            success_definition=(
                "first reach the calibrated desk in order, then keep the jar lid "
                "separated from the jar for 1 second"
            ),
        )
        return meta


__all__ = [
    "HolodeckApproachDeskThenPickCup",
    "HolodeckApproachDeskThenMoveCupToCenter",
    "HolodeckApproachDeskThenCloseDrawer",
    "HolodeckApproachDeskThenOpenJar",
]
