"""Object and desk-centered multi-hop workflows built from Holodeck primitives."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    normalize_holodeck_scene_name,
)
from robocasa.environments.holodeck.tasks.approach_desk import HolodeckApproachDesk
from robocasa.environments.holodeck.tasks.approach_trash_can import (
    HolodeckApproachTrashCan,
)
from robocasa.environments.holodeck.tasks.approach_window import (
    HolodeckApproachWindow,
)
from robocasa.environments.holodeck.tasks.articulation import (
    HolodeckCloseDrawer,
    HolodeckOpenDrawer,
)
from robocasa.environments.holodeck.tasks.take_tissue_box_from_open_drawer import (
    HolodeckTakeTissueBoxFromOpenDrawer,
    _HolodeckDrawerTissueBoxMixin,
)
from robocasa.environments.holodeck.tasks.laptop import (
    HolodeckCloseLaptop,
    HolodeckOpenLaptop,
    _HolodeckLaptopMixin,
)
from robocasa.environments.holodeck.tasks.ice_cube_in_glass import (
    HolodeckIceCubeInGlass,
)
from robocasa.environments.holodeck.tasks.move_cup import (
    HolodeckMoveCupToCenter,
    HolodeckMoveCupToCorner,
)
from robocasa.environments.holodeck.tasks.place_cup_on_placemat import (
    HolodeckPlaceCupOnPlacemat,
)
from robocasa.environments.holodeck.tasks.pick_box_from_floor import (
    HolodeckPickBoxFromFloor,
)
from robocasa.environments.holodeck.tasks.pick_cup_from_desk import (
    HolodeckPickCupFromDesk,
)
from robocasa.environments.holodeck.tasks.place_box_on_floor import (
    HolodeckPlaceBoxOnFloor,
)
from robocasa.environments.holodeck.tasks.place_box_on_desk import (
    HolodeckPlaceBoxOnDesk,
)
from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.push_chair_to_desk import (
    HolodeckPushChairToDesk,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckBoxNavigationMixin:
    """Shared ordered-stage predicates for navigation followed by a box task."""

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _begin_box_navigation_stage(self):
        self._box_navigation_start_xy = self._robot_base_xy().copy()
        self._box_navigation_moved = False

    def _box_navigation_reached(self):
        if (
            np.linalg.norm(self._robot_base_xy() - self._box_navigation_start_xy)
            >= 0.15
        ):
            self._box_navigation_moved = True
        return bool(
            self._box_navigation_moved
            and self._box_navigation_distance() <= self._box_navigation_radius
        )

    def _box_lifted(self):
        return bool(
            float(self.sim.data.get_body_xpos(self.objects["box"].root_body)[2])
            > self._box_initial_z + HolodeckPickBoxFromFloor.MIN_LIFT_DELTA
        )

    def _box_picked(self):
        """Require the box to leave the floor before the placement stage."""
        return bool(
            float(self.sim.data.get_body_xpos(self.objects["box"].root_body)[2])
            > self._box_initial_z + 0.05
        )

    def _box_placed_on_desk(self):
        return bool(self._box_on_desk())

    def _begin_box_lift_stage(self):
        self._box_lift_stage_started = True

    def _reset_box_composite(self):
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()


class HolodeckTakeTissueBoxThenPlaceCup(
    _HolodeckDrawerTissueBoxMixin,
    RoomOpenArticulation,
    HolodeckMultiHopComposite,
):
    """Remove tissue, place it on the desk, then place a cup on a placemat."""

    ABSTRACT = False
    ATOMIC_SKILL = "TakeTissueBoxThenPlaceCup"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Dresser",)
    TARGET_LANG = "dresser drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUCCESS_JOINT_SELECTION = "target"
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "take_tissue_box_from_open_drawer"
    CUP_SPEC_TASK = "place_cup_on_placemat"
    # The Objaverse desks in train_1664/train_2025 have no semantic support
    # region. Reuse the calibrated desk footprint and height from the shared
    # center-cup workflow for the tissue-box placement target.
    DESTINATION_TABLE_SPEC_TASK = HolodeckApproachDesk.SPEC_TASK
    DESTINATION_SUPPORT_SPEC_TASK = "move_cup_to_center"
    # The 1018 tissue-box destination desk has one active baked laptop. Remove
    # it only for this composite task so the shared tissue-box/cup tabletop is
    # available for the requested placements.
    CALIBRATED_LAPTOP_BODIES = {
        "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
    }
    LAPTOP_CLEARED_SCENES = ("train_1664", "train_2025")
    OPEN_STAND_DISTANCE = 0.35
    PLACEMAT_EDGE_TOLERANCE = HolodeckPlaceCupOnPlacemat.PLACEMAT_EDGE_TOLERANCE
    SUPPORT_Z_TOLERANCE = HolodeckPlaceCupOnPlacemat.SUPPORT_Z_TOLERANCE
    SURFACE_Z_TOLERANCE = HolodeckPlaceCupOnPlacemat.SURFACE_Z_TOLERANCE
    MAX_LINEAR_SPEED = HolodeckPlaceCupOnPlacemat.MAX_LINEAR_SPEED
    MAX_ANGULAR_SPEED = HolodeckPlaceCupOnPlacemat.MAX_ANGULAR_SPEED
    COMPOSITE_STAGES = (
        "grasp and remove the tissue box from the initially open drawer",
        "place the tissue box fully on the calibrated desk",
        "place the cup fully on the calibrated placemat",
    )
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "take_tissue_box",
            "_tissue_box_removed_stage",
            task_cls="HolodeckTakeTissueBoxFromOpenDrawer",
            hold_steps=None,
            metadata={"skill": "pick", "object": "tissue_box", "source": "drawer"},
        ),
        HolodeckAtomicTask(
            "place_tissue_box_on_desk",
            "_tissue_box_released_on_desk",
            task_cls=HolodeckTakeTissueBoxFromOpenDrawer,
            hold_steps=5,
            on_enter="_begin_tissue_box_placement",
            metadata={"skill": "place", "object": "tissue_box", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "place_cup_on_placemat",
            "_cup_placed_after_tissue_box",
            task_cls=HolodeckPlaceCupOnPlacemat,
            hold_steps=5,
            on_enter="_begin_cup_stage",
            metadata={"skill": "pick_place", "object": "cup", "target": "placemat"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        laptop_bodies = self.CALIBRATED_LAPTOP_BODIES.get(self.scene_name, ())
        if laptop_bodies:
            removed = self.remove_scene_bodies(laptop_bodies)
            missing = set(laptop_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured task laptop bodies are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _articulation_robot_pose(self):
        spec = self._get_articulation_task_spec()
        composite_pose = spec.get("composite_robot_pose") if spec else None
        if composite_pose is not None:
            return np.asarray(composite_pose["pos"], dtype=float), float(
                composite_pose["yaw"]
            )
        return super()._articulation_robot_pose()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.tissue_task_spec = self.task_spec
        self.cup_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.CUP_SPEC_TASK,
            required=(
                "table_body",
                "cup_scale",
                "cup_pose",
                "placemat_scale",
                "placemat_pose",
                "target_half_size",
                "support_z",
            ),
        )
        # In train_1018 this composite must keep the cup stage at the desk
        # beside the drawer interaction area.  Do not change the standalone
        # place-cup calibration, which intentionally uses the other desk.
        cup_layout = self.tissue_task_spec.get("composite_cup_layout")
        if cup_layout is not None:
            self.cup_task_spec = dict(self.cup_task_spec)
            self.cup_task_spec.update(cup_layout)
        self.cup_desk_fixture = self.get_fixture(id=self.cup_task_spec["table_body"])
        if self.cup_desk_fixture is not None:
            self.fixture_refs["cup_surface"] = self.cup_desk_fixture
            self.fixture_refs["placemat_surface"] = self.cup_desk_fixture
        self.task_spec = self.tissue_task_spec

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.cup_desk_fixture is not None)

    def _setup_scene(self):
        super()._setup_scene()
        if self.cup_desk_fixture is None:
            raise RuntimeError(f"{self.scene_name}: cup desk is missing")

    def _create_objects(self):
        super()._create_objects()
        root = robocasa.models.assets_root
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(root, HolodeckPlaceCupOnPlacemat.CUP_ASSET),
            scale=float(self.cup_task_spec["cup_scale"]),
        )
        placemat = MJCFObject(
            name="placemat",
            mjcf_path=os.path.join(root, HolodeckPlaceCupOnPlacemat.PLACEMAT_ASSET),
            scale=float(self.cup_task_spec["placemat_scale"]),
        )
        self.objects[cup.name] = cup
        self.objects[placemat.name] = placemat
        self.model.merge_objects([cup, placemat])

    def _place_object(self, name, pose):
        obj = self.objects[name]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))

    def _reset_internal(self):
        self._cup_layout_ready = False
        super()._reset_internal()
        self.task_spec = self.cup_task_spec
        self._place_object("placemat", self.cup_task_spec["placemat_pose"])
        self._place_object("cup", self.cup_task_spec["cup_pose"])
        self.sim.forward()
        self._initial_placemat_pos = self.object_pos("placemat").copy()
        self._initial_cup_pos = self.object_pos("cup").copy()
        if not self._cup_call("_initial_layout_valid"):
            raise RuntimeError(f"{self.scene_name}: cup placemat reset is invalid")
        self.task_spec = self.tissue_task_spec
        self._cup_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_cup_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _tissue_box_removed_stage(self):
        """Expose the drawer primitive's post-action removal state to the FSM."""
        return bool(
            getattr(self, "_tissue_box_removed_after_open_state", False)
            and self._drawer_open_enough(self.target_semantic_joint.fraction(self.sim))
        )

    def _begin_tissue_box_placement(self):
        self._tissue_box_removed_before_placement = bool(
            getattr(self, "_tissue_box_removed_after_open_state", False)
        )

    def _tissue_box_released_on_desk(self):
        return bool(
            getattr(self, "_tissue_box_removed_before_placement", False)
            and self.object_on_region(
                "tissue_box",
                self.destination_region,
                z_tol=0.05,
                require_contact=True,
            )
        )

    def _tissue_box_still_placed_on_desk(self):
        return bool(
            self.object_on_region(
                "tissue_box",
                self.destination_region,
                z_tol=0.05,
                require_contact=True,
            )
        )

    def _cup_call(self, method):
        saved_spec = self.task_spec
        self.task_spec = self.cup_task_spec
        try:
            return getattr(HolodeckPlaceCupOnPlacemat, method)(self)
        finally:
            self.task_spec = saved_spec

    # The placemat primitive calls these helpers through ``self``. Keep the
    # composite independent of its setup lifecycle while preserving those
    # internal calls.
    def _cup_on_placemat(self):
        return HolodeckPlaceCupOnPlacemat._cup_on_placemat(self)

    def _object_is_supported_by_desk(self, name):
        return HolodeckPlaceCupOnPlacemat._object_is_supported_by_desk(self, name)

    def _object_top_z(self, name):
        return HolodeckPlaceCupOnPlacemat._object_top_z(self, name)

    def _cup_is_stable(self):
        return HolodeckPlaceCupOnPlacemat._cup_is_stable(self)

    def _placemat_is_stable(self):
        return HolodeckPlaceCupOnPlacemat._placemat_is_stable(self)

    def _cup_was_manipulated(self):
        return HolodeckPlaceCupOnPlacemat._cup_was_manipulated(self)

    def _begin_cup_stage(self):
        pass

    def _cup_placed_after_tissue_box(self):
        return bool(
            self._cup_call("_cup_on_placemat")
            and self._tissue_box_still_placed_on_desk()
        )

    def _post_action(self, action):
        return super()._post_action(action)

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the tissue box out of the open drawer, place it on the desk, "
                "then move the cup to the placemat"
            ),
            source_object="tissue_box",
            source_fixture=self.target_fixture.stable_id,
            source_joint=self.target_joint,
            target_object="cup",
            target_fixture=self.cup_desk_fixture.stable_id,
            success_definition=(
                "first lift the tissue box after the drawer is open more "
                f"than {self.MIN_LIFT_DELTA:.2f} m above its reset height, and "
                "place it while fully supported by and in contact with the "
                "calibrated desk, then place the cup fully on the "
                "calibrated placemat"
            ),
        )
        return meta


class HolodeckApproachDeskThenOpenLaptop(HolodeckOpenLaptop, HolodeckMultiHopComposite):
    """Walk to the calibrated desk, then open its existing laptop.

    The two stages share one scene and the same baked laptop; the navigation
    condition therefore cannot be bypassed by opening the laptop first.
    """

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachThenOpenLaptop"
    TASK_FAMILY = "composite"
    SPEC_TASK = "open_laptop"
    NAVIGATION_SPEC_TASK = "approach_desk"
    # Keep this composite's scene scope explicit; it is expanded only together
    # with the corresponding scene-specific laptop and desk calibrations.
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    MIN_APPROACH_DISPLACEMENT = 0.15
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation"},
        ),
        HolodeckAtomicTask(
            "open_laptop",
            "_raw_success",
            task_cls=HolodeckOpenLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        # Objaverse desk metadata is not consistently classified as ``desk``
        # in train_1664/train_2025.  The scene calibration already names the
        # exact collision body, so resolve that body directly like the atomic
        # approach task does.
        self.desk_fixture = self.get_fixture(id=self.navigation_spec["desk_body"])
        if self.desk_fixture is None:
            return
        self.fixture_refs["approach_desk"] = self.desk_fixture
        pose = self._approach_desk_spawn_pose()
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _approach_desk_spawn_pose(self):
        """Use the task-local remote spawn when the scene provides one."""
        return self.navigation_spec.get(
            "desk_composite_robot_pose", self.navigation_spec["robot_pose"]
        )

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.desk_fixture is not None)

    def _place_robot_for_articulation(self, distance):
        """Keep the far navigation spawn instead of laptop's close stand pose."""
        del distance
        self.sim.forward()
        return self.count_body_collisions() == 0

    def _setup_scene(self):
        super()._setup_scene()
        distance = self.robot_fixture_collision_distance(
            self.desk_fixture, max_distance=self.success_radius + 1.0
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: composite navigation spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        # Suppress the generic stage validator until the reset has restored the
        # robot pose; otherwise it can compare against the previous episode.
        self._approach_desk_resetting = True
        super()._reset_internal()
        pose = self._approach_desk_spawn_pose()
        if not (
            self._set_robot_free_pose(
                np.asarray(pose["pos"], dtype=float)[:2], float(pose["yaw"])
            )
            and self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(
                f"{self.scene_name}: failed to restore calibrated approach-desk spawn"
            )
        self.robot_base_pos = np.asarray(pose["pos"], dtype=float).copy()
        self.robot_base_ori = np.array([0.0, 0.0, float(pose["yaw"])])
        self._approach_desk_start_xy = self._robot_base_xy().copy()
        self._approach_desk_resetting = False

    def _desk_reached(self):
        if getattr(self, "_approach_desk_resetting", False):
            return False
        start_xy = getattr(self, "_approach_desk_start_xy", None)
        if start_xy is None:
            return False
        if np.linalg.norm(self._robot_base_xy() - start_xy) < float(
            self.MIN_APPROACH_DISPLACEMENT
        ):
            return False
        return bool(
            self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.success_radius + 0.25,
            )
            <= self.success_radius
        )

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        """Require both navigation and laptop stages to be complete."""
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then open the laptop",
            target_fixture=self.desk_fixture.stable_id,
            target_joint=self.target_joint,
            success_definition=(
                "first move at least "
                f"{self.MIN_APPROACH_DISPLACEMENT:.2f} m from the calibrated spawn "
                "and stay within the desk collision distance, then open the "
                "specified laptop lid for its calibrated hold interval"
            ),
        )
        return meta


class HolodeckOpenDrawerThenTakeTissueBox(
    _HolodeckDrawerTissueBoxMixin,
    RoomOpenArticulation,
    HolodeckMultiHopComposite,
):
    """Open the calibrated drawer, then remove its tissue box.

    This retains the existing tissue-box asset and drawer receptacle calibration,
    and starts at the scene-specific calibrated opening fraction.
    """

    ABSTRACT = False
    ATOMIC_SKILL = "OpenDrawerThenTakeTissueBox"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Dresser",)
    TARGET_LANG = "dresser drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUCCESS_JOINT_SELECTION = "target"
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "open_drawer_then_take_tissue_box"
    OPEN_STAND_DISTANCE = 0.35
    # This workflow ends after removing the tissue box.  It does not use the
    # destination desk, so a desk support region is not part of its reset
    # contract.  Placement workflows below opt back into the strict check.
    DESTINATION_REGION_REQUIRED = False
    START_DRAWER_CLOSED = True
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "open_drawer",
            "_raw_success",
            task_cls=HolodeckOpenDrawer,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "drawer"},
        ),
        HolodeckAtomicTask(
            "take_tissue_box",
            "_tissue_box_removed_after_open",
            task_cls=HolodeckTakeTissueBoxFromOpenDrawer,
            hold_steps=None,
            on_enter="_begin_tissue_box_removal_stage",
            metadata={"skill": "pick", "object": "tissue_box"},
        ),
    )
    COMPOSITE_STAGES = (
        "open the calibrated drawer from its closed state",
        "lift and remove the tissue box from that open drawer",
    )

    def _articulation_robot_pose(self):
        pose = super()._articulation_robot_pose()
        if self.__class__ is not HolodeckOpenDrawerThenTakeTissueBox:
            return pose
        spec = self._get_articulation_task_spec()
        composite_pose = spec.get("composite_robot_pose") if spec else None
        if composite_pose is None:
            return pose
        return np.asarray(composite_pose["pos"], dtype=float), float(
            composite_pose["yaw"]
        )

    def _set_tissue_box_in_drawer(self):
        if self.__class__ is not HolodeckOpenDrawerThenTakeTissueBox:
            return super()._set_tissue_box_in_drawer()
        spec = self._get_articulation_task_spec()
        offset = spec.get("initial_tissue_box_offset") if spec else None
        if offset is None:
            return super()._set_tissue_box_in_drawer()
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: "
                "initial_tissue_box_offset must contain 3 values"
            )
        center, _ = self.drawer_region.world_pose(self.sim)
        center = np.asarray(center, dtype=float).copy()
        center += offset
        joint = self.objects["tissue_box"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        for z in np.arange(float(center[2]) + 0.08, float(center[2]) - 0.12, -0.002):
            qpos[:3] = [float(center[0]), float(center[1]), float(z)]
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
            self.sim.forward()
            if self._tissue_box_in_drawer():
                return
        raise RuntimeError(
            f"{self.scene_name}: no stable tissue-box pose inside drawer"
        )

    def _reset_internal(self):
        self._drawer_layout_ready = False
        super()._reset_internal()
        self._tissue_box_removed = False
        self._invalid_pre_open_lift = False
        self._drawer_layout_ready = True
        self._validate_initial_composite_state()
        if self._raw_success():
            raise RuntimeError(f"{self.scene_name}: drawer composite starts open")

    def _validate_initial_composite_state(self):
        if not getattr(self, "_drawer_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _begin_tissue_box_removal_stage(self):
        # Preserve ordering without requiring a grasp event: a lift observed
        # before the drawer stage completed invalidates the removal stage.
        self._invalid_pre_open_lift = bool(
            getattr(self, "_tissue_box_removed_after_open_state", False)
        )
        self._tissue_box_removed = False

    def _tissue_box_removed_after_open(self):
        # The ordered state machine already proves that the drawer stage was
        # completed. Match the atomic tissue-box predicate: only the box's
        # post-reset lift state determines this stage, with the composite
        # hold counter supplying the one-second interval.
        if getattr(self, "_invalid_pre_open_lift", False):
            return False
        removed = bool(getattr(self, "_tissue_box_removed_after_open_state", False))
        if removed:
            self._tissue_box_removed = True
        return removed

    def _check_success(self):
        """Require the declared drawer and tissue-box stages in order."""
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="open the dresser drawer, then take out the tissue box",
            source_fixture=self.target_fixture.stable_id,
            source_joint=self.target_joint,
            source_object="tissue_box",
            source_model=self.TISSUE_BOX_ASSET,
            lift_delta=float(self.MIN_LIFT_DELTA),
            success_definition=(
                "open the calibrated drawer first, then lift the tissue "
                f"box more than {self.MIN_LIFT_DELTA:.2f} m above its reset height"
            ),
        )
        return meta


class HolodeckOpenDrawerTakeTissueBoxToDesk(HolodeckOpenDrawerThenTakeTissueBox):
    """Open the drawer, remove its tissue box, and place it on the desk."""

    ABSTRACT = False
    ATOMIC_SKILL = "OpenDrawerTakeTissueBoxToDesk"
    DIFFICULTY_LEVEL = "L3"
    # This placement workflow has dedicated calibration only for the two
    # scenes where its drawer, tissue box, and desk layout are validated.
    SUPPORTED_SCENE_IDS = (117, 1197)
    SPEC_TASK = "open_drawer_take_tissue_box_to_desk"
    DESTINATION_REGION_REQUIRED = True
    DESTINATION_TABLE_SPEC_TASK = HolodeckApproachDesk.SPEC_TASK
    DESTINATION_SUPPORT_SPEC_TASK = "move_cup_to_center"
    # The default Holodeck layout adds a laptop to this desk for desk/laptop
    # tasks.  This workflow has no laptop stage; leave the calibrated desk
    # surface clear so the tissue box can actually be placed on it.
    LAPTOP_CLEARED_SCENES = ("train_1664", "train_2025")
    # Scene 1197's desk calibration intentionally accepts contact anywhere on
    # the desktop. Do not restrict placement to the narrow support patch used
    # by the center-cup workflow.
    FULL_DESK_CONTACT_ONLY_SCENES = ("train_1197",)
    # ``None`` delegates to the tissue-box primitive's HOLD_SECONDS and lets
    # the composite convert the one-second requirement using control_freq.
    PLACEMENT_HOLD_STEPS = None
    SUCCESS_DEBUG_ENABLED = True
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "open_drawer",
            "_raw_success",
            task_cls=HolodeckOpenDrawer,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "drawer"},
        ),
        HolodeckAtomicTask(
            "take_tissue_box",
            "_tissue_box_removed_after_open",
            task_cls=HolodeckTakeTissueBoxFromOpenDrawer,
            hold_steps=None,
            on_enter="_begin_tissue_box_removal_stage",
            metadata={"skill": "pick", "object": "tissue_box"},
        ),
        HolodeckAtomicTask(
            "place_tissue_box_on_desk",
            "_tissue_box_released_on_desk",
            task_cls=HolodeckTakeTissueBoxFromOpenDrawer,
            hold_steps=PLACEMENT_HOLD_STEPS,
            on_enter="_begin_tissue_box_placement",
            metadata={"skill": "place", "object": "tissue_box"},
        ),
    )

    def _required_stage_hold_steps(self, task):
        # Keep the scene117 evaluation hold isolated from the close-drawer task.
        if (
            type(self) is HolodeckOpenDrawerTakeTissueBoxToDesk
            and self.scene_name == "train_117"
            and task.name == "place_tissue_box_on_desk"
        ):
            return 5
        return super()._required_stage_hold_steps(task)

    def _begin_tissue_box_placement(self):
        self._tissue_box_removed_before_placement = bool(self._tissue_box_removed)

    def _tissue_box_on_destination_desk(self):
        """Check contact with the calibrated desk surface for this scene."""
        return self.object_on_region(
            "tissue_box",
            self.destination_region,
            z_tol=0.05,
            require_contact=True,
            require_bounds=(self.scene_name not in self.FULL_DESK_CONTACT_ONLY_SCENES),
        )

    def _tissue_box_released_on_desk(self):
        return bool(
            getattr(self, "_tissue_box_removed_before_placement", False)
            and self._tissue_box_removed
            and self._tissue_box_on_destination_desk()
        )

    def _tissue_box_still_placed_on_desk(self):
        """Check that the tissue box remains on and in contact with the desk."""
        return bool(self._tissue_box_on_destination_desk())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        placement_hold = (
            "5 control steps"
            if type(self) is HolodeckOpenDrawerTakeTissueBoxToDesk
            and self.scene_name == "train_117"
            else f"{self.HOLD_SECONDS:.1f} s"
        )
        meta.update(
            lang=(
                "open the dresser drawer, take out the tissue box, then place "
                "it on the desk"
            ),
            success_definition=(
                "open the calibrated drawer, lift its tissue box more than "
                f"{self.MIN_LIFT_DELTA:.2f} m above its reset height "
                "after opening, then place the box while it is fully supported "
                "by and in contact "
                f"with the named desk for {placement_hold}"
            ),
        )
        return meta


class HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer(
    HolodeckOpenDrawerTakeTissueBoxToDesk
):
    """Complete the tissue-box placement workflow by closing the drawer."""

    ABSTRACT = False
    ATOMIC_SKILL = "OpenDrawerTakeTissueBoxToDeskThenCloseDrawer"
    DIFFICULTY_LEVEL = "L4"
    CLOSE_DRAWER_SPEC_TASK = "close_drawer"
    SUCCESS_DEBUG_ENABLED = True
    # Keep this separate workflow's existing bounded placement predicate.
    FULL_DESK_CONTACT_ONLY_SCENES = ()
    ATOMIC_TASKS = (
        *HolodeckOpenDrawerTakeTissueBoxToDesk.ATOMIC_TASKS,
        HolodeckAtomicTask(
            "close_drawer",
            "_drawer_closed_after_placement",
            task_cls=HolodeckCloseDrawer,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "drawer"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.close_drawer_spec = get_holodeck_task_spec(
            self.scene_name,
            self.CLOSE_DRAWER_SPEC_TASK,
            required=("success_close_fraction",),
        )

    def _drawer_closed_after_placement(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            <= float(self.close_drawer_spec["success_close_fraction"])
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "open the dresser drawer, take out the tissue box, place it on "
                "the desk, then close the drawer"
            ),
            success_definition=(
                "open the calibrated drawer, lift its tissue box more than "
                f"{self.MIN_LIFT_DELTA:.2f} m above its reset height, "
                "then place it on "
                "the named desk, then close that same drawer while the box remains "
                "supported on the desk"
            ),
        )
        return meta


class HolodeckApproachDeskThenCloseLaptopMoveCup(
    HolodeckCloseLaptop,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, close its laptop, then move the cup to a corner."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenCloseLaptopMoveCup"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "close_laptop"
    COMPOSITE_SPEC_TASK = "approach_desk_then_close_laptop_move_cup"
    COMPOSITE_SCENE_IDS = (1197, 1664, 2025)
    NAVIGATION_SPEC_TASK = "approach_desk"
    # This workflow stays on the original cup desk in train_1018.  The
    # shared approach_desk spec serves the laptop and chair workflows, which
    # intentionally target the other desk.
    NAVIGATION_RELOCATIONS = {
        "train_1018": {
            "desk_body": "desk_699341cef0b075b18b23569c42fb1e52_1_0_0",
            "robot_pose": {
                "pos": [5.35, 1.25, 0.0],
                "yaw": 3.141592653589793,
            },
        },
    }
    LAPTOP_BODY = "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    LAPTOP_RELOCATIONS = {
        # Keep the laptop centered on the calibrated desk.  The root pose is
        # expressed in the scene world frame and retains each asset's baked
        # orientation / support height.
        "train_117": {
            "pos": [5.670000, 4.800000, 0.774781],
            "quat": [0.5, 0.5, -0.5, -0.5],
        },
        "train_1018": {
            "pos": [3.250000, 1.250000, 0.885105],
            "quat": [0.5, 0.5, 0.5, 0.5],
        },
    }
    # Keep the composite's cup layout identical to the corner atomic task.
    # Navigation still uses its own remote spawn pose below.
    CUP_SPEC_TASK = HolodeckMoveCupToCorner.SPEC_TASK
    COMPOSITE_CUP_SPEC_TASK = "approach_desk_then_close_laptop_move_cup"
    # Reuse the corner-placement geometry constants from the cup primitive;
    # this composite calls its calibration helpers without inheriting it.
    CORNER_TARGET_TOLERANCE = HolodeckMoveCupToCorner.CORNER_TARGET_TOLERANCE
    INITIAL_CUP_TOWARD_ROBOT = HolodeckMoveCupToCorner.INITIAL_CUP_TOWARD_ROBOT
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "close_laptop",
            "_raw_success",
            task_cls=HolodeckCloseLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
        HolodeckAtomicTask(
            "move_cup_to_corner",
            "_cup_in_target_state",
            task_cls=HolodeckMoveCupToCorner,
            hold_steps=5,
            on_enter="_begin_cup_stage",
            metadata={"skill": "pick_place", "object": "glass_cup"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        composite_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
        )
        replaced_body = HolodeckMoveCupToCorner.CALIBRATED_REPLACED_BODIES.get(
            self.scene_name
        )
        if replaced_body is not None and replaced_body != composite_spec.get(
            "source_body"
        ):
            removed = self.remove_scene_bodies((replaced_body,))
        else:
            removed = set()
        if (
            replaced_body is not None
            and replaced_body != composite_spec.get("source_body")
            and replaced_body not in removed
        ):
            raise RuntimeError(
                f"{self.scene_name}: configured source body is missing: "
                f"{replaced_body}"
            )
        # ``HolodeckCloseLaptop`` removes the selected spec's
        # ``remove_scene_bodies`` and ``source_body`` below.  Do not remove
        # the composite cleanup list here as well: doing so makes the laptop
        # mixin see an already-removed body and report it as missing.
        relocation = self.LAPTOP_RELOCATIONS.get(self.scene_name)
        if relocation is not None:
            laptop_body = self.find_scene_body(self.LAPTOP_BODY)
            if laptop_body is None:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{self.LAPTOP_BODY}"
                )
            laptop_body.set(
                "pos", " ".join(f"{value:.6f}" for value in relocation["pos"])
            )
            laptop_body.set(
                "quat", " ".join(f"{value:.6f}" for value in relocation["quat"])
            )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.task_spec
        navigation_task = (
            self.COMPOSITE_SPEC_TASK
            if self.scene_name
            in {f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS}
            else self.NAVIGATION_SPEC_TASK
        )
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            navigation_task,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        navigation_override = self.NAVIGATION_RELOCATIONS.get(self.scene_name)
        if navigation_override is not None:
            self.navigation_spec = dict(self.navigation_spec)
            self.navigation_spec.update(navigation_override)
        cup_task = (
            self.COMPOSITE_CUP_SPEC_TASK
            if self.scene_name
            in {f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS}
            else self.CUP_SPEC_TASK
        )
        self.cup_task_spec = get_holodeck_task_spec(
            self.scene_name,
            cup_task,
            required=(
                "desk_body",
                "cup_scale",
                "cup_pose",
                "initial_cup_base_xy",
                "robot_pose",
                "target_base_xy",
                "target_tolerance",
                "support_z",
                "support_z_tolerance",
                "upright_local_axis",
                "min_upright_cosine",
                "max_linear_speed",
                "max_angular_speed",
                "tabletop_bounds_xy",
            ),
        )
        # Objaverse exports do not always classify the calibrated desk as a
        # ``desk``. Resolve the exact stable body first, matching the atomic
        # cup task's semantic/raw/category fallback order.
        desk_body = self.navigation_spec["desk_body"]
        semantic_fixtures = getattr(self.semantic_scene, "fixtures", {}).values()
        self.desk_fixture = next(
            (
                fixture
                for fixture in semantic_fixtures
                if fixture.body_name == desk_body
            ),
            None,
        )
        if self.desk_fixture is None:
            self.desk_fixture = next(
                (
                    fixture
                    for fixture in getattr(self, "fixtures", {}).values()
                    if fixture.body_name == desk_body
                ),
                None,
            )
        if self.desk_fixture is None:
            desks = self.get_fixture(category="desk", return_all=True) or ()
            self.desk_fixture = next(
                (fixture for fixture in desks if fixture.body_name == desk_body),
                None,
            )
        if self.desk_fixture is None:
            self.desk_fixture = next(
                (
                    fixture
                    for fixture in self.get_fixture(return_all=True) or ()
                    if fixture.body_name == self.navigation_spec["desk_body"]
                ),
                None,
            )
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
            self.fixture_refs["cup_surface"] = self.desk_fixture
            pose = self.navigation_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

        saved_spec = self.task_spec
        self.task_spec = self.cup_task_spec
        try:
            self._corner_target_base_xys = (
                HolodeckMoveCupToCorner._get_corner_target_base_xys(self)
            )
        finally:
            self.task_spec = saved_spec

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(
                robocasa.models.assets_root, HolodeckMoveCupToCorner.CUP_ASSET
            ),
            scale=float(self.cup_task_spec["cup_scale"]),
        )
        self.objects[cup.name] = cup
        self.model.merge_objects([cup])

    def _setup_scene(self):
        super()._setup_scene()
        self._set_initial_cup_state()
        self.sim.forward()
        if self._laptop_cup_colliding():
            raise RuntimeError(
                f"{self.scene_name}: laptop and glass cup collide at reset"
            )
        self._success_hold_steps = 0

        distance = self.robot_fixture_collision_distance(
            self.desk_fixture, max_distance=self.success_radius + 1.0
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: composite navigation spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        super()._reset_internal()
        self._set_initial_cup_state()
        self.sim.forward()
        self._success_hold_steps = 0

    def _set_initial_cup_state(self):
        """Restore the corner atomic task's calibrated cup pose."""
        pose = dict(self.cup_task_spec["cup_pose"])
        pose["pos"] = [
            *self.cup_task_spec["initial_cup_base_xy"],
            float(self.cup_task_spec["cup_pose"]["pos"][2]),
        ]
        joint = self.objects["cup"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))

    def _cup_call(self, method, *args):
        saved_spec = self.task_spec
        self.task_spec = self.cup_task_spec
        try:
            return getattr(HolodeckMoveCupToCorner, method)(self, *args)
        finally:
            self.task_spec = saved_spec

    # The cup primitive methods are delegated onto this composite instance.
    # These bridges preserve the primitive's internal self-calls without
    # adding its setup lifecycle to the composite MRO.
    def _cup_body_ids(self):
        return HolodeckMoveCupToCorner._cup_body_ids(self)

    def _desk_body_ids(self):
        return HolodeckMoveCupToCorner._desk_body_ids(self)

    def _cup_base_center(self):
        return HolodeckMoveCupToCorner._cup_base_center(self)

    def _cup_bottom_z(self):
        return HolodeckMoveCupToCorner._cup_bottom_z(self)

    def _cup_is_upright(self):
        return HolodeckMoveCupToCorner._cup_is_upright(self)

    def _cup_contacts_desk(self):
        return HolodeckMoveCupToCorner._cup_contacts_desk(self)

    def _cup_released(self):
        return HolodeckMoveCupToCorner._cup_released(self)

    def _cup_in_target_region(self, base_center):
        return HolodeckMoveCupToCorner._cup_in_target_region(self, base_center)

    def _target_base_xy(self):
        return HolodeckMoveCupToCorner._target_base_xy(self)

    def _target_tolerance(self):
        return HolodeckMoveCupToCorner._target_tolerance(self)

    def _desk_target_frame(self):
        """Reuse the corner cup primitive's desk-local target basis."""
        return HolodeckMoveCupToCorner._desk_target_frame(self)

    def _corner_target_frame(self):
        return HolodeckMoveCupToCorner._corner_target_frame(self)

    def _place_robot_for_articulation(self, distance):
        """Keep the far navigation spawn instead of the close laptop pose."""
        del distance
        self.sim.forward()
        return self.count_body_collisions() == 0

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.success_radius + 0.25,
            )
            <= self.success_radius
        )

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _laptop_cup_colliding(self):
        laptop_root = self.sim.model.body_name2id(self.laptop_task_spec["target_body"])
        cup_root = self.obj_body_id["cup"]
        laptop_bodies = self._body_tree_ids([laptop_root])
        cup_bodies = self._body_tree_ids([cup_root])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in laptop_bodies and body2 in cup_bodies) or (
                body2 in laptop_bodies and body1 in cup_bodies
            ):
                return True
        return False

    def _begin_cup_stage(self):
        pass

    def _cup_in_target_state(self):
        return bool(self._cup_call("_cup_in_target_state"))

    def _post_action(self, action):
        return super()._post_action(action)

    def _check_success(self):
        """Use the composite state machine, not the inherited laptop predicate."""
        return HolodeckMultiHopComposite._check_success(self)

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.desk_fixture is not None)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, close the laptop, then move the glass cup to a corner",
            target_fixture=self.desk_fixture.stable_id,
            target_joint=self.target_joint,
            success_definition=(
                "first stay within the calibrated desk collision distance for "
                f"{HolodeckApproachDesk.HOLD_STEPS} control steps, then close the "
                "calibrated laptop and place the calibrated glass cup in a "
                "robot-side desk-corner target in contact with the desk"
            ),
        )
        return meta


# Container workflows


import numpy as np

from robocasa.environments.holodeck.tasks.mug_lid import (
    HolodeckCloseMugLid,
    HolodeckOpenMugLid,
)
from robocasa.environments.holodeck.tasks.sugar_cube_in_jar import (
    HolodeckSugarCubeInJar,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckSugarJarCompositeMixin:
    """Reuse one jar, removable lid, and sugar cube across serial stages."""

    CLOSE_LID_SPEC_TASK = "close_mug_lid"
    START_LID_CLOSED = False

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.close_lid_spec = get_holodeck_task_spec(
            self.scene_name,
            self.CLOSE_LID_SPEC_TASK,
            required=("mug_pose", "closed_lid_pose", "close_distance"),
        )

    def _validate_initial_composite_state(self):
        # The sugar task places jar/lid/sugar after the composite reset hook;
        # ignore the pre-placement MJCF default poses and validate again below.
        if not getattr(self, "_jar_composite_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _reset_internal(self):
        self._jar_composite_layout_ready = False
        super()._reset_internal()
        # The sugar-jar and standalone close-lid calibrations can use
        # different absolute jar positions. Anchor the close target to the
        # jar pose actually installed by HolodeckSugarCubeInJar.
        self._jar_reference_pos = self.object_pos("jar").copy()
        if self.START_LID_CLOSED:
            self._place(self.objects["jar_lid"], self._closed_lid_pose_for_jar())
            self.sim.forward()

        # HolodeckSugarCubeInJar owns the shared asset layout. Reset the
        # composite state only after that layout (and, for the three-hop task,
        # its closed-lid override) is fully in place.
        self._jar_composite_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()
        if self.START_LID_CLOSED and self._lid_is_open():
            raise RuntimeError(f"{self.scene_name}: jar composite starts open")
        if not self.START_LID_CLOSED and not self._lid_is_open():
            raise RuntimeError(
                f"{self.scene_name}: add-sugar composite needs an open jar"
            )

    def _begin_sugar_stage(self):
        """Mark the start of the target-state placement stage."""
        return None

    def _begin_close_lid_stage(self):
        """Require an open lid when the final sealing stage begins."""
        self._lid_open_when_closing = self._lid_is_open()

    def _sugar_inserted(self):
        """Target-state predicate for the sugar cube placement."""
        return bool(self._sugar_in_jar())

    def _lid_is_closed(self):
        lid_pos = self.object_pos("jar_lid")
        closed_pos = np.asarray(self._closed_lid_pose_for_jar()["pos"], dtype=float)
        return bool(
            np.linalg.norm(lid_pos - closed_pos)
            <= float(self.close_lid_spec["close_distance"])
        )

    def _closed_lid_pose_for_jar(self):
        """Translate the standalone lid calibration onto this composite jar."""
        jar_reference = getattr(self, "_jar_reference_pos", None)
        if jar_reference is None:
            jar_reference = self.object_pos("jar")
        jar_pos = np.asarray(jar_reference, dtype=float)
        close_mug_pos = np.asarray(self.close_lid_spec["mug_pose"]["pos"], dtype=float)
        standalone_closed_pos = np.asarray(
            self.close_lid_spec["closed_lid_pose"]["pos"], dtype=float
        )
        pose = dict(self.close_lid_spec["closed_lid_pose"])
        pose["pos"] = (jar_pos + standalone_closed_pos - close_mug_pos).tolist()
        return pose

    def _lid_is_open(self):
        return not self._objects_in_contact("jar_lid", "jar")

    def _objects_in_contact(self, first_name, second_name):
        first_bodies = self._body_tree_ids([self.obj_body_id[first_name]])
        second_bodies = self._body_tree_ids([self.obj_body_id[second_name]])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            bodies = {
                int(self.sim.model.geom_bodyid[contact.geom1]),
                int(self.sim.model.geom_bodyid[contact.geom2]),
            }
            if (bodies & first_bodies) and (bodies & second_bodies):
                return True
        return False

    def _lid_closed_after_sugar(self):
        return bool(
            getattr(self, "_lid_open_when_closing", False) and self._lid_is_closed()
        )

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._sugar_inserted()
            and self._lid_closed_after_sugar()
        )


class HolodeckSugarCubeThenCloseJar(
    _HolodeckSugarJarCompositeMixin,
    HolodeckSugarCubeInJar,
    HolodeckMultiHopComposite,
):
    """Put sugar into an already open jar, then seal that same jar."""

    ABSTRACT = False
    ATOMIC_SKILL = "AddSugarThenCloseJar"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    # This composite has its own object layout; keep the atomic sugar task's
    # calibration unchanged.
    SPEC_TASK = "sugar_cube_then_close_jar"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SUCCESS_DEBUG_INTERVAL_SECONDS = 5.0
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "add_sugar",
            "_sugar_inserted",
            task_cls=HolodeckSugarCubeInJar,
            hold_steps=None,
            on_enter="_begin_sugar_stage",
            metadata={"skill": "pick_place", "object": "sugar_cube"},
        ),
        HolodeckAtomicTask(
            "close_jar",
            "_lid_closed_after_sugar",
            task_cls=HolodeckCloseMugLid,
            hold_steps=None,
            on_enter="_begin_close_lid_stage",
            metadata={"skill": "articulation", "target": "jar_lid"},
        ),
    )

    def _reset_internal(self):
        super()._reset_internal()
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._log_success_conditions()
        return ret

    def _log_success_conditions(self):
        """Print the composite success predicates every five simulated seconds."""
        sim_time = float(self.sim.data.time)
        interval = float(self.SUCCESS_DEBUG_INTERVAL_SECONDS)
        next_log_time = getattr(self, "_success_debug_next_log_time", interval)
        if sim_time < next_log_time:
            return
        while next_log_time <= sim_time:
            next_log_time += interval
        self._success_debug_next_log_time = next_log_time

        def status(condition):
            return "OK" if condition else "MISSING"

        sugar_in_jar = bool(self._sugar_in_jar())
        sugar_local, jar_local = self._sugar_and_jar_local_bbox_points()
        sugar_local_min = np.min(sugar_local, axis=0)
        sugar_local_max = np.max(sugar_local, axis=0)
        jar_local_min = np.min(jar_local, axis=0) - self.JAR_BBOX_TOLERANCE
        jar_local_max = np.max(jar_local, axis=0) + self.JAR_BBOX_TOLERANCE
        add_sugar_success = bool(sugar_in_jar)

        lid_open_when_closing = bool(getattr(self, "_lid_open_when_closing", False))
        lid_closed = bool(self._lid_is_closed())
        close_jar_success = bool(lid_open_when_closing and lid_closed)
        stage = self.current_atomic_task
        close_stage_active = self.stage_index >= 1
        stage_name = stage.name if stage is not None else "complete"
        stage_hold = self._stage_hold_steps
        stage_target = self._required_stage_hold_steps(stage) if stage else 0
        task_success = bool(stage is None and add_sugar_success and close_jar_success)

        print(
            f"[{type(self).__name__}][{self.scene_name}] "
            f"t={sim_time:.2f}s stage={stage_name} "
            f"hold={stage_hold}/{stage_target}; "
            f"in_jar={status(sugar_in_jar)}, "
            f"success={status(add_sugar_success)}); "
            f"measurements(sugar_bbox_min={np.array2string(sugar_local_min, precision=4)}, "
            f"sugar_bbox_max={np.array2string(sugar_local_max, precision=4)}, "
            f"jar_bbox_min={np.array2string(jar_local_min, precision=4)}, "
            f"jar_bbox_max={np.array2string(jar_local_max, precision=4)}, "
            f"close_jar(open_when_closing="
            f"{status(lid_open_when_closing) if close_stage_active else 'N/A'}, "
            f"closed={status(lid_closed) if close_stage_active else 'N/A'}, "
            f"success={status(close_jar_success) if close_stage_active else 'N/A'}); "
            f"task={status(task_success)}",
            flush=True,
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="put the sugar cube into the open jar, then put the lid back on",
            source_object="sugar_cube",
            target_object="jar",
            target_lid="jar_lid",
            success_definition=(
                "first keep the sugar cube fully inside the jar's outer octagonal "
                "boundary, then keep the lid within its calibrated closed-pose "
                "tolerance"
            ),
        )
        return meta


class HolodeckOpenJarAddSugarThenCloseJar(
    _HolodeckSugarJarCompositeMixin,
    HolodeckSugarCubeInJar,
    HolodeckMultiHopComposite,
):
    """Open a sealed jar, add sugar, and seal the same jar again."""

    ABSTRACT = False
    ATOMIC_SKILL = "OpenJarAddSugarThenCloseJar"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    # Preserve the atomic layout for the open/add/close workflow.
    SPEC_TASK = "sugar_cube_in_jar"
    COMPOSITE_SPEC_TASK = "open_jar_add_sugar_then_close_jar"
    COMPOSITE_SCENE_IDS = (1197, 1664, 2025)
    START_LID_CLOSED = True
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "open_jar",
            "_lid_is_open",
            task_cls=HolodeckOpenMugLid,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "jar_lid"},
        ),
        HolodeckAtomicTask(
            "add_sugar",
            "_sugar_inserted",
            task_cls=HolodeckSugarCubeInJar,
            hold_steps=None,
            on_enter="_begin_sugar_stage",
            metadata={"skill": "pick_place", "object": "sugar_cube"},
        ),
        HolodeckAtomicTask(
            "close_jar",
            "_lid_closed_after_sugar",
            task_cls=HolodeckCloseMugLid,
            hold_steps=None,
            on_enter="_begin_close_lid_stage",
            metadata={"skill": "articulation", "target": "jar_lid"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_baked_scene_fixtures()

    def _lid_is_open(self):
        """Require actual departure from closed pose for the opening stage.

        The empty jar used by this workflow may have no contact pair at its
        calibrated closed pose, so contact-only detection can complete
        ``open_jar`` during reset.  Keep the correction local to this task;
        standalone lid tasks and the already-open sugar workflow retain their
        existing predicates.
        """
        return bool(
            not self._lid_is_closed() and not self._objects_in_contact("jar_lid", "jar")
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "remove the jar lid, put the sugar cube into the jar, then put "
                "the lid back on"
            ),
            source_object="sugar_cube",
            target_object="jar",
            target_lid="jar_lid",
            success_definition=(
                "remove the initially closed lid first, then keep the sugar cube "
                "fully inside the jar's outer octagonal boundary, and finally "
                "keep the lid within its calibrated closed-pose tolerance"
            ),
        )
        return meta


# Shelf workflows


import numpy as np

from robocasa.environments.holodeck.tasks.shelf_folder import (
    HolodeckRemoveFolderFromShelf,
    _HolodeckShelfFolderMixin,
)
from robocasa.environments.ithor_atomic.pick_place import resolve_task_support_region
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckRemoveFolderFromShelfToDesk(
    _HolodeckShelfFolderMixin,
    HolodeckMultiHopComposite,
):
    """Remove the calibrated folder from the shelf and place it on the desk."""

    ABSTRACT = False
    ATOMIC_SKILL = "RemoveFolderFromShelfToDesk"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    MODE = "remove"
    SPEC_TASK = "remove_folder_from_shelf"
    INSERT_SPEC_TASK = "insert_folder_into_shelf"
    SUPPORTED_SCENE_IDS = (1018, 1197, 2025)
    COMPOSITE_SPEC_TASK = "remove_folder_from_shelf_to_desk"
    COMPOSITE_SCENE_IDS = (1197, 2025)
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "remove_folder_from_shelf",
            "_folder_removed_from_shelf",
            task_cls=HolodeckRemoveFolderFromShelf,
            hold_steps=None,
            metadata={"skill": "pick", "object": "folder"},
        ),
        HolodeckAtomicTask(
            "place_folder_on_desk",
            "_folder_released_on_desk",
            task_cls=HolodeckRemoveFolderFromShelf,
            hold_steps=None,
            metadata={"skill": "place", "object": "folder"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.INSERT_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        self.destination_spec = get_holodeck_task_spec(
            self.scene_name,
            self.INSERT_SPEC_TASK,
            required=("source_table_body",),
        )
        self.source_table_fixture = self.get_fixture(
            id=self.destination_spec["source_table_body"]
        )
        self.source_desk_region = None
        if self.source_table_fixture is None:
            return
        support_regions = self.get_region(
            fixture=self.source_table_fixture, kind="support", return_all=True
        )
        if not support_regions:
            support_regions = []
            for region in self.get_region(
                fixture=self.source_table_fixture,
                kind="receptacle",
                return_all=True,
            ):
                try:
                    support_regions.append(resolve_task_support_region(region))
                except ValueError:
                    continue
        if not support_regions:
            # Some Holodeck exports do not expose desktop semantic sites. The
            # composite calibration carries an explicit task-local footprint
            # for that case, just like the insert shelf workflow.
            self.source_desk_region = self._configured_source_desk_region()
        else:
            support_region = max(
                support_regions,
                key=lambda region: (
                    float(np.prod(region.half_size[:2])),
                    region.stable_id,
                ),
            )
            self.source_desk_region = _full_desk_support_region(
                self.source_table_fixture, support_region
            )
        self.fixture_refs["destination_desk"] = self.source_table_fixture
        self.region_refs["destination_desk"] = self.source_desk_region

    def _folder_removed_from_shelf(self):
        return not self._folder_contacts_shelf()

    def _begin_folder_placement(self):
        pass

    def _folder_released_on_desk(self):
        return bool(self._folder_on_desk(require_released=False, require_bounds=False))

    def _validate_initial_composite_state(self):
        """Validate only after the shelf mixin has placed the folder."""
        if not getattr(self, "_folder_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _check_success(self):
        """Require both ordered stages, not the shelf mixin's atomic result.

        The shelf-folder mixin supplies an atomic ``_check_success`` and is
        first in this class's MRO.  Delegate explicitly to the composite
        state machine so moving the folder clear of the shelf cannot finish
        this two-hop task before the desk placement is completed.
        """
        return bool(HolodeckMultiHopComposite._check_success(self))

    def reward(self, action=None):
        return float(self._check_success())

    def _reset_internal(self):
        # ``_HolodeckShelfFolderMixin._reset_internal`` calls the composite
        # reset before it applies the calibrated folder pose. Defer the first
        # stage check until that pose and its shelf contact are ready.
        self._folder_layout_ready = False
        super()._reset_internal()
        self._folder_layout_ready = True
        self._folder_removed_before_placement = False
        # The shelf mixin places the folder after the composite base has reset
        # its generic progress hook. Re-run the state initialization against
        # the calibrated folder pose rather than the MJCF default pose.
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.source_table_fixture is not None
            and self.source_desk_region is not None
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="remove the folder from the shelf, then place it on the desk",
            success_definition=(
                "keep the folder out of contact with the calibrated shelf, "
                "then keep it in contact with the complete named desk footprint "
                "for one second"
            ),
        )
        return meta


# Study workflows


import numpy as np

from robocasa.environments.holodeck.tasks.approach_desk import HolodeckApproachDesk
from robocasa.environments.holodeck.tasks.laptop import HolodeckOpenLaptop
from robocasa.environments.holodeck.tasks.pick_book_from_dresser import (
    HolodeckPickBookFromDresser,
)
from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.push_chair_to_desk import (
    HolodeckPushChairToDesk,
)
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_atomic.pick_place import resolve_task_support_region
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robocasa.models.scenes.thor_semantics import SemanticRegion


def _largest_support_region(env, fixture):
    """Return the largest usable support region on a fixture."""
    regions = env.get_region(fixture=fixture, kind="support", return_all=True)
    if not regions:
        regions = []
        for region in env.get_region(
            fixture=fixture, kind="receptacle", return_all=True
        ):
            try:
                regions.append(resolve_task_support_region(region))
            except ValueError:
                continue
    if not regions:
        raise RuntimeError(
            f"{env.scene_name}: target desk has no usable support region"
        )
    return max(
        regions,
        key=lambda region: (float(np.prod(region.half_size[:2])), region.stable_id),
    )


def _full_desk_support_region(fixture, support_region):
    """Build a placement region covering the desk's complete horizontal footprint.

    Desk exports can expose the desktop as a narrow receptacle site (for
    example, only the front strip).  The fixture bounds describe the complete
    desk footprint, while the selected support region supplies the measured
    tabletop height.  Keep this view task-local so the shared semantic
    registry remains unchanged.
    """

    half_size = np.asarray(fixture.half_size, dtype=float).copy()
    half_size[2] = np.asarray(support_region.half_size, dtype=float)[2]
    support_center = np.asarray(support_region.world_pose()[0], dtype=float)
    fixture_center = np.asarray(fixture.center, dtype=float)
    center = support_center.copy()
    center[:2] = fixture_center[:2]
    return SemanticRegion(
        stable_id=f"{fixture.stable_id}:region:full_desktop",
        fixture_id=support_region.fixture_id,
        kind="support",
        parent_body=None,
        local_center=center.copy(),
        half_size=half_size,
        local_yaw=0.0,
        _world_center=center,
        _world_yaw=0.0,
    )


class HolodeckPickBookFromSourceToDesk(
    HolodeckPickBookFromDresser,
    HolodeckMultiHopComposite,
):
    """Pick up the calibrated book and place it on the desk."""

    ABSTRACT = False
    ATOMIC_SKILL = "PickCarryPlaceBook"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUCCESS_DEBUG_INTERVAL_SECONDS = 5.0
    # The destination desk calibration exists only for the original book
    # workflow scenes. Do not inherit the atomic pick task's wider scope.
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    DESTINATION_SPEC_TASK = "pick_book_to_desk"
    # These destination desks contain a baked laptop that blocks the
    # calibrated book workflow. Remove it only for these book workflows.
    CALIBRATED_LAPTOP_BODIES = {
        "train_117": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
        "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
    }
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "pick_book",
            "_book_lifted_from_source",
            task_cls=HolodeckPickBookFromDresser,
            hold_steps=None,
            metadata={"skill": "pick", "object": "book"},
        ),
        HolodeckAtomicTask(
            "place_book_on_desk",
            "_book_on_desk",
            task_cls=HolodeckPickBookFromDresser,
            hold_steps=None,
            metadata={"skill": "place", "object": "book", "target": "desk"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.destination_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DESTINATION_SPEC_TASK,
            required=("desk_body",),
        )
        self.destination_desk_fixture = self.get_fixture(
            id=self.destination_spec["desk_body"]
        )
        if self.destination_desk_fixture is None:
            return
        support_region = _largest_support_region(self, self.destination_desk_fixture)
        self.destination_region = _full_desk_support_region(
            self.destination_desk_fixture, support_region
        )
        self.fixture_refs["destination_desk"] = self.destination_desk_fixture
        self.region_refs["destination_desk"] = self.destination_region

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.destination_desk_fixture is not None
            and self.destination_region is not None
        )

    def _validate_initial_composite_state(self):
        # HolodeckPickBookFromDresser places the book after the composite reset
        # hook returns. Validate the actual source pose below that placement.
        if not getattr(self, "_book_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _reset_internal(self):
        super()._reset_internal()
        self._book_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS

    def _post_action(self, action):
        # This composite intentionally measures only the two observable book
        # states below.  Do not inherit the atomic pick task's grasp history.
        ret = HolodeckMultiHopComposite._post_action(self, action)
        self._log_success_conditions()
        return ret

    def _book_lifted_from_source(self):
        return bool(self._book_root_z() > self._book_initial_z + self.lift_delta)

    def _book_on_desk(self):
        return bool(
            self._book_on_region(
                self.destination_region,
                require_contact=True,
                require_bounds=False,
            )
        )

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self) and self._book_on_desk()
        )

    def _log_success_conditions(self):
        """Print the pick and placement predicates every five simulated seconds."""
        sim_time = float(self.sim.data.time)
        interval = float(self.SUCCESS_DEBUG_INTERVAL_SECONDS)
        next_log_time = getattr(self, "_success_debug_next_log_time", interval)
        if sim_time < next_log_time:
            return
        while next_log_time <= sim_time:
            next_log_time += interval
        self._success_debug_next_log_time = next_log_time

        def status(condition):
            return "OK" if condition else "MISSING"

        book_z = self._book_root_z()
        lift_delta = book_z - self._book_initial_z
        lifted = book_z > self._book_initial_z + self.lift_delta
        pick_success = self._book_lifted_from_source()
        book_on_desk = self._book_on_region(
            self.destination_region,
            require_contact=True,
            require_bounds=False,
        )
        placement_success = self._book_on_desk()
        stage = self.current_atomic_task
        stage_name = stage.name if stage is not None else "complete"
        stage_target = self._required_stage_hold_steps(stage) if stage else 0

        print(
            f"[{type(self).__name__}][{self.scene_name}] "
            f"t={sim_time:.2f}s stage={stage_name} "
            f"hold={self._stage_hold_steps}/{stage_target}; "
            "pick_book("
            f"lift_delta={lift_delta:.3f}m>{self.lift_delta:.3f}m="
            f"{status(lifted)}, "
            f"success={status(pick_success)}); "
            "place_book_on_desk("
            f"support_contact={status(book_on_desk)}, "
            f"success={status(placement_success)}); "
            f"task={status(self._check_success())}",
            flush=True,
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="pick up the book, then place it on the desk",
            source_fixture=self.source_fixture.stable_id,
            target_fixture=self.destination_desk_fixture.stable_id,
            source_object="book",
            success_definition=(
                "keep the book root more than "
                f"{self.lift_delta:.2f} m above its reset height for "
                f"{HolodeckPickBookFromDresser.HOLD_SECONDS:.1f} s, then keep its "
                "book body in contact with the desk for "
                f"{HolodeckPickBookFromDresser.HOLD_SECONDS:.1f} s"
            ),
        )
        return meta


class HolodeckApproachDeskPullChairThenOpenLaptop(
    HolodeckOpenLaptop,
    HolodeckMultiHopComposite,
):
    """Reach the desk, pull out its chair, and open the desk laptop."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachPullChairThenOpenLaptop"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    COMPOSITE_SPEC_TASK = "approach_desk_pull_chair_then_open_laptop"
    COMPOSITE_SCENE_IDS = (1197, 1664, 2025)
    SPEC_TASK = "open_laptop"
    BEHAVIOR = "open"
    OPEN_INIT_FRACTION = 0.0
    SUCCESS_OPEN_FRACTION = 0.95
    NAVIGATION_SPEC_TASK = "approach_desk"
    CHAIR_SPEC_TASK = "pull_chair_from_desk"
    CHAIR_MASS = HolodeckPushChairToDesk.CHAIR_MASS
    CHAIR_INERTIA = HolodeckPushChairToDesk.CHAIR_INERTIA
    CHAIR_CONTACT_FRICTION = HolodeckPushChairToDesk.CHAIR_CONTACT_FRICTION
    MIN_CHAIR_PULL_DISPLACEMENT = HolodeckPullChairFromDesk.MIN_PULL_DISPLACEMENT
    CHAIR_ASSET = HolodeckPushChairToDesk.CHAIR_ASSET
    CHAIR_SPAWN_NAME = HolodeckPushChairToDesk.CHAIR_SPAWN_NAME
    LAPTOP_BODY = "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    LAPTOP_RELOCATIONS = {
        "train_117": {
            # Shift toward the robot-side edge of the calibrated desk.
            "pos": [5.550000, 4.800000, 0.774781],
            "quat": [0.5, 0.5, -0.5, -0.5],
        },
        # The chair in room 2 is paired with desk_0ec..., where the laptop
        # already has a calibrated baked-scene pose.
        "train_1018": {
            "pos": [7.504230, 0.886466, 0.798343],
            "quat": [0.5, 0.5, -0.5, -0.5],
        },
    }
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "pull_chair",
            "_chair_pulled_after_approach",
            task_cls=HolodeckPullChairFromDesk,
            hold_steps=None,
            on_enter="_begin_chair_stage",
            metadata={"skill": "pull", "object": "chair"},
        ),
        HolodeckAtomicTask(
            "open_laptop",
            "_laptop_open_after_chair",
            task_cls=HolodeckOpenLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.CHAIR_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        relocation = self.LAPTOP_RELOCATIONS.get(self.scene_name)
        if relocation is not None:
            laptop_body = self.find_scene_body(self.LAPTOP_BODY)
            if laptop_body is None:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{self.LAPTOP_BODY}"
                )
            laptop_body.set(
                "pos", " ".join(f"{value:.6f}" for value in relocation["pos"])
            )
            laptop_body.set(
                "quat", " ".join(f"{value:.6f}" for value in relocation["quat"])
            )
        if self.scene_name == "train_1018":
            # This composite inherits the laptop primitive rather than the
            # pull-chair initializer. Remove the baked chair before spawning
            # the task-local sliding chair from the shared pull spec.
            chair_spec = get_holodeck_task_spec(self.scene_name, self.CHAIR_SPEC_TASK)
            remove_bodies = tuple(chair_spec.get("remove_scene_bodies", ()))
            removed = self.remove_scene_bodies(remove_bodies)
            if missing := set(remove_bodies) - removed:
                raise RuntimeError(
                    f"{self.scene_name}: configured chair blockers are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        # Laptop setup is provided by the OpenLaptop MRO. Add only the chair
        # calibration here so the two primitives do not overwrite task_spec.
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.CHAIR_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.task_spec
        self.chair_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.CHAIR_SPEC_TASK,
            required=(
                "chair_body",
                "chair_joint",
                "desk_body",
                "chair_start_pose",
                "robot_pose",
                "target_xy",
                "target_tolerance",
            ),
        )
        self._pull_origin_xy = np.asarray(
            self.chair_task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        self.task_spec = self.chair_task_spec
        # This composite reaches the chair calibration through a sibling task
        # in the MRO, so explicitly create the task-local chair here.
        HolodeckPushChairToDesk._ensure_task_chair_fixture(self)
        self.chair_body = self.task_spec["chair_body"]
        self.chair_joint = self.task_spec["chair_joint"]
        HolodeckPushChairToDesk._configure_chair_dynamics(self)
        self.target_xy = np.asarray(self.task_spec["target_xy"], dtype=float)
        self.target_tolerance = np.asarray(
            self.task_spec["target_tolerance"], dtype=float
        )
        self.chair_fixture = next(
            (
                fixture
                for fixture in self.get_fixture(category="chair", return_all=True)
                if fixture.body_name == self.chair_body
            ),
            None,
        )
        self.desk_fixture = next(
            (
                fixture
                for fixture in self.get_fixture(category="desk", return_all=True)
                if fixture.body_name == self.task_spec["desk_body"]
            ),
            None,
        )
        if self.desk_fixture is None:
            desk_body = self.task_spec["desk_body"]
            self.desk_fixture = next(
                (
                    fixture
                    for fixture in tuple(
                        getattr(self.semantic_scene, "fixtures", {}).values()
                    )
                    + tuple(getattr(self, "fixtures", {}).values())
                    if fixture.body_name == desk_body
                ),
                None,
            )
        HolodeckPullChairFromDesk._disable_desk_drawer_collisions(self)
        if self.chair_fixture is not None:
            self.fixture_refs["pull_chair"] = self.chair_fixture
        if self.desk_fixture is not None:
            self.fixture_refs["chair_desk"] = self.desk_fixture

        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.navigation_desk_fixture = self.get_fixture(
            id=self.navigation_spec["desk_body"]
        )
        if self.navigation_desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
            pose = self.navigation_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )
        self.task_spec = self.laptop_task_spec

    def _is_scene_valid(self):
        return bool(
            self.navigation_desk_fixture is not None
            and self.chair_fixture is not None
            and self.target_fixture is not None
        )

    def _place_robot_for_articulation(self, distance):
        """Keep the far navigation spawn instead of laptop's close pose."""
        del distance
        self.sim.forward()
        return self.count_body_collisions() == 0

    def _setup_scene(self):
        self.task_spec = self.laptop_task_spec
        HolodeckOpenLaptop._setup_scene(self)
        self.task_spec = self.chair_task_spec
        HolodeckPushChairToDesk._set_initial_chair_state(self, self.chair_task_spec)
        self.sim.forward()
        chair_xy = np.asarray(
            self.sim.data.body_xpos[self.sim.model.body_name2id(self.chair_body)][:2],
            dtype=float,
        )
        chair_start_xy = np.asarray(
            self.chair_task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        if (
            np.linalg.norm(chair_xy - chair_start_xy)
            >= HolodeckPullChairFromDesk.MIN_PULL_DISPLACEMENT
        ):
            raise RuntimeError(
                f"{self.scene_name}: pull-chair reset already exceeds the "
                "required displacement"
            )
        self.task_spec = self.laptop_task_spec
        distance = self.robot_fixture_collision_distance(
            self.navigation_desk_fixture, max_distance=self.success_radius + 1.0
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: composite navigation spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        super()._reset_internal()
        HolodeckPushChairToDesk._set_initial_chair_state(self, self.chair_task_spec)
        self.sim.forward()
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.navigation_desk_fixture,
                max_distance=self.success_radius + 0.25,
            )
            <= self.success_radius
        )

    def _begin_chair_stage(self):
        body_id = self.sim.model.body_name2id(self.chair_body)
        self._chair_stage_start_xy = np.asarray(
            self.sim.data.body_xpos[body_id][:2], dtype=float
        ).copy()

    def _chair_pulled_after_approach(self):
        return HolodeckPullChairFromDesk._chair_has_min_pull_displacement(self)

    def _laptop_open_after_chair(self):
        # The ordered composite state machine already records completion of
        # the chair stage.  Chair position must not constrain this stage.
        return bool(RoomOpenArticulation._raw_success(self))

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._laptop_open_after_chair()
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, pull out the chair, then open the laptop",
            target_fixture=self.navigation_desk_fixture.stable_id,
            target_object="laptop",
            target_joint=self.target_joint,
            success_definition=(
                "first reach the calibrated desk, then move the chair center at "
                "least "
                f"{self.MIN_CHAIR_PULL_DISPLACEMENT:.2f} m, "
                "from its reset position and keep it there for 1 second, then "
                "open the calibrated laptop lid; the chair's final position is "
                "irrelevant after the pull stage"
            ),
        )
        return meta


class HolodeckMoveCupToCenterThenOpenLaptop(
    HolodeckOpenLaptop,
    HolodeckMultiHopComposite,
):
    """Move the glass cup from its corner to center, then open the laptop."""

    ABSTRACT = False
    ATOMIC_SKILL = "MoveCupThenOpenLaptop"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    COMPOSITE_SPEC_TASK = "move_cup_to_center_then_open_laptop"
    COMPOSITE_SCENE_IDS = (1197, 1664, 2025)
    SPEC_TASK = "open_laptop"
    BEHAVIOR = "open"
    OPEN_INIT_FRACTION = 0.0
    SUCCESS_OPEN_FRACTION = 0.95
    # The cup stage uses the center atomic task's calibrated layout verbatim.
    CUP_SPEC_TASK = HolodeckMoveCupToCenter.SPEC_TASK
    # This composite delegates its first-stage reset validation to the cup
    # primitive, so it must carry the primitive's calibration contract.
    INITIAL_CUP_DISTANCE = HolodeckMoveCupToCenter.INITIAL_CUP_DISTANCE
    INITIAL_MIN_TARGET_CLEARANCE = HolodeckMoveCupToCenter.INITIAL_MIN_TARGET_CLEARANCE
    INITIAL_DESK_COLLISION_DISTANCE = (
        HolodeckMoveCupToCenter.INITIAL_DESK_COLLISION_DISTANCE
    )
    INITIAL_DESK_DISTANCE_TOLERANCE = (
        HolodeckMoveCupToCenter.INITIAL_DESK_DISTANCE_TOLERANCE
    )
    INITIAL_REQUIRE_LATERAL_TARGET_OFFSET = (
        HolodeckMoveCupToCenter.INITIAL_REQUIRE_LATERAL_TARGET_OFFSET
    )
    INITIAL_REQUIRE_RIGHT_TARGET_OFFSET = (
        HolodeckMoveCupToCenter.INITIAL_REQUIRE_RIGHT_TARGET_OFFSET
    )
    INITIAL_MAX_LONGITUDINAL_TARGET_OFFSET = (
        HolodeckMoveCupToCenter.INITIAL_MAX_LONGITUDINAL_TARGET_OFFSET
    )
    # Match the center atomic task's known G1 collision-model exceptions when
    # the composite keeps its cup-stage spawn pose during laptop setup.
    INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES = (
        HolodeckMoveCupToCenter.INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES
    )
    LAPTOP_BODY = "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    LAPTOP_RELOCATIONS = {
        "train_117": {
            # Keep the laptop on the far side of the target. The center
            # atomic task starts the cup at y=4.40, which overlaps the
            # Laptop_26 collision footprint at the old y=4.20 pose.
            "pos": [5.670000, 5.200000, 0.774781],
            "quat": [0.5, 0.5, -0.5, -0.5],
        },
        "train_1018": {
            # Mirror the same left-side placement in room 2.
            "pos": [3.250000, 0.650000, 0.885105],
            "quat": [0.5, 0.5, 0.5, 0.5],
        },
    }
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "move_cup_to_center",
            "_cup_centered",
            task_cls=HolodeckMoveCupToCenter,
            hold_steps=5,
            on_enter="_begin_cup_stage",
            metadata={"skill": "pick_place", "object": "glass_cup"},
        ),
        HolodeckAtomicTask(
            "open_laptop",
            "_laptop_open_after_cup",
            task_cls=HolodeckOpenLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.CUP_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        composite_spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        replaced_body = HolodeckMoveCupToCenter.CALIBRATED_REPLACED_BODIES.get(
            self.scene_name
        )
        if replaced_body is not None and replaced_body != composite_spec.get(
            "source_body"
        ):
            removed = self.remove_scene_bodies((replaced_body,))
        else:
            removed = set()
        if (
            replaced_body is not None
            and replaced_body != composite_spec.get("source_body")
            and replaced_body not in removed
        ):
            raise RuntimeError(
                f"{self.scene_name}: configured source body is missing: "
                f"{replaced_body}"
            )
        relocation = self.LAPTOP_RELOCATIONS.get(self.scene_name)
        if relocation is not None:
            laptop_body = self.find_scene_body(self.LAPTOP_BODY)
            if laptop_body is None:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{self.LAPTOP_BODY}"
                )
            laptop_body.set(
                "pos", " ".join(f"{value:.6f}" for value in relocation["pos"])
            )
            laptop_body.set(
                "quat", " ".join(f"{value:.6f}" for value in relocation["quat"])
            )
            # In the requested cup workflows, only the lid is actuated; keep
            # the laptop lower half fixed at this calibrated desk pose.
            if self.scene_name in {"train_117", "train_1018"}:
                self._fix_laptop_base(self.LAPTOP_BODY)
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.CUP_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.task_spec
        self.cup_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.CUP_SPEC_TASK,
            required=(
                "desk_body",
                "cup_scale",
                "cup_pose",
                "initial_cup_base_xy",
                "robot_pose",
                "target_base_xy",
                "target_tolerance",
                "support_z",
                "support_z_tolerance",
                "upright_local_axis",
                "min_upright_cosine",
                "max_linear_speed",
                "max_angular_speed",
            ),
        )
        self.task_spec = self.cup_task_spec
        self.desk_fixture = next(
            (
                fixture
                for fixture in self.get_fixture(category="desk", return_all=True)
                if fixture.body_name == self.cup_task_spec["desk_body"]
            ),
            None,
        )
        if self.desk_fixture is None:
            desk_body = self.cup_task_spec["desk_body"]
            self.desk_fixture = next(
                (
                    fixture
                    for fixture in tuple(
                        getattr(self.semantic_scene, "fixtures", {}).values()
                    )
                    + tuple(getattr(self, "fixtures", {}).values())
                    if fixture.body_name == desk_body
                ),
                None,
            )
        if self.desk_fixture is not None:
            self.fixture_refs["cup_surface"] = self.desk_fixture

        # The cup task is the first stage and determines the initial stand pose.
        pose = self.cup_task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(
                robocasa.models.assets_root, HolodeckMoveCupToCenter.CUP_ASSET
            ),
            scale=float(self.cup_task_spec["cup_scale"]),
        )
        self.objects[cup.name] = cup
        self.model.merge_objects([cup])

    def _is_scene_valid(self):
        return bool(self.desk_fixture is not None and self.target_fixture is not None)

    def _place_robot_for_articulation(self, distance):
        """Keep the far cup-stage spawn instead of laptop's close pose."""
        del distance
        self.sim.forward()
        return bool(
            self.count_body_collisions() == 0
            or self.scene_name in self.INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES
        )

    def _finalize_articulation_robot_pose(self, distance):
        """Retain the cup-stage pose for laptop setup.

        ``RoomOpenArticulation`` bypasses ``_place_robot_for_articulation``
        when a calibrated laptop spec is active, so apply the same exception
        explicitly for scenes whose atomic cup calibration uses the fixed G1
        collision model.
        """
        del distance
        self.sim.forward()
        return bool(
            self.count_body_collisions() == 0
            or self.scene_name in self.INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES
        )

    def _setup_scene(self):
        self.task_spec = self.cup_task_spec
        self._set_initial_cup_state()
        self._success_hold_steps = 0
        self.task_spec = self.laptop_task_spec
        HolodeckOpenLaptop._setup_scene(self)
        self.task_spec = self.cup_task_spec
        self.sim.forward()
        if self._laptop_cup_colliding():
            raise RuntimeError(
                f"{self.scene_name}: laptop and glass cup collide at reset"
            )

    def _set_initial_cup_state(self):
        joint = self.objects["cup"].joints[0]
        pose = dict(self.cup_task_spec["cup_pose"])
        pose["pos"] = [
            *self.cup_task_spec["initial_cup_base_xy"],
            float(self.cup_task_spec["cup_pose"]["pos"][2]),
        ]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))

    def _laptop_cup_colliding(self):
        laptop_root = self.sim.model.body_name2id(self.laptop_task_spec["target_body"])
        cup_root = self.obj_body_id["cup"]
        laptop_bodies = self._body_tree_ids([laptop_root])
        cup_bodies = self._body_tree_ids([cup_root])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in laptop_bodies and body2 in cup_bodies) or (
                body2 in laptop_bodies and body1 in cup_bodies
            ):
                return True
        return False

    def _reset_internal(self):
        self._cup_layout_ready = False
        super()._reset_internal()
        self.task_spec = self.cup_task_spec
        self._set_initial_cup_state()
        self.sim.forward()
        if (
            self.scene_name not in self.INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES
            and not HolodeckMoveCupToCenter._initial_layout_valid(self)
        ):
            raise RuntimeError(f"{self.scene_name}: cup reset layout is invalid")
        self._success_hold_steps = 0
        self.task_spec = self.laptop_task_spec
        self._cup_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_cup_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _begin_cup_stage(self):
        pass

    def _cup_call(self, method):
        saved_spec = self.task_spec
        self.task_spec = self.cup_task_spec
        try:
            return getattr(HolodeckMoveCupToCenter, method)(self)
        finally:
            self.task_spec = saved_spec

    def _cup_centered(self):
        return bool(self._cup_call("_cup_in_target_state"))

    def _initial_cup_base_xy(self):
        return HolodeckMoveCupToCenter._initial_cup_base_xy(self)

    def _initial_layout_valid(self):
        saved_spec = self.task_spec
        self.task_spec = self.cup_task_spec
        try:
            return HolodeckMoveCupToCenter._initial_layout_valid(self)
        finally:
            self.task_spec = saved_spec

    def _uses_g1_collision_model(self):
        return HolodeckMoveCupToCenter._uses_g1_collision_model(self)

    def _cup_body_ids(self):
        return HolodeckMoveCupToCenter._cup_body_ids(self)

    def _desk_body_ids(self):
        return HolodeckMoveCupToCenter._desk_body_ids(self)

    def _cup_base_center(self):
        return HolodeckMoveCupToCenter._cup_base_center(self)

    def _cup_bottom_z(self):
        return HolodeckMoveCupToCenter._cup_bottom_z(self)

    def _cup_is_upright(self):
        return HolodeckMoveCupToCenter._cup_is_upright(self)

    def _cup_contacts_desk(self):
        return HolodeckMoveCupToCenter._cup_contacts_desk(self)

    def _cup_released(self):
        return HolodeckMoveCupToCenter._cup_released(self)

    def _cup_in_target_region(self, base_center):
        return HolodeckMoveCupToCenter._cup_in_target_region(self, base_center)

    def _target_base_xy(self):
        return HolodeckMoveCupToCenter._target_base_xy(self)

    def _target_tolerance(self):
        return HolodeckMoveCupToCenter._target_tolerance(self)

    def _laptop_open_after_cup(self):
        return bool(self._cup_centered() and RoomOpenArticulation._raw_success(self))

    def _post_action(self, action):
        return super()._post_action(action)

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._laptop_open_after_cup()
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="move the glass cup from the corner to the center, then open the laptop",
            target_fixture=self.desk_fixture.stable_id,
            target_object="laptop",
            target_joint=self.laptop_task_spec["target_joint"],
            success_definition=(
                "place the calibrated glass cup upright and in contact with the desk "
                "at its center, "
                "then open the calibrated laptop lid"
            ),
        )
        return meta


class HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop(
    _HolodeckShelfFolderMixin,
    RoomOpenArticulation,
    HolodeckMultiHopComposite,
):
    """Remove a shelf folder, place it on a desk, and open the laptop."""

    ABSTRACT = False
    ATOMIC_SKILL = "RemoveFolderToDeskThenOpenLaptop"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    MODE = "remove"
    SPEC_TASK = "remove_folder_from_shelf"
    LAPTOP_SPEC_TASK = "open_laptop"
    INSERT_SPEC_TASK = "insert_folder_into_shelf"
    # This workflow owns a scene-specific laptop calibration and spawns that
    # laptop explicitly below. Do not also apply HolodeckEnv's default laptop
    # replacement, which would remove the same baked source body twice.
    USE_DEFAULT_ASSET_LAYOUT = False
    SUPPORTED_SCENE_IDS = (1018, 1197, 2025)
    COMPOSITE_SPEC_TASK = "remove_folder_from_shelf_to_desk_then_open_laptop"
    COMPOSITE_SCENE_IDS = (1197, 2025)

    TARGET_CATEGORIES = ("Laptop",)
    LAPTOP_ASSET = _HolodeckLaptopMixin.LAPTOP_ASSET
    LAPTOP_ASSET_ID = _HolodeckLaptopMixin.LAPTOP_ASSET_ID
    LAPTOP_BODY = "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    # Move the baked laptop to the center of the desk beside the shelf spawn.
    LAPTOP_RELOCATION = {
        "pos": [3.250000, 1.250000, 0.885105],
        "quat": [0.5, 0.5, 0.5, 0.5],
    }
    TARGET_LANG = "laptop lid"
    JOINT_TOKENS = ("lid",)
    TARGET_JOINT_TYPES = ("hinge",)
    SUCCESS_JOINT_SELECTION = "target"
    BEHAVIOR = "open"
    OPEN_INIT_FRACTION = 0.0
    SUCCESS_OPEN_FRACTION = 0.95
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)

    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "remove_folder_from_shelf",
            "_folder_removed_from_shelf",
            task_cls=HolodeckRemoveFolderFromShelf,
            hold_steps=None,
            metadata={"skill": "pick", "object": "folder", "target": "shelf"},
        ),
        HolodeckAtomicTask(
            "place_folder_on_desk",
            "_folder_released_on_desk",
            task_cls=HolodeckRemoveFolderFromShelf,
            hold_steps=None,
            metadata={"skill": "place", "object": "folder", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "open_laptop",
            "_laptop_open_after_folder",
            task_cls=HolodeckOpenLaptop,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
    )

    def _get_articulation_task_spec(self):
        """Load the laptop calibration while the folder uses ``SPEC_TASK``."""
        key = (self.scene_name, self.LAPTOP_SPEC_TASK)
        if getattr(self, "_articulation_task_spec_key", None) != key:
            self.articulation_task_spec = get_holodeck_task_spec(
                self.scene_name,
                self.LAPTOP_SPEC_TASK,
                required=(
                    "target_body",
                    "target_joint",
                    "robot_pose",
                    "closed_qpos",
                    "open_qpos",
                ),
            )
            self._articulation_task_spec_key = key
        return self.articulation_task_spec

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _articulation_fixture_query(self):
        """Use the laptop body calibration instead of a fixture stable ID."""
        return None

    def _select_semantic_candidate(self, candidates):
        """Select the exact calibrated laptop body from semantic candidates."""
        target_body = self._get_articulation_task_spec()["target_body"]
        return next(
            (fixture for fixture in candidates if fixture.body_name == target_body),
            None,
        )

    def _configure_articulation_task(self):
        # Laptop calibrations specify qpos endpoints, not the generic iTHOR
        # open-fraction fields.
        self.OPEN_INIT_FRACTION = 0.0
        self.SUCCESS_OPEN_FRACTION = 0.95
        self.HOLD_STEPS = RoomOpenArticulation.HOLD_STEPS

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.LAPTOP_SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.INSERT_SPEC_TASK = self.COMPOSITE_SPEC_TASK
            spec = get_holodeck_task_spec(self.scene_name, self.COMPOSITE_SPEC_TASK)
            remove_scene_bodies = list(spec.get("remove_scene_bodies", ()))
            source_body = spec.get("source_body")
            if source_body and source_body not in remove_scene_bodies:
                remove_scene_bodies.append(source_body)
            if remove_scene_bodies:
                removed = self.remove_scene_bodies(tuple(remove_scene_bodies))
                missing = set(remove_scene_bodies) - removed
                if missing:
                    raise RuntimeError(
                        f"{self.scene_name}: configured laptop bodies are missing: "
                        f"{sorted(missing)}"
                    )
        else:
            laptop_body = self.find_scene_body(self.LAPTOP_BODY)
            if laptop_body is None:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{self.LAPTOP_BODY}"
                )
            laptop_body.set(
                "pos",
                " ".join(f"{value:.6f}" for value in self.LAPTOP_RELOCATION["pos"]),
            )
            laptop_body.set(
                "quat",
                " ".join(f"{value:.6f}" for value in self.LAPTOP_RELOCATION["quat"]),
            )
            if self.scene_name == "train_1018":
                _HolodeckLaptopMixin._fix_laptop_base(self, self.LAPTOP_BODY)
        super()._setup_baked_scene_fixtures()

    def _finalize_articulation_robot_pose(self, distance):
        """Defer the collision check until the shelf spawn is calibrated."""
        del distance
        self.sim.forward()
        return True

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.LAPTOP_SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.INSERT_SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.task_spec = self._get_articulation_task_spec()
            _HolodeckLaptopMixin._spawn_laptop(self)
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.articulation_task_spec

        destination_spec = get_holodeck_task_spec(
            self.scene_name,
            self.INSERT_SPEC_TASK,
            required=("source_table_body",),
        )
        self.destination_spec = destination_spec
        self.source_table_fixture = self._find_laptop_desk()
        self.source_desk_region = None
        if self.source_table_fixture is not None:
            support_regions = self._laptop_desk_support_regions(
                self.source_table_fixture
            )
            if not support_regions:
                # Objaverse desks may have no semantic tabletop sites. The
                # folder calibration carries the task-local desktop footprint
                # for exactly this export case.
                self.source_desk_region = self._configured_source_desk_region()
            else:
                laptop_pose = self.laptop_task_spec.get("laptop_pose")
                laptop_xy = np.asarray(
                    (laptop_pose or self.LAPTOP_RELOCATION)["pos"][:2], dtype=float
                )
                containing = [
                    region
                    for region in support_regions
                    if region.contains_xy(laptop_xy)
                ]
                self.source_desk_region = max(
                    containing or support_regions,
                    key=lambda region: (
                        float(np.prod(region.half_size[:2]))
                        if containing
                        else -float(
                            np.linalg.norm(region.world_pose()[0][:2] - laptop_xy)
                        ),
                        region.stable_id,
                    ),
                )
            self.fixture_refs["destination_desk"] = self.source_table_fixture
            self.region_refs["destination_desk"] = self.source_desk_region

        # The shelf mixin leaves the folder calibration in task_spec. Keep it
        # there because all folder predicates read its calibrated regions.
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "shelf_body",
                "robot_pose",
                "folder_pose",
                "folder_scale",
                "region_center",
                "region_half_size",
                "source_center",
                "source_half_size",
            ),
        )
        self._apply_folder_scale_override()

    def _laptop_desk_support_regions(self, fixture):
        """Return physical support views for every usable desktop region."""
        regions = self.get_region(fixture=fixture, kind="support", return_all=True)
        for region in self.get_region(
            fixture=fixture, kind="receptacle", return_all=True
        ):
            try:
                regions.append(resolve_task_support_region(region))
            except ValueError:
                continue
        return regions

    def _find_laptop_desk(self):
        """Find the nearest desk whose semantic footprint contains the laptop."""
        laptop_pose = self.laptop_task_spec.get("laptop_pose")
        laptop_xy = np.asarray(
            (laptop_pose or self.LAPTOP_RELOCATION)["pos"][:2], dtype=float
        )
        robot_xy = np.asarray(self.task_spec["robot_pose"]["pos"][:2], dtype=float)
        scored = []
        for fixture in self.get_fixture(category="desk", return_all=True):
            regions = self._laptop_desk_support_regions(fixture)
            if any(region.contains_xy(laptop_xy) for region in regions):
                distance = float(np.linalg.norm(fixture.center[:2] - robot_xy))
                scored.append((distance, fixture.stable_id, fixture))
        if scored:
            return min(scored, key=lambda item: item[:2])[2]
        return self.get_fixture(id=self.destination_spec["source_table_body"])

    def _is_scene_valid(self):
        return bool(
            self.shelf_fixture is not None
            and self.source_table_fixture is not None
            and self.source_desk_region is not None
            and self.target_fixture is not None
        )

    def _setup_scene(self):
        # The MRO runs RoomOpenArticulation first and the shelf mixin then
        # calibrates the actual folder spawn. Restore the laptop's closed
        # initial qpos after both setup paths have completed.
        super()._setup_scene()
        self.task_spec = self.laptop_task_spec
        joint = self.target_semantic_joint
        joint.closed_qpos = float(self.laptop_task_spec["closed_qpos"])
        joint.open_qpos = float(self.laptop_task_spec["open_qpos"])
        self.CLOSED_QPOS = joint.closed_qpos
        self._set_laptop_initial_state()
        self.sim.forward()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "shelf_body",
                "robot_pose",
                "folder_pose",
                "folder_scale",
                "region_center",
                "region_half_size",
                "source_center",
                "source_half_size",
            ),
        )
        self._apply_folder_scale_override()

    def _set_laptop_initial_state(self):
        self.sim.data.set_joint_qpos(
            self.target_joint, self.target_semantic_joint.closed_qpos
        )
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)

    def _folder_removed_from_shelf(self):
        return not self._folder_contacts_shelf()

    def _begin_folder_placement(self):
        pass

    def _folder_released_on_desk(self):
        return bool(self._folder_on_desk(require_released=False))

    def _laptop_open_after_folder(self):
        return bool(RoomOpenArticulation._raw_success(self))

    def _post_action(self, action):
        ret = super()._post_action(action)
        # self._log_success_conditions()
        return ret

    def _validate_initial_composite_state(self):
        if not getattr(self, "_folder_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _reset_internal(self):
        self._folder_layout_ready = False
        super()._reset_internal()
        self._set_laptop_initial_state()
        self.sim.forward()
        self._folder_layout_ready = True
        self._folder_removed_before_placement = False
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _check_success(self):
        return bool(HolodeckMultiHopComposite._check_success(self))

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "remove the folder completely from the shelf, place it on the "
                "desk, then open the laptop"
            ),
            source_fixture=self.shelf_fixture.stable_id,
            target_fixture=self.source_table_fixture.stable_id,
            target_object="laptop",
            target_joint=self.target_joint,
            success_definition=(
                "first completely clear the folder from the calibrated shelf, "
                "then place it on the desk, and finally open the laptop"
            ),
        )
        return meta


class HolodeckCloseLaptopCloseDrawerThenApproachWindow(
    HolodeckCloseLaptop,
    HolodeckMultiHopComposite,
):
    """Close the laptop, close the calibrated drawer, and reach the right window."""

    ABSTRACT = False
    ATOMIC_SKILL = "CloseLaptopCloseDrawerThenApproachWindow"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    DRAWER_SPEC_TASK = "close_drawer"
    WINDOW_SPEC_TASK = "approach_window"

    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "close_laptop",
            "_laptop_closed",
            task_cls=HolodeckCloseLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
        HolodeckAtomicTask(
            "close_drawer",
            "_drawer_closed_after_laptop",
            task_cls=HolodeckCloseDrawer,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "specified_drawer"},
        ),
        HolodeckAtomicTask(
            "approach_rightmost_window",
            "_window_reached",
            task_cls=HolodeckApproachWindow,
            hold_steps=HolodeckApproachWindow.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "rightmost_window"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.task_spec
        self.drawer_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DRAWER_SPEC_TASK,
            required=(
                "target_fixture",
                "target_joint",
                "closed_qpos",
                "open_qpos",
                "success_close_fraction",
            ),
        )
        self.drawer_fixture = self.get_fixture(
            id=self.drawer_task_spec["target_fixture"]
        )
        self.drawer_joint = (
            self.drawer_fixture.joints.get(self.drawer_task_spec["target_joint"])
            if self.drawer_fixture is not None
            else None
        )
        if self.drawer_fixture is None or self.drawer_joint is None:
            raise RuntimeError(
                f"{self.scene_name}: calibrated drawer target is missing"
            )
        self.window_spec = get_holodeck_task_spec(
            self.scene_name,
            self.WINDOW_SPEC_TASK,
            required=("window_body", "success_radius"),
        )
        self.window_body = self.window_spec["window_body"]
        self.window_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.window_body
            ),
            None,
        )
        self.fixture_refs["close_drawer"] = self.drawer_fixture
        if self.window_fixture is not None:
            self.fixture_refs["approach_window"] = self.window_fixture

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and getattr(self, "drawer_fixture", None) is not None
            and getattr(self, "drawer_joint", None) is not None
            and self.window_body
        )

    def _setup_scene(self):
        self.task_spec = self.laptop_task_spec
        HolodeckCloseLaptop._setup_scene(self)
        self._set_drawer_open_state()
        self.task_spec = self.laptop_task_spec

    def _reset_internal(self):
        self._laptop_layout_ready = False
        super()._reset_internal()
        self._set_drawer_open_state()
        self._laptop_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_laptop_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _laptop_closed(self):
        return bool(RoomOpenArticulation._raw_success(self))

    def _set_drawer_open_state(self):
        self.sim.data.set_joint_qpos(
            self.drawer_joint.name, float(self.drawer_task_spec["open_qpos"])
        )
        self.sim.data.set_joint_qvel(self.drawer_joint.name, 0.0)
        self.sim.forward()

    def _drawer_closed_after_laptop(self):
        return bool(
            self.drawer_joint.fraction(self.sim)
            <= float(self.drawer_task_spec["success_close_fraction"])
        )

    def _window_reached(self):
        return bool(
            HolodeckApproachWindow.success_distance(self)
            <= float(self.window_spec["success_radius"])
        )

    def _check_success(self):
        return bool(HolodeckMultiHopComposite._check_success(self))

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="close the laptop, close the specified drawer, then walk to the rightmost window",
            target_fixture=self.drawer_fixture.stable_id,
            target_object="specified drawer",
            target_window=self.window_body,
            success_definition=(
                "close the calibrated laptop first, close the specified drawer, "
                "then remain within the rightmost window's calibrated collision distance"
            ),
        )
        return meta


class HolodeckApproachTrashCanThenPickBoxFromFloor(
    _HolodeckBoxNavigationMixin,
    HolodeckPickBoxFromFloor,
    HolodeckMultiHopComposite,
):
    """Reach the trash can, then lift the calibrated cardboard box."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenPickBoxFromFloor"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = HolodeckPickBoxFromFloor.SUPPORTED_SCENE_IDS
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_box_navigation_stage",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
        HolodeckAtomicTask(
            "pick_box_from_floor",
            "_box_lifted",
            task_cls=HolodeckPickBoxFromFloor,
            hold_steps=None,
            on_enter="_begin_box_lift_stage",
            metadata={
                "skill": "lift",
                "object": "box",
                "source": "floor",
                "min_lift_m": 0.30,
            },
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.box_task_spec = self.task_spec
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_trash_can",
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self._box_navigation_radius = float(self.trash_spec["success_radius"])
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.trash_can_body
            ),
            None,
        )
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        pose = self.trash_spec.get(
            "composite_robot_pose", self.trash_spec["robot_pose"]
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _box_navigation_distance(self):
        return HolodeckApproachTrashCan.success_distance(self)

    def _trash_can_reached(self):
        return self._box_navigation_reached()

    def _setup_scene(self):
        super()._setup_scene()
        trash_distance = HolodeckApproachTrashCan.success_distance(self)
        if (
            not np.isfinite(trash_distance)
            or trash_distance < self._box_navigation_radius + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: trash-can composite spawn is too close "
                f"({trash_distance:.3f} m; needs >= "
                f"{self._box_navigation_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        self._box_layout_ready = False
        super()._reset_internal()
        self._move_box_next_to_trash_can()
        self._box_initial_z = float(self.object_pos("box")[2])
        self._box_layout_ready = True
        self._reset_box_composite()

    def _move_box_next_to_trash_can(self):
        """Move only this composite's box layout beside its trash can.

        The atomic pick task has already reset the box from its own calibrated
        ``pick_box_from_floor.box_pose``.  Derive a nearby pose from that
        position without changing the shared atomic specification.
        """
        box = self.objects["box"]
        box_qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        box_xy = np.asarray(box_qpos[:2], dtype=float)
        trash_xy = np.asarray(
            self.sim.data.get_body_xpos(self.trash_can_body), dtype=float
        )[:2]
        direction = box_xy - trash_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0])
        else:
            direction /= norm
        box_qpos[:2] = trash_xy + 0.95 * direction
        self.sim.data.set_joint_qpos(box.joints[0], box_qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the trash can, then pick up the cardboard box from the floor",
            target_object="box",
            success_definition=(
                "first reach the calibrated trash can, then raise the cardboard box "
                "more than 0.30 m above its floor reset height for 1 second"
            ),
        )
        return meta


class HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk(
    _HolodeckBoxNavigationMixin,
    HolodeckPlaceBoxOnDesk,
    HolodeckMultiHopComposite,
):
    """Reach the trash can, return to the desk, then place the floor box."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenPickBoxAndPlaceOnDesk"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    TRASH_FIRST_ROBOT_POSE_KEY = "trash_first_composite_robot_pose"
    MIN_DESK_APPROACH_DISPLACEMENT = 0.15
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_box_navigation_stage",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached_after_trash_can",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            on_enter="_begin_desk_stage_after_trash_can",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "pick_box_from_floor",
            "_box_picked",
            task_cls=HolodeckPickBoxFromFloor,
            hold_steps=None,
            metadata={"skill": "pick", "object": "box", "source": "floor"},
        ),
        HolodeckAtomicTask(
            "place_box_on_desk",
            "_box_placed_on_desk",
            task_cls=HolodeckPlaceBoxOnDesk,
            hold_steps=None,
            metadata={"skill": "place", "object": "box", "target": "desk"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.box_task_spec = self.task_spec
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_trash_can",
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self._box_navigation_radius = float(self.trash_spec["success_radius"])
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_desk",
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.navigation_desk_fixture = self.get_fixture(
            id=self.navigation_spec["desk_body"]
        )
        if self.navigation_desk_fixture is None:
            raise RuntimeError(
                f"{self.scene_name}: calibrated approach-desk body is missing: "
                f"{self.navigation_spec['desk_body']}"
            )
        if self.navigation_desk_fixture.body_name != self.desk_body:
            raise RuntimeError(
                f"{self.scene_name}: box placement and approach-desk targets differ"
            )
        self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
        self._desk_navigation_radius = float(self.navigation_spec["success_radius"])
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.trash_can_body
            ),
            None,
        )
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        # A trash-first workflow can require a different orientation from
        # composites whose first stage approaches the desk.
        pose = self.trash_spec.get(
            self.TRASH_FIRST_ROBOT_POSE_KEY,
            self.trash_spec.get(
                "desk_composite_robot_pose",
                self.trash_spec.get(
                    "composite_robot_pose", self.trash_spec["robot_pose"]
                ),
            ),
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        # The atomic place-box task is intentionally restricted to train_117
        # and validates its legacy desk body.  This composite uses the
        # scene-local desk selected by ``place_box_on_desk.table_body``.
        return self.desk_fixture is not None

    def _box_navigation_distance(self):
        return HolodeckApproachTrashCan.success_distance(self)

    def _trash_can_reached(self):
        return self._box_navigation_reached()

    def _begin_desk_stage_after_trash_can(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _desk_reached_after_trash_can(self):
        if (
            np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy)
            >= self.MIN_DESK_APPROACH_DISPLACEMENT
        ):
            self._desk_stage_moved = True
        return bool(
            self._desk_stage_moved
            and self.robot_fixture_collision_distance(
                self.navigation_desk_fixture,
                max_distance=self._desk_navigation_radius + 0.25,
            )
            <= self._desk_navigation_radius
        )

    def _setup_scene(self):
        super()._setup_scene()
        trash_distance = HolodeckApproachTrashCan.success_distance(self)
        if (
            not np.isfinite(trash_distance)
            or trash_distance < self._box_navigation_radius + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: trash-can composite spawn is too close "
                f"({trash_distance:.3f} m; needs >= "
                f"{self._box_navigation_radius + 0.15:.3f} m)"
            )
        desk_distance = self.robot_fixture_collision_distance(
            self.navigation_desk_fixture,
            max_distance=self._desk_navigation_radius + 1.0,
        )
        if (
            not np.isfinite(desk_distance)
            or desk_distance < self._desk_navigation_radius + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: desk composite spawn is too close "
                f"({desk_distance:.3f} m; needs >= "
                f"{self._desk_navigation_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        self._box_layout_ready = False
        super()._reset_internal()
        self._box_initial_z = float(self.object_pos("box")[2])
        self._box_layout_ready = True
        self._reset_box_composite()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        desk_fixture = getattr(self, "desk_fixture", None)
        meta.update(
            lang="walk to the trash can, walk to the desk, then pick up the cardboard box and place it on the desk",
            target_object="box",
            target_fixture=(
                desk_fixture.stable_id
                if desk_fixture is not None
                else HolodeckPlaceBoxOnDesk.DESK_BODY
            ),
            success_definition=(
                "first reach the calibrated trash can, then move at least "
                f"{self.MIN_DESK_APPROACH_DISPLACEMENT:.2f} m to the calibrated desk, "
                "then pick up the floor box and keep it in contact with the desk "
                "for 1 second"
            ),
        )
        return meta


class HolodeckCloseLaptopThenApproachTrashCanThenPickBox(
    _HolodeckBoxNavigationMixin,
    HolodeckCloseLaptop,
    HolodeckMultiHopComposite,
):
    """Close the laptop, reach the trash can, then lift the floor box."""

    ABSTRACT = False
    ATOMIC_SKILL = "CloseLaptopThenApproachTrashCanThenPickBox"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "close_laptop",
            "_laptop_closed",
            task_cls=HolodeckCloseLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_box_navigation_stage",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
        HolodeckAtomicTask(
            "pick_box_from_floor",
            "_box_lifted",
            task_cls=HolodeckPickBoxFromFloor,
            hold_steps=None,
            on_enter="_begin_box_lift_stage",
            metadata={"skill": "lift", "object": "box", "min_lift_m": 0.30},
        ),
    )

    def _setup_task_fixtures(self):
        # Initialize the shared Holodeck scene once, then bind each primitive
        # to its own calibration. Calling the atomic classes' ``super`` chain
        # here would pass the box spec into the laptop initializer.
        HolodeckMultiHopComposite._setup_task_fixtures(self)
        self.box_task_spec = get_holodeck_task_spec(
            self.scene_name, "pick_box_from_floor", required=("robot_pose", "box_pose")
        )
        self.laptop_task_spec = get_holodeck_task_spec(
            self.scene_name, "close_laptop", required=("robot_pose",)
        )
        composite_laptop_pose = self.laptop_task_spec.get("composite_robot_pose")
        if composite_laptop_pose is not None:
            self.laptop_task_spec["robot_pose"] = composite_laptop_pose
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_trash_can",
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self._box_navigation_radius = float(self.trash_spec["success_radius"])
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.trash_can_body
            ),
            None,
        )
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        laptop_spec = self.laptop_task_spec
        self.task_spec = laptop_spec
        if laptop_spec.get("fixture_mode") == "spawn_asset":
            # The atomic laptop initializer is not part of this composite's
            # manually controlled fixture setup.  Spawn the calibrated laptop
            # here so the later articulation setup sees the same body/joint
            # contract as the atomic task.
            self._spawn_laptop()
        candidates = self._semantic_candidates()
        self.target_fixture = next(
            (
                fixture
                for fixture in candidates
                if fixture.body_name == laptop_spec["target_body"]
            ),
            None,
        )
        if self.target_fixture is not None:
            self.fixture_refs["articulation_target"] = self.target_fixture
        pose = self.laptop_task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _box_navigation_distance(self):
        return HolodeckApproachTrashCan.success_distance(self)

    def _g1_target_z(self):
        """Use this composite's actual robot root as the joint-height reference."""
        root_body = self.robots[0].robot_model.root_body
        body_id = self.sim.model.body_name2id(root_body)
        return float(self.sim.data.body_xpos[body_id][2])

    def _trash_can_reached(self):
        return self._box_navigation_reached()

    def _create_objects(self):
        HolodeckMultiHopComposite._create_objects(self)
        xml_path = HolodeckPickBoxFromFloor._make_box_xml()
        try:
            box = MJCFObject(
                name="box",
                mjcf_path=xml_path,
                scale=float(self.box_task_spec["box_scale"]),
            )
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.objects[box.name] = box
        self.model.merge_objects([box])

    def _setup_scene(self):
        self.task_spec = self.laptop_task_spec
        HolodeckCloseLaptop._setup_scene(self)
        self.sim.forward()
        trash_distance = HolodeckApproachTrashCan.success_distance(self)
        if (
            not np.isfinite(trash_distance)
            or trash_distance < self._box_navigation_radius + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: trash-can composite spawn is too close "
                f"({trash_distance:.3f} m; needs >= {self._box_navigation_radius + 0.15:.3f} m)"
            )
        self.task_spec = self.box_task_spec

    def _reset_internal(self):
        self._box_layout_ready = False
        # Bypass the atomic reset methods' shared ``task_spec`` slot: laptop
        # and box each have distinct calibrated specifications.
        HolodeckMultiHopComposite._reset_internal(self)
        self.task_spec = self.box_task_spec
        HolodeckPlaceBoxOnDesk._set_box_pose(self)
        HolodeckApproachTrashCanThenPickBoxFromFloor._move_box_next_to_trash_can(self)
        if getattr(self, "target_joint", None) is not None:
            self.sim.data.set_joint_qpos(
                self.target_joint, float(self.laptop_task_spec.get("open_qpos", 0.0))
            )
            self.sim.data.set_joint_qvel(self.target_joint, 0.0)
        self.sim.forward()
        self._box_initial_z = float(self.object_pos("box")[2])
        self._box_layout_ready = True
        self._reset_box_composite()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    def _laptop_closed(self):
        return bool(RoomOpenArticulation._raw_success(self))

    def _check_success(self):
        # RoomOpenArticulation's predicate only covers the laptop stage. The
        # ordered composite state machine additionally requires navigation and lift.
        return bool(HolodeckMultiHopComposite._check_success(self))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="close the laptop, walk to the trash can, then pick up the cardboard box and lift it",
            target_object="box",
            success_definition=(
                "close the calibrated laptop, reach the calibrated trash can, "
                "then raise the cardboard box more than 0.30 m above reset height"
            ),
        )
        return meta


class HolodeckApproachWindowThenPickBox(
    _HolodeckBoxNavigationMixin,
    HolodeckPickBoxFromFloor,
    HolodeckMultiHopComposite,
):
    """Reach the rightmost window, then lift the calibrated floor box."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachWindowThenPickBox"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = HolodeckPickBoxFromFloor.SUPPORTED_SCENE_IDS
    BOX_SPEC_TASK = "approach_window_then_pick_box"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_rightmost_window",
            "_window_reached",
            task_cls=HolodeckApproachWindow,
            hold_steps=HolodeckApproachWindow.HOLD_STEPS,
            on_enter="_begin_box_navigation_stage",
            metadata={"skill": "navigation", "target": "rightmost_window"},
        ),
        HolodeckAtomicTask(
            "pick_box_from_floor",
            "_box_lifted",
            task_cls=HolodeckPickBoxFromFloor,
            hold_steps=None,
            on_enter="_begin_box_lift_stage",
            metadata={"skill": "lift", "object": "box", "min_lift_m": 0.30},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.box_task_spec = dict(self.task_spec)
        composite_box_spec = get_holodeck_task_spec(
            self.scene_name, self.BOX_SPEC_TASK, required=("box_pose",)
        )
        self.box_task_spec["box_pose"] = composite_box_spec["box_pose"]
        self._ground_composite_box_on_collision_geometry = bool(
            composite_box_spec.get("ground_box_on_collision_geometry", False)
        )
        self.window_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_window",
            required=("window_body", "robot_pose", "success_radius"),
        )
        self.window_body = self.window_spec["window_body"]
        self._box_navigation_radius = float(self.window_spec["success_radius"])
        pose = self.window_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _box_navigation_distance(self):
        return HolodeckApproachWindow.success_distance(self)

    def _window_reached(self):
        return self._box_navigation_reached()

    def _reset_internal(self):
        self._box_layout_ready = False
        super()._reset_internal()
        # The atomic reset above keeps its own calibration intact. Apply only
        # this composite's box pose after the atomic floor reset completes.
        self._set_box_pose(self.box_task_spec["box_pose"], task_spec=self.box_task_spec)
        if self._ground_composite_box_on_collision_geometry:
            self._ground_composite_box()
        self._box_layout_ready = True
        self._reset_box_composite()

    def _ground_composite_box(self):
        """Ground this layout from physical geoms, excluding its semantic bbox."""
        box = self.objects["box"]
        box_body_id = self.sim.model.body_name2id(box.root_body)
        box_body_ids = self._body_tree_ids([box_body_id])
        collision_geom_ids = self._collidable_geoms(box_body_ids)
        if not collision_geom_ids:
            raise RuntimeError(f"{self.scene_name}: box has no collision geometry")

        signs = np.asarray(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        min_z = np.inf
        for geom_id in collision_geom_ids:
            aabb = np.asarray(self.sim.model.geom_aabb[geom_id], dtype=float)
            rotation = np.asarray(
                self.sim.data.geom_xmat[geom_id], dtype=float
            ).reshape(3, 3)
            center = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
            geom_min_z = min(
                float((center + rotation @ (aabb[:3] + aabb[3:] * sign))[2])
                for sign in signs
            )
            min_z = min(min_z, geom_min_z)

        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        qpos[2] -= float(min_z) + 0.001
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        self._box_initial_z = self._box_root_z()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the rightmost window, then pick up the cardboard box from the floor and lift it",
            target_object="box",
            target_window=self.window_body,
            success_definition="first reach the rightmost window, then raise the cardboard box more than 0.30 m above reset height",
        )
        return meta


class HolodeckPullChairThenApproachTrashCanThenPickBox(
    _HolodeckBoxNavigationMixin,
    HolodeckPickBoxFromFloor,
    HolodeckPullChairFromDesk,
    HolodeckMultiHopComposite,
):
    """Pull out the desk chair, reach the trash can, then lift the box."""

    ABSTRACT = False
    ATOMIC_SKILL = "PullChairThenApproachTrashCanThenPickBox"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "pull_chair_from_desk",
            "_chair_pulled",
            task_cls=HolodeckPullChairFromDesk,
            hold_steps=None,
            on_enter="_begin_pull_chair_stage",
            metadata={"skill": "pull", "object": "chair"},
        ),
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_box_navigation_stage",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
        HolodeckAtomicTask(
            "pick_box_from_floor",
            "_box_lifted",
            task_cls=HolodeckPickBoxFromFloor,
            hold_steps=None,
            on_enter="_begin_box_lift_stage",
            metadata={
                "skill": "lift",
                "object": "box",
                "source": "floor",
                "min_lift_m": 0.30,
            },
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.box_task_spec = self.task_spec
        self.chair_task_spec = get_holodeck_task_spec(
            self.scene_name,
            "pull_chair_from_desk",
            required=(
                "chair_body",
                "chair_joint",
                "chair_start_pose",
                "target_xy",
                "target_tolerance",
            ),
        )
        self._pull_origin_xy = np.asarray(
            self.chair_task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        self.pull_chair_body = self.chair_task_spec["chair_body"]
        self.pull_chair_joint = self.chair_task_spec["chair_joint"]
        self.pull_target_xy = np.asarray(self.chair_task_spec["target_xy"], dtype=float)
        self.pull_target_tolerance = np.asarray(
            self.chair_task_spec["target_tolerance"], dtype=float
        )
        self.chair_body = self.pull_chair_body
        self.chair_joint = self.pull_chair_joint
        self.target_xy = self.pull_target_xy
        self.target_tolerance = self.pull_target_tolerance
        pull_pose = self.chair_task_spec.get(
            "composite_robot_pose", self.chair_task_spec.get("robot_pose")
        )
        if pull_pose is not None:
            self.set_robot_base_pose(
                np.asarray(pull_pose["pos"], dtype=float), float(pull_pose["yaw"])
            )
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_trash_can",
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self._box_navigation_radius = float(self.trash_spec["success_radius"])

    def _box_navigation_distance(self):
        return HolodeckApproachTrashCan.success_distance(self)

    def _begin_pull_chair_stage(self):
        body_id = self.sim.model.body_name2id(self.pull_chair_body)
        self._chair_pull_start_xy = np.asarray(
            self.sim.data.body_xpos[body_id][:2], dtype=float
        ).copy()

    def _chair_pulled(self):
        return HolodeckPullChairFromDesk._chair_has_min_pull_displacement(
            self,
            chair_body=self.pull_chair_body,
            origin_xy=self._pull_origin_xy,
        )

    def _chair_in_target_region(self):
        body_id = self.sim.model.body_name2id(self.pull_chair_body)
        chair_xy = np.asarray(self.sim.data.body_xpos[body_id][:2], dtype=float)
        return bool(
            np.all(np.abs(chair_xy - self.pull_target_xy) <= self.pull_target_tolerance)
        )

    def _trash_can_reached(self):
        return self._box_navigation_reached()

    def _setup_scene(self):
        # Run the pull-chair scene checks with the chair calibration. Calling
        # ``super`` here would continue into PickBoxFromFloor, which expects a
        # box ``room_id`` while this temporary spec is the chair spec.
        saved = (
            self.chair_body,
            self.chair_joint,
            self.target_xy,
            self.target_tolerance,
        )
        self.chair_body = self.pull_chair_body
        self.chair_joint = self.pull_chair_joint
        self.target_xy = self.pull_target_xy
        self.target_tolerance = self.pull_target_tolerance
        self.task_spec = self.chair_task_spec
        try:
            HolodeckPullChairFromDesk._setup_scene(self)
        finally:
            (
                self.chair_body,
                self.chair_joint,
                self.target_xy,
                self.target_tolerance,
            ) = saved

        # Apply the floor-box checks after restoring the box calibration. This
        # keeps both inherited initialization contracts valid in one scene.
        self.task_spec = self.box_task_spec
        expected_room = self._normalized_room_id(self.task_spec["room_id"])
        robot_room = self.room_of_point([*self._robot_base_xy(), 0.0])
        if self._normalized_room_id(robot_room) != expected_room:
            raise RuntimeError(
                f"{self.scene_name}: robot is in room {robot_room}, "
                f"expected room {expected_room}"
            )
        if not self._robot_footprint_is_in_room(radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: box floor task robot starts off floor"
            )
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: box floor task robot has spawn collision"
            )

    def _reset_internal(self):
        self._box_layout_ready = False
        self._pull_origin_xy = np.asarray(
            self.chair_task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        saved = (
            self.chair_body,
            self.chair_joint,
            self.target_xy,
            self.target_tolerance,
        )
        self.chair_body = self.pull_chair_body
        self.chair_joint = self.pull_chair_joint
        self.target_xy = self.pull_target_xy
        self.target_tolerance = self.pull_target_tolerance
        HolodeckMultiHopComposite._reset_internal(self)
        HolodeckPushChairToDesk._set_initial_chair_state(self, self.chair_task_spec)
        self.task_spec = self.box_task_spec
        # Reuse the calibrated floor placement, including compiled-geometry
        # grounding, while keeping the box task local to this composite.
        HolodeckPlaceBoxOnDesk._set_box_pose(self)
        # Match the trash-can composite's local box placement without
        # changing the shared atomic ``pick_box_from_floor`` calibration.
        HolodeckApproachTrashCanThenPickBoxFromFloor._move_box_next_to_trash_can(self)
        self.sim.forward()
        self.chair_body, self.chair_joint, self.target_xy, self.target_tolerance = saved
        self._box_initial_z = float(self.object_pos("box")[2])
        self._box_layout_ready = True
        self._reset_box_composite()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="pull the chair out from under the desk, walk to the trash can, then pick up and lift the cardboard box",
            target_object="box",
            success_definition=(
                f"pull the chair at least {HolodeckPullChairFromDesk.MIN_PULL_DISPLACEMENT:.2f} m "
                "from its reset position and keep it there for 1 second, reach "
                "the trash can, then raise the floor box more than 0.30 m above "
                "its reset height and keep holding it for 1 second"
            ),
        )
        return meta


# Short aliases keep the action-oriented names convenient for task catalogs.
HolodeckApproachWindowThenPickBoxFromFloor = HolodeckApproachWindowThenPickBox
HolodeckCloseLaptopThenApproachTrashCanThenPickBoxFromFloor = (
    HolodeckCloseLaptopThenApproachTrashCanThenPickBox
)


class HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop(
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
):
    """Remove a shelf folder, place it on the desk, and close the laptop."""

    ABSTRACT = False
    ATOMIC_SKILL = "RemoveFolderToDeskThenCloseLaptop"
    DIFFICULTY_LEVEL = "L4"
    LAPTOP_SPEC_TASK = "close_laptop"
    BEHAVIOR = "close"
    OPEN_INIT_FRACTION = 1.0
    SUCCESS_CLOSE_FRACTION = 0.05
    ATOMIC_TASKS = (
        *HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop.ATOMIC_TASKS[:2],
        HolodeckAtomicTask(
            "close_laptop",
            "_laptop_closed_after_folder",
            task_cls=HolodeckCloseLaptop,
            hold_steps=None,
            on_enter="_begin_laptop_close_stage",
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
    )

    def _configure_articulation_task(self):
        # The close-laptop spec stores endpoints but, unlike drawer specs, does
        # not need a separate fraction field.  The atomic laptop task uses the
        # same calibrated five-percent closed threshold.
        self.OPEN_INIT_FRACTION = 1.0
        self.SUCCESS_CLOSE_FRACTION = 0.05
        self.HOLD_STEPS = RoomOpenArticulation.HOLD_STEPS

    def _set_laptop_initial_state(self):
        self.sim.data.set_joint_qpos(
            self.target_joint, self.target_semantic_joint.open_qpos
        )
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)

    def _laptop_closed_after_folder(self):
        return bool(
            getattr(self, "_laptop_closed_after_stage", False)
            and not getattr(self, "_invalid_early_laptop_close", False)
            and RoomOpenArticulation._raw_success(self)
        )

    def _begin_laptop_close_stage(self):
        self._success_hold = 0
        self._laptop_closed_after_stage = False
        self._invalid_early_laptop_close = RoomOpenArticulation._raw_success(self)

    def _post_action(self, action):
        ret = super()._post_action(action)
        if (
            self.current_atomic_task is not None
            and self.current_atomic_task.name == "close_laptop"
            and RoomOpenArticulation._raw_success(self)
        ):
            self._laptop_closed_after_stage = True
        return ret

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "remove the folder completely from the shelf, place it on the "
                "desk, then close the laptop"
            ),
            success_definition=(
                "first clear the complete folder bounding box from the shelf and "
                "place it on the desk, then close the calibrated laptop lid"
            ),
        )
        return meta


class HolodeckPickBookFromSourceToDeskThenApproachTrashCan(
    HolodeckPickBookFromSourceToDesk
):
    """Place the calibrated book on the desk, then walk to the trash can."""

    ABSTRACT = False
    ATOMIC_SKILL = "PickBookFromSourceToDeskThenApproachTrashCan"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    TRASH_SPEC_TASK = "approach_trash_can"
    ATOMIC_TASKS = (
        *HolodeckPickBookFromSourceToDesk.ATOMIC_TASKS,
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached_after_book",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_trash_stage_after_book",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            self.TRASH_SPEC_TASK,
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.trash_can_body
            ),
            None,
        )
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        # The inherited book stages must start beside the calibrated source.
        source_pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(source_pose["pos"], dtype=float), float(source_pose["yaw"])
        )

    def _setup_scene(self):
        super()._setup_scene()
        distance = HolodeckApproachTrashCan.success_distance(self)
        if distance < self.trash_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: book composite spawn is too close to trash can "
                f"({distance:.3f} m; needs >= {self.trash_success_radius + 0.15:.3f} m)"
            )

    @property
    def trash_success_radius(self):
        return float(self.trash_spec["success_radius"])

    def _begin_trash_stage_after_book(self):
        self._trash_stage_start_xy = self._robot_base_xy().copy()
        self._trash_stage_moved = False

    def _trash_can_reached_after_book(self):
        if np.linalg.norm(self._robot_base_xy() - self._trash_stage_start_xy) >= 0.15:
            self._trash_stage_moved = True
        return bool(
            self._trash_stage_moved
            and HolodeckApproachTrashCan.success_distance(self)
            <= self.trash_success_radius
        )

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="pick up the book, place it on the desk, then walk to the trash can",
            success_definition=(
                "first complete the calibrated book pick-and-place on the desk, "
                "then move at least 0.15 m and remain within the trash-can distance"
            ),
        )
        return meta


class HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan(
    _HolodeckDrawerTissueBoxMixin,
    RoomOpenArticulation,
    HolodeckMultiHopComposite,
):
    """Remove the tissue box, close its drawer, then walk to the trash can."""

    ABSTRACT = False
    ATOMIC_SKILL = "TakeTissueBoxThenCloseDrawerThenApproachTrashCan"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Dresser",)
    TARGET_LANG = "dresser drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUCCESS_JOINT_SELECTION = "target"
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)
    SUPPORTED_SCENE_IDS = (117, 1018)
    SPEC_TASK = "take_tissue_box_from_open_drawer"
    CLOSE_DRAWER_SPEC_TASK = "close_drawer"
    TRASH_SPEC_TASK = "approach_trash_can"
    OPEN_STAND_DISTANCE = 0.35
    PLACEMENT_HOLD_STEPS = 5
    COMPOSITE_STAGES = (
        "grasp and remove the tissue box from the initially open drawer",
        "place the tissue box fully on the calibrated desk",
        "close the drawer while the tissue box remains on the desk",
        "move to and hold at the calibrated trash can",
    )
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "take_tissue_box",
            "_tissue_box_removed_stage",
            task_cls=HolodeckTakeTissueBoxFromOpenDrawer,
            hold_steps=None,
            metadata={"skill": "pick", "object": "tissue_box"},
        ),
        HolodeckAtomicTask(
            "place_tissue_box_on_desk",
            "_tissue_box_released_on_desk",
            task_cls=HolodeckTakeTissueBoxFromOpenDrawer,
            hold_steps=PLACEMENT_HOLD_STEPS,
            on_enter="_begin_tissue_box_placement",
            metadata={"skill": "place", "object": "tissue_box", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "close_drawer",
            "_drawer_closed_after_tissue_box",
            task_cls=HolodeckCloseDrawer,
            hold_steps=None,
            metadata={"skill": "articulation", "target": "drawer"},
        ),
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached_after_drawer",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_trash_stage_after_drawer",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.tissue_task_spec = self.task_spec
        self.close_drawer_spec = get_holodeck_task_spec(
            self.scene_name,
            self.CLOSE_DRAWER_SPEC_TASK,
            required=("success_close_fraction",),
        )
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            self.TRASH_SPEC_TASK,
            required=("trash_can_body", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.trash_can_body
            ),
            None,
        )
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        self.task_spec = self.tissue_task_spec

    def _tissue_box_removed_stage(self):
        return bool(
            getattr(self, "_tissue_box_removed_after_open_state", False)
            and self._drawer_open_enough(self.target_semantic_joint.fraction(self.sim))
        )

    def _begin_tissue_box_placement(self):
        self._tissue_box_removed_before_placement = self._tissue_box_removed_stage()

    def _tissue_box_released_on_desk(self):
        return bool(
            getattr(self, "_tissue_box_removed_before_placement", False)
            and self.object_on_region(
                "tissue_box",
                self.destination_region,
                z_tol=0.05,
                require_contact=True,
            )
        )

    def _tissue_box_still_placed_on_desk(self):
        return bool(
            self.object_on_region(
                "tissue_box",
                self.destination_region,
                z_tol=0.05,
                require_contact=True,
            )
        )

    def _drawer_closed_after_tissue_box(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            <= float(self.close_drawer_spec["success_close_fraction"])
        )

    def _begin_trash_stage_after_drawer(self):
        self._trash_stage_start_xy = self._robot_base_xy().copy()
        self._trash_stage_moved = False

    def _trash_can_reached_after_drawer(self):
        if np.linalg.norm(self._robot_base_xy() - self._trash_stage_start_xy) >= 0.15:
            self._trash_stage_moved = True
        return bool(
            self._trash_stage_moved
            and HolodeckApproachTrashCan.success_distance(self)
            <= self.trash_success_radius
        )

    @property
    def trash_success_radius(self):
        return float(self.trash_spec["success_radius"])

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _reset_internal(self):
        self._drawer_layout_ready = False
        super()._reset_internal()
        self._tissue_box_removed_before_placement = False
        self._tissue_box_placed_before_close = False
        self._drawer_open_before_close = False
        self._drawer_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_drawer_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the tissue box out, place it on the desk, close the drawer, "
                "then walk to the trash can"
            ),
            success_definition=(
                "first lift the tissue box after the drawer is open more "
                f"than {self.MIN_LIFT_DELTA:.2f} m above its reset height, and "
                "place it fully on the calibrated desk, close that "
                "drawer while the box remains on the desk, then move to and hold "
                "at the calibrated trash can"
            ),
        )
        return meta


class HolodeckApproachDeskThenIceCubeInGlass(
    HolodeckIceCubeInGlass,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, then put the calibrated ice cube into the cup."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenIceCubeInGlass"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    NAVIGATION_SPEC_TASK = "approach_desk"
    # The original approach-desk targets carry this baked laptop. Remove it
    # only for this workflow so the cup and ice cube have usable tabletop
    # space; later scenes declare their cleanup in the atomic YAML spec.
    CALIBRATED_LAPTOP_BODIES = {
        "train_117": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
        "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
    }
    COMPOSITE_LAYOUTS = {
        "train_117": {
            "table_body": "desk_aeb6ab35aea76da67db0d3cd45d3e821_1_0_0",
            "cup_pose": {
                "pos": [5.45, 4.8, 0.7773],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "ice_pose": {
                "pos": [5.45, 4.45, 0.72108],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_1018": {
            "table_body": "desk_0ecf411cb557754bf283627e911b16c6_1_0_0",
            "cup_pose": {
                "pos": [7.55, 1.02, 0.8009],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "ice_pose": {
                "pos": [7.55, 0.68, 0.7248],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_1197": {
            "table_body": "obja_45dadf1d294c461195e92c1cdeb8aabf_1_0_0",
            "robot_pose": {
                "pos": [3.5, 2.1, 0.0],
                "yaw": 0.0,
            },
            "cup_pose": {
                "pos": [5.2822904116, 2.225, 0.8061],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "ice_pose": {
                "pos": [5.3, 2.1, 0.7501],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_1664": {
            "table_body": "obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0",
            "robot_pose": {
                "pos": [3.5, 1.8, 0.0],
                "yaw": 0.0,
            },
            "cup_pose": {
                "pos": [5.6822904116, 1.575, 0.8257],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "ice_pose": {
                "pos": [5.7, 1.45, 0.7697],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_2025": {
            "table_body": "obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0",
            "robot_pose": {
                "pos": [3.5, 1.95, 0.0],
                "yaw": 0.06627822076846923,
            },
            "cup_pose": {
                "pos": [5.6222904116, 1.625, 0.8459],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "ice_pose": {
                "pos": [5.64, 1.5, 0.7899],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
    }
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "put_ice_in_cup",
            "_ice_inserted_after_approach",
            task_cls=HolodeckIceCubeInGlass,
            hold_steps=5,
            on_enter="_begin_ice_stage_after_approach",
            metadata={"skill": "pick_place", "object": "ice_cube", "target": "cup"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        laptop_bodies = self.CALIBRATED_LAPTOP_BODIES.get(self.scene_name, ())
        if laptop_bodies:
            removed = self.remove_scene_bodies(laptop_bodies)
            missing = set(laptop_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured task laptop bodies are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.ice_task_spec = dict(self.task_spec)
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        # The requested desk is the desk/table on which the calibrated cup
        # and ice cube are placed, while the shared navigation pose supplies a
        # remote, collision-free initial position.
        layout = self.COMPOSITE_LAYOUTS[self.scene_name]
        if layout["table_body"] != self.navigation_spec["desk_body"]:
            raise RuntimeError(
                f"{self.scene_name}: ice composite layout must use the "
                "approach_desk target desk"
            )
        self.ice_task_spec.update(layout)
        self.table_fixture = self.get_fixture(id=self.ice_task_spec["table_body"])
        if self.table_fixture is not None:
            self.fixture_refs["table"] = self.table_fixture
        self.desk_fixture = self.get_fixture(id=self.navigation_spec["desk_body"])
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
            pose = layout.get("robot_pose", self.navigation_spec["robot_pose"])
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )
        self.task_spec = self.ice_task_spec

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.desk_fixture is not None)

    def _setup_scene(self):
        super()._setup_scene()
        distance = self.robot_fixture_collision_distance(
            self.desk_fixture, max_distance=self.success_radius + 1.0
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: ice composite spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _begin_ice_stage_after_approach(self):
        pass

    def _ice_inserted_after_approach(self):
        return bool(HolodeckIceCubeInGlass._ice_in_cup_region(self))

    def _post_action(self, action):
        return super()._post_action(action)

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.desk_fixture, max_distance=self.success_radius + 0.25
            )
            <= self.success_radius
        )

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then put the ice cube into the glass cup",
            target_fixture=self.desk_fixture.stable_id,
            success_definition=(
                "first reach the calibrated cup desk, then keep the ice cube inside "
                "the calibrated glass cup"
            ),
        )
        return meta


class HolodeckCloseJarThenApproachDeskThenApproachTrashCan(
    HolodeckCloseMugLid,
    HolodeckMultiHopComposite,
):
    """Close the jar lid, walk to the desk, then walk to the trash can."""

    ABSTRACT = False
    ATOMIC_SKILL = "CloseJarThenApproachDeskThenApproachTrashCan"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1664)
    DESK_SPEC_TASK = "approach_desk"
    TRASH_SPEC_TASK = "approach_trash_can"

    def __init__(self, *args, scene_ids=None, **kwargs):
        """Reject scene selections outside this composite's calibrated scope."""
        allowed = {str(scene_id) for scene_id in self.SUPPORTED_SCENE_IDS}
        requested = (
            self.SUPPORTED_SCENE_IDS
            if scene_ids is None
            else (
                (scene_ids,) if isinstance(scene_ids, (int, str)) else tuple(scene_ids)
            )
        )
        unsupported = [
            scene_id
            for scene_id in requested
            if normalize_holodeck_scene_name(scene_id).removeprefix("train_")
            not in allowed
        ]
        if unsupported:
            raise ValueError(
                f"{self.__class__.__name__} does not support scene ids: {unsupported}; "
                f"supported: {self.SUPPORTED_SCENE_IDS}"
            )
        super().__init__(
            *args,
            scene_ids=self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids,
            **kwargs,
        )

    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "close_jar",
            "_jar_closed",
            task_cls=HolodeckCloseMugLid,
            hold_steps=None,
            metadata={"skill": "pick_place", "object": "jar_lid", "target": "jar"},
        ),
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached_after_close",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            on_enter="_begin_desk_stage_after_close",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached_after_desk",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_trash_stage_after_desk",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        # The atomic parent loads ``close_mug_lid``. This composite may have a
        # scene-specific layout, so prefer its explicit calibration when present.
        try:
            self.jar_task_spec = get_holodeck_task_spec(
                self.scene_name,
                "close_jar_then_approach_desk_then_approach_trash_can",
            )
        except KeyError:
            self.jar_task_spec = self.task_spec
        self.desk_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DESK_SPEC_TASK,
            required=("desk_body", "success_radius"),
        )
        desk_body = self.jar_task_spec.get("desk_body", self.desk_spec["desk_body"])
        self.desk_fixture = self.get_fixture(id=desk_body)
        if self.desk_fixture is None:
            self.desk_fixture = self.get_fixture(id=desk_body, return_all=False)
        # Keep the inherited mug-lid validity and metadata fields aligned with
        # the composite's relocated table as well.
        self.table_body = self.jar_task_spec.get("table_body", desk_body)
        self.table_fixture = self.desk_fixture
        self._has_calibrated_table_body = (
            self.find_scene_body(self.table_body) is not None
        )
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            self.TRASH_SPEC_TASK,
            required=("trash_can_body", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
        composite_pose = self.jar_task_spec.get(
            "composite_robot_pose", self.jar_task_spec.get("robot_pose")
        )
        if composite_pose is not None:
            self.set_robot_base_pose(
                np.asarray(composite_pose["pos"], dtype=float),
                float(composite_pose["yaw"]),
            )
        self.task_spec = self.jar_task_spec

    def _setup_scene(self):
        super()._setup_scene()
        trash_distance = HolodeckApproachTrashCan.success_distance(self)
        if trash_distance < self.trash_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: jar composite spawn is too close to trash can "
                f"({trash_distance:.3f} m; needs >= {self.trash_success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        self._jar_layout_ready = False
        super()._reset_internal()
        robot_body_ids = set(
            self._body_tree_ids(
                [self.sim.model.body_name2id(self.robots[0].robot_model.root_body)]
            )
        )
        desk_body_ids = set(
            self._body_tree_ids(
                [
                    self.sim.model.body_name2id(body_name)
                    for body_name in self.desk_fixture.body_names
                    if body_name in self.sim.model.body_names
                ]
            )
        )
        penetrations = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_pair = (
                int(self.sim.model.geom_bodyid[contact.geom1]),
                int(self.sim.model.geom_bodyid[contact.geom2]),
            )
            if (
                (body_pair[0] in robot_body_ids and body_pair[1] in desk_body_ids)
                or (body_pair[1] in robot_body_ids and body_pair[0] in desk_body_ids)
            ) and float(contact.dist) < -1e-4:
                penetrations.append(float(contact.dist))
        if penetrations:
            raise RuntimeError(
                f"{self.scene_name}: jar composite robot spawn penetrates the desk "
                f"({len(penetrations)} contact(s); deepest={min(penetrations):.4f} m)"
            )
        self._jar_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_jar_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    @property
    def desk_success_radius(self):
        return float(self.desk_spec["success_radius"])

    @property
    def trash_success_radius(self):
        return float(self.trash_spec["success_radius"])

    def _jar_closed(self):
        return bool(self._lid_is_closed())

    def _begin_desk_stage_after_close(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _desk_reached_after_close(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy) >= 0.15:
            self._desk_stage_moved = True
        return bool(
            self._desk_stage_moved
            and self.robot_fixture_collision_distance(
                self.desk_fixture, max_distance=self.desk_success_radius + 0.25
            )
            <= self.desk_success_radius
        )

    def _begin_trash_stage_after_desk(self):
        self._trash_stage_start_xy = self._robot_base_xy().copy()
        self._trash_stage_moved = False

    def _trash_can_reached_after_desk(self):
        if np.linalg.norm(self._robot_base_xy() - self._trash_stage_start_xy) >= 0.15:
            self._trash_stage_moved = True
        return bool(
            self._trash_stage_moved
            and HolodeckApproachTrashCan.success_distance(self)
            <= self.trash_success_radius
        )

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="close the jar lid, walk to the desk, then walk to the trash can",
            success_definition=(
                "first keep the lid in the calibrated closed pose, then reach the "
                "desk and finally reach the trash can in order"
            ),
        )
        return meta


class HolodeckApproachTrashCanThenApproachDeskThenOpenJar(
    HolodeckOpenMugLid,
    HolodeckMultiHopComposite,
):
    """Reach the trash can, return to the jar desk, then open the jar lid."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenApproachDeskThenOpenJar"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018)
    TRASH_SPEC_TASK = "approach_trash_can"
    # Scene 1018 has separate calibrated desks for the standalone jar task
    # and the approach-desk stage. This composite must place the jar on the
    # desk that the navigation stage targets.
    COMPOSITE_LAYOUTS = {
        "train_1018": {
            "table_body": "desk_0ecf411cb557754bf283627e911b16c6_1_0_0",
            "mug_pose": {
                "pos": [7.55, 1.02, 0.7983895],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "lid_pose": {
                "pos": [7.55, 1.02, 0.8944795],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
    }
    # In train_1018 the calibrated jar desk also carries a baked laptop at
    # the jar pose. Remove it only for this workflow to keep the jar clear.
    CALIBRATED_LAPTOP_BODIES = {
        "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
    }
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
        HolodeckAtomicTask(
            "approach_desk",
            "_jar_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            on_enter="_begin_jar_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "open_jar",
            "_jar_opened",
            task_cls=HolodeckOpenMugLid,
            hold_steps=None,
            on_enter="_begin_open_jar_stage",
            metadata={"skill": "pick_place", "object": "jar_lid", "target": "jar"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        laptop_bodies = self.CALIBRATED_LAPTOP_BODIES.get(self.scene_name, ())
        if laptop_bodies:
            removed = self.remove_scene_bodies(laptop_bodies)
            missing = set(laptop_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured task laptop bodies are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.jar_task_spec = dict(self.task_spec)
        desk_spec = get_holodeck_task_spec(
            self.scene_name,
            "approach_desk",
            required=("desk_body", "robot_pose", "success_radius"),
        )
        layout = self.COMPOSITE_LAYOUTS.get(self.scene_name)
        if layout is not None:
            if layout["table_body"] != desk_spec["desk_body"]:
                raise RuntimeError(
                    f"{self.scene_name}: jar composite layout must use the "
                    "approach_desk target desk"
                )
            self.jar_task_spec.update(layout)
        elif self.jar_task_spec["table_body"] != desk_spec["desk_body"]:
            raise RuntimeError(
                f"{self.scene_name}: jar composite must use the approach_desk target desk"
            )
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            self.TRASH_SPEC_TASK,
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        self.trash_can_body = self.trash_spec["trash_can_body"]
        self.jar_desk_fixture = self.get_fixture(id=self.jar_task_spec["table_body"])
        if self.jar_desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.jar_desk_fixture
        # Composite workflows may need a remote spawn while the atomic trash
        # task keeps its own calibrated start pose.
        pose = self.trash_spec.get(
            "composite_robot_pose", self.trash_spec["robot_pose"]
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )
        self.task_spec = self.jar_task_spec

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.jar_desk_fixture is not None)

    def _setup_scene(self):
        super()._setup_scene()
        distance = HolodeckApproachTrashCan.success_distance(self)
        if distance < self.trash_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: jar-open composite spawn is too close to trash can "
                f"({distance:.3f} m; needs >= {self.trash_success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        self._jar_layout_ready = False
        super()._reset_internal()
        self._jar_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_jar_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    @property
    def trash_success_radius(self):
        return float(self.trash_spec["success_radius"])

    @property
    def desk_success_radius(self):
        return 1.0

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _trash_can_reached(self):
        return bool(
            HolodeckApproachTrashCan.success_distance(self) <= self.trash_success_radius
        )

    def _begin_jar_desk_stage(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _jar_desk_reached(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy) >= 0.15:
            self._desk_stage_moved = True
        return bool(
            self._desk_stage_moved
            and self.robot_fixture_collision_distance(
                self.jar_desk_fixture, max_distance=self.desk_success_radius + 0.25
            )
            <= self.desk_success_radius
        )

    def _jar_opened(self):
        return bool(self._lid_is_open())

    def _begin_open_jar_stage(self):
        return None

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the trash can, walk to the desk, then open the jar lid",
            target_fixture=self.jar_desk_fixture.stable_id,
            success_definition=(
                "first reach the calibrated trash can, return to the jar's desk, "
                "then keep the lid in the calibrated open position"
            ),
        )
        return meta


class _HolodeckOneSecondStageHoldMixin:
    """Require every stage in these ordered workflows to hold for 1 s."""

    STAGE_HOLD_SECONDS = 1.0

    def _required_stage_hold_steps(self, task):
        inherited = super()._required_stage_hold_steps(task)
        return max(
            inherited,
            max(1, int(round(float(self.STAGE_HOLD_SECONDS) * self.control_freq))),
        )


class HolodeckIceCubeThenApproachWindowThenPickBox(
    _HolodeckOneSecondStageHoldMixin,
    HolodeckApproachWindowThenPickBox,
):
    """Put ice in a cup, visit the rightmost window, then lift a floor box."""

    ABSTRACT = False
    ATOMIC_SKILL = "IceCubeThenApproachWindowThenPickBox"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1664, 2025)
    ICE_SPEC_TASK = "ice_cube_then_approach_window_then_pick_box"
    USE_DEFAULT_ASSET_LAYOUT = False
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "put_ice_in_cup",
            "_ice_inserted",
            task_cls=HolodeckIceCubeInGlass,
            metadata={"skill": "pick_place", "object": "ice_cube", "target": "cup"},
        ),
        HolodeckAtomicTask(
            "approach_rightmost_window",
            "_window_reached",
            task_cls=HolodeckApproachWindow,
            on_enter="_begin_box_navigation_stage",
            metadata={"skill": "navigation", "target": "rightmost_window"},
        ),
        HolodeckAtomicTask(
            "pick_box_from_floor",
            "_box_lifted",
            task_cls=HolodeckPickBoxFromFloor,
            on_enter="_begin_box_lift_stage",
            metadata={"skill": "lift", "object": "box", "min_lift_m": 0.30},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.box_task_spec = dict(self.task_spec)
        self.ice_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.ICE_SPEC_TASK,
            required=(
                "table_body",
                "object_scale",
                "cup_pose",
                "ice_pose",
                "box_pose",
            ),
        )
        self.box_task_spec["box_pose"] = self.ice_task_spec["box_pose"]
        self._ground_composite_box_on_collision_geometry = bool(
            self.ice_task_spec.get("ground_box_on_collision_geometry", False)
        )
        self.ice_table_fixture = self.get_fixture(id=self.ice_task_spec["table_body"])
        if self.ice_table_fixture is not None:
            self.fixture_refs["ice_table"] = self.ice_table_fixture
            pose = self.ice_task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )
        self.task_spec = self.box_task_spec

    def _setup_baked_scene_fixtures(self):
        spec = get_holodeck_task_spec(self.scene_name, self.ICE_SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured ice composite bodies are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.ice_table_fixture is not None)

    def _create_objects(self):
        super()._create_objects()
        root = robocasa.models.assets_root
        for name, asset in (
            ("cup", HolodeckIceCubeInGlass.CUP_ASSET),
            ("ice_cube", HolodeckIceCubeInGlass.ICE_ASSET),
        ):
            obj = MJCFObject(
                name=name,
                mjcf_path=os.path.join(root, asset),
                scale=float(self.ice_task_spec["object_scale"]),
            )
            self.objects[name] = obj
            self.model.merge_objects([obj])

    def _place_ice_object(self, name, pose):
        obj = self.objects[name]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))

    def _reset_internal(self):
        self._ice_layout_ready = False
        super()._reset_internal()
        self._place_ice_object("cup", self.ice_task_spec["cup_pose"])
        self._place_ice_object("ice_cube", self.ice_task_spec["ice_pose"])
        self.sim.forward()
        self._ice_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if getattr(self, "_ice_layout_ready", False):
            super()._validate_initial_composite_state()

    def _ice_inserted(self):
        return bool(HolodeckIceCubeInGlass._ice_in_cup_region(self))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "put the ice cube in the cup, walk to the rightmost window, "
                "then pick up the cardboard box from the floor"
            ),
            success_definition=(
                "keep the ice cube in the cup for 1 second, then remain near "
                "the rightmost window for 1 second, then lift the box more than "
                "0.30 m for 1 second"
            ),
        )
        return meta


class HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer(
    _HolodeckOneSecondStageHoldMixin,
    HolodeckCloseDrawer,
    HolodeckMultiHopComposite,
):
    """Reach the desk, place its box on the floor, then close the drawer."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskPlaceBoxOnFloorThenCloseDrawer"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1197)
    BOX_SPEC_TASK = "place_box_on_floor"
    NAVIGATION_SPEC_TASK = "approach_desk"
    USE_DEFAULT_ASSET_LAYOUT = False
    # Reuse the floor-contact tolerance required by the delegated box predicate.
    CONTACT_MARGIN = HolodeckPlaceBoxOnFloor.CONTACT_MARGIN
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "place_box_on_floor",
            "_box_on_floor",
            task_cls=HolodeckPlaceBoxOnFloor,
            metadata={"skill": "place", "object": "box", "target": "floor"},
        ),
        HolodeckAtomicTask(
            "close_drawer",
            "_drawer_closed",
            task_cls=HolodeckCloseDrawer,
            metadata={"skill": "articulation", "target": "drawer"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        spec = get_holodeck_task_spec(self.scene_name, self.BOX_SPEC_TASK)
        scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        removed = self.remove_scene_bodies(scene_bodies)
        missing = set(spec.get("remove_scene_bodies", ())) - removed
        if missing:
            raise RuntimeError(
                f"{self.scene_name}: missing box blockers: {sorted(missing)}"
            )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.box_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.BOX_SPEC_TASK,
            required=("table_body", "box_pose", "box_scale"),
        )
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.desk_fixture = self.get_fixture(id=self.navigation_spec["desk_body"])
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
        pose = self.navigation_spec.get(
            "desk_composite_robot_pose", self.navigation_spec["robot_pose"]
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.desk_fixture is not None)

    def _create_objects(self):
        super()._create_objects()
        xml_path = HolodeckPickBoxFromFloor._make_box_xml()
        try:
            box = MJCFObject(
                name="box",
                mjcf_path=xml_path,
                scale=float(self.box_task_spec["box_scale"]),
            )
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.objects[box.name] = box
        self.model.merge_objects([box])

    def _place_box_on_desk(self):
        box = self.objects["box"]
        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        pose = self.box_task_spec["box_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        self._box_floor_geom_id = self.sim.model.geom_name2id("floor")

    def _reset_internal(self):
        self._box_layout_ready = False
        super()._reset_internal()
        self._place_box_on_desk()
        if self._box_on_floor():
            raise RuntimeError(f"{self.scene_name}: box starts on the floor")
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: composite reset has robot collision"
            )
        self._box_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.desk_fixture, max_distance=self.success_radius + 0.25
            )
            <= self.success_radius
        )

    def _box_on_floor(self):
        return bool(HolodeckPlaceBoxOnFloor._box_contacts_floor(self))

    def _box_root_z(self):
        return float(
            self.sim.data.body_xpos[
                self.sim.model.body_name2id(self.objects["box"].root_body)
            ][2]
        )

    def _drawer_closed(self):
        return bool(RoomOpenArticulation._raw_success(self))

    def _post_action(self, action):
        return HolodeckMultiHopComposite._post_action(self, action)

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, place the cardboard box on the floor, then close the drawer",
            success_definition=(
                "remain near the desk for 1 second, keep the box in contact with "
                "the floor for 1 second, then keep the specified drawer closed for 1 second"
            ),
        )
        return meta


class HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat(
    _HolodeckOneSecondStageHoldMixin,
    HolodeckPlaceCupOnPlacemat,
    HolodeckMultiHopComposite,
):
    """Lift the cup, walk around the desk, then place that cup on its mat."""

    ABSTRACT = False
    ATOMIC_SKILL = "PickCupThenApproachDeskThenPlaceCupOnPlacemat"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1664, 2025)
    SOURCE_SPEC_TASK = "pick_cup_then_approach_desk_then_place_cup_on_placemat"
    COMPOSITE_SPEC_SCENES = (
        "train_1018",
        "train_1664",
        "train_2025",
    )
    NAVIGATION_SPEC_TASK = "approach_desk"
    # Composite-only ergonomic variant: the cup keeps the canonical asset and
    # task scale used elsewhere, while this private MJCF narrows its diameter
    # and elongates its height for a more reliable G1 grasp.
    CUP_ASSET = "objects/lightwheel/glass_cup/GlassCup014/model_long_grip.xml"
    CUP_RADIAL_SCALE = 0.6
    CUP_VERTICAL_SCALE = 1.2
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "pick_cup",
            "_cup_lifted",
            task_cls=HolodeckPickCupFromDesk,
            on_enter="_begin_cup_lift_stage",
            metadata={"skill": "lift", "object": "cup", "min_lift_m": 0.30},
        ),
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached_after_lift",
            task_cls=HolodeckApproachDesk,
            on_enter="_begin_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "place_cup_on_placemat",
            "_cup_on_placemat",
            task_cls=HolodeckPlaceCupOnPlacemat,
            metadata={"skill": "place", "object": "cup", "target": "placemat"},
        ),
    )

    def _uses_composite_spec(self):
        return self.scene_name in self.COMPOSITE_SPEC_SCENES

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.SOURCE_SPEC_TASK
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.SOURCE_SPEC_TASK
        super()._setup_task_fixtures()
        self.target_task_spec = dict(self.task_spec)
        self.source_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SOURCE_SPEC_TASK,
            required=(
                "source_table_body",
                "source_robot_pose",
                "source_cup_pose",
                "source_support_z",
            ),
        )
        self.source_table_fixture = self.get_fixture(
            id=self.source_task_spec["source_table_body"]
        )
        if self.source_table_fixture is not None:
            self.fixture_refs["source_cup_surface"] = self.source_table_fixture
            pose = self.source_task_spec["source_robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "success_radius"),
        )
        # Keep the elongated cup supported on both calibrated desks.  The
        # source/target specs are copied per task instance, so other cup tasks
        # retain their original poses and dimensions.
        self._adjust_elongated_cup_poses()
        if self.navigation_spec["desk_body"] != self.target_task_spec["table_body"]:
            raise RuntimeError(
                f"{self.scene_name}: cup and navigation desks must match"
            )
        self.navigation_desk_fixture = self.get_fixture(
            id=self.navigation_spec["desk_body"]
        )
        if self.navigation_desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
        self.task_spec = self.target_task_spec

    def _adjust_elongated_cup_poses(self):
        old_half_height = abs(
            float(self.target_task_spec["cup_pose"]["pos"][2])
            - float(self.target_task_spec["support_z"])
        )
        new_half_height = old_half_height * self.CUP_VERTICAL_SCALE
        delta_z = new_half_height - old_half_height
        source_pose = dict(self.source_task_spec["source_cup_pose"])
        source_pose["pos"] = list(source_pose["pos"])
        source_pose["pos"][2] += delta_z
        self.source_task_spec["source_cup_pose"] = source_pose
        target_pose = dict(self.target_task_spec["cup_pose"])
        target_pose["pos"] = list(target_pose["pos"])
        target_pose["pos"][2] += delta_z
        self.target_task_spec["cup_pose"] = target_pose

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.source_table_fixture is not None
            and self.navigation_desk_fixture is not None
        )

    def _reset_internal(self):
        self._cup_layout_ready = False
        HolodeckEnv._reset_internal(self)
        self._place("placemat", self.target_task_spec["placemat_pose"])
        self._place("cup", self.source_task_spec["source_cup_pose"])
        self.sim.forward()
        self._initial_placemat_pos = self.object_pos("placemat").copy()
        self._initial_cup_pos = self.object_pos("cup").copy()
        self._success_hold = 0
        if not self._initial_layout_valid():
            raise RuntimeError(f"{self.scene_name}: cup placemat reset is invalid")
        self._cup_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _initial_layout_valid(self):
        return bool(
            not self._cup_on_placemat()
            and self._object_is_supported_by_desk("placemat")
            and self._object_is_supported_by_desk("cup")
            and np.linalg.norm(
                self.object_pos("cup")[:2] - self.object_pos("placemat")[:2]
            )
            > 0.05
        )

    def _object_is_supported_by_desk(self, name):
        support_z = (
            self.target_task_spec["support_z"]
            if name == "placemat"
            else self.source_task_spec["source_support_z"]
        )
        return bool(
            abs(self.object_bottom_z(name) - float(support_z))
            <= self.SUPPORT_Z_TOLERANCE
        )

    def _validate_initial_composite_state(self):
        if getattr(self, "_cup_layout_ready", False):
            super()._validate_initial_composite_state()

    def _begin_cup_lift_stage(self):
        self._cup_lift_start_z = self._cup_root_z()

    def _cup_root_z(self):
        return float(self.sim.data.get_body_xpos(self.objects["cup"].root_body)[2])

    def _cup_lifted(self):
        return bool(
            self._cup_root_z()
            > self._cup_lift_start_z + HolodeckPickCupFromDesk.MIN_LIFT_DELTA
        )

    def _begin_desk_stage(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _robot_base_xy(self):
        return np.asarray(
            self.sim.data.get_body_xpos(self.robots[0].robot_model.root_body),
            dtype=float,
        )[:2]

    def _desk_reached_after_lift(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy) >= 0.15:
            self._desk_stage_moved = True
        return bool(
            self._desk_stage_moved
            and self.robot_fixture_collision_distance(
                self.navigation_desk_fixture,
                max_distance=float(self.navigation_spec["success_radius"]) + 0.25,
            )
            <= float(self.navigation_spec["success_radius"])
        )

    def _post_action(self, action):
        return HolodeckMultiHopComposite._post_action(self, action)

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            source_fixture=self.source_table_fixture.stable_id,
            target_fixture=self.navigation_desk_fixture.stable_id,
            lang="pick up the cup and lift it, walk to the desk, then move the cup onto the placemat",
            success_definition=(
                "lift the cup more than 0.30 m for 1 second, move at least 0.15 m "
                "and remain near the desk for 1 second, then keep the cup on the placemat for 1 second"
            ),
        )
        return meta


class HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk(
    _HolodeckOneSecondStageHoldMixin,
    HolodeckPlaceBoxOnDesk,
    HolodeckMultiHopComposite,
):
    """Place the desk box on the floor, then return it to the desk."""

    ABSTRACT = False
    ATOMIC_SKILL = "PlaceBoxOnFloorThenPlaceBoxOnDesk"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    BOX_SPEC_TASK = "place_box_on_floor"
    BOX_SCALE = HolodeckPlaceBoxOnDesk.BOX_SCALE
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "place_box_on_floor",
            "_box_on_floor",
            task_cls=HolodeckPlaceBoxOnFloor,
            metadata={"skill": "place", "object": "box", "target": "floor"},
        ),
        HolodeckAtomicTask(
            "place_box_on_desk",
            "_box_placed_on_desk",
            task_cls=HolodeckPlaceBoxOnDesk,
            metadata={"skill": "place", "object": "box", "target": "desk"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        # Apply only the box calibration needed by this composite.
        box_spec = get_holodeck_task_spec(self.scene_name, self.BOX_SPEC_TASK)
        box_bodies = tuple(box_spec.get("remove_scene_bodies", ()))
        if box_bodies:
            removed = self.remove_scene_bodies(box_bodies)
            missing = set(box_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured box bodies are missing: {sorted(missing)}"
                )
        HolodeckPlaceBoxOnFloor._add_desk_support(self, box_spec)
        # Both box atomics list the same laptop body; it was removed above,
        # so invoke the shared baked-scene setup without removing it twice.
        HolodeckEnv._setup_baked_scene_fixtures(self)

    def _setup_task_fixtures(self):
        HolodeckMultiHopComposite._setup_task_fixtures(self)
        self.box_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.BOX_SPEC_TASK,
            required=("room_id", "table_body", "box_pose", "box_scale"),
        )
        self.box_desk_body = self.box_task_spec["table_body"]
        self.desk_body = self.box_desk_body
        self.desk_fixture = self.get_fixture(id=self.box_desk_body)
        self.fixture_refs["place_box_on_desk"] = self.desk_fixture
        pose = self.box_task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )
        if self.find_scene_body(self.box_desk_body) is None:
            raise RuntimeError(
                f"{self.scene_name}: desk body is missing: {self.box_desk_body}"
            )

    def _is_scene_valid(self):
        return bool(self.find_scene_body(self.box_task_spec["table_body"]) is not None)

    def _create_objects(self):
        HolodeckMultiHopComposite._create_objects(self)
        xml_path = HolodeckPickBoxFromFloor._make_box_xml()
        try:
            box = MJCFObject(
                name="box",
                mjcf_path=xml_path,
                scale=float(self.BOX_SCALE),
            )
            scale = float(self.BOX_SCALE)
            bbox_elem = ET.Element(
                "geom",
                {
                    "pos": "0 0 0",
                    "size": " ".join(
                        str(float(value) * scale)
                        for value in HolodeckPickBoxFromFloor.BOX_HALF_SIZE
                    ),
                },
            )
            box._regions["bbox"] = {"elem": bbox_elem}
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.objects[box.name] = box
        self.model.merge_objects([box])

    def _reset_internal(self):
        self._box_layout_ready = False
        # Bypass HolodeckPlaceBoxOnDesk._reset_internal: it expects the
        # atomic task's ``self.task_spec`` and would apply the wrong pose.
        HolodeckEnv._reset_internal(self)
        box = self.objects["box"]
        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        pose = self.box_task_spec["box_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        support_top_z = self.box_task_spec.get("support_top_z")
        if support_top_z is not None:
            points, _ = self._body_geom_points([box.root_body])
            qpos[2] += float(support_top_z) - float(np.min(points[:, 2])) - 0.001
            self.sim.data.set_joint_qpos(box.joints[0], qpos)
            self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
            self.sim.forward()
        else:
            # The floor-task pose was calibrated for its larger box scale.
            # Re-seat this composite's smaller box on the actual desk top so
            # its initial support remains valid without changing either
            # atomic task's calibration.
            box_points, _ = self._body_geom_points([box.root_body])
            desk_points, _ = self._body_geom_points([self.box_desk_body])
            if not len(box_points) or not len(desk_points):
                raise RuntimeError(f"{self.scene_name}: missing box/desk geometry")
            qpos[2] += (
                float(np.max(desk_points[:, 2]) - np.min(box_points[:, 2])) - 0.001
            )
            self.sim.data.set_joint_qpos(box.joints[0], qpos)
            self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
            self.sim.forward()
        self._box_floor_geom_id = self.sim.model.geom_name2id("floor")
        self._box_initial_z = self._box_root_z()
        if self._box_on_floor():
            raise RuntimeError(f"{self.scene_name}: box starts on the floor")
        if not self._box_contacts_desk() and not self._box_at_desk_height():
            raise RuntimeError(f"{self.scene_name}: box is not supported by the desk")
        self._box_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if getattr(self, "_box_layout_ready", False):
            super()._validate_initial_composite_state()

    def _box_on_floor(self):
        return bool(HolodeckPlaceBoxOnFloor._box_contacts_floor(self))

    def _box_placed_on_desk(self):
        return bool(self._box_contacts_desk() or self._box_at_desk_height())

    def _box_root_z(self):
        return float(
            self.sim.data.body_xpos[
                self.sim.model.body_name2id(self.objects["box"].root_body)
            ][2]
        )

    def _box_contacts_desk(self):
        box_id = self.sim.model.body_name2id(self.objects["box"].root_body)
        box_bodies = set(self._body_tree_ids([box_id]))
        desk_id = self.sim.model.body_name2id(self.box_desk_body)
        box_desk_body_ids = set(self._body_tree_ids([desk_id]))
        return any(
            float(contact.dist) <= HolodeckPlaceBoxOnFloor.CONTACT_MARGIN
            and (
                (
                    int(self.sim.model.geom_bodyid[contact.geom1]) in box_bodies
                    and int(self.sim.model.geom_bodyid[contact.geom2])
                    in box_desk_body_ids
                )
                or (
                    int(self.sim.model.geom_bodyid[contact.geom2]) in box_bodies
                    and int(self.sim.model.geom_bodyid[contact.geom1])
                    in box_desk_body_ids
                )
            )
            for contact in (self.sim.data.contact[i] for i in range(self.sim.data.ncon))
        )

    def _box_at_desk_height(self):
        """Accept a calibrated shallow overlap when MuJoCo emits no contact."""
        box_points, _ = self._body_geom_points([self.objects["box"].root_body])
        desk_points, _ = self._body_geom_points([self.box_desk_body])
        if not len(box_points) or not len(desk_points):
            return False
        box_min = np.min(box_points, axis=0)
        box_max = np.max(box_points, axis=0)
        desk_max = np.max(desk_points, axis=0)
        xy_overlap = np.all(box_max[:2] >= np.min(desk_points, axis=0)[:2]) and np.all(
            np.max(desk_points, axis=0)[:2] >= box_min[:2]
        )
        support_top_z = self.box_task_spec.get("support_top_z")
        if support_top_z is None:
            expected_box_bottom_z = float(desk_max[2])
        else:
            expected_box_bottom_z = float(support_top_z) - 0.001
        return bool(
            xy_overlap and abs(float(box_min[2]) - expected_box_bottom_z) <= 0.01
        )

    def _post_action(self, action):
        return HolodeckMultiHopComposite._post_action(self, action)

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="place the cardboard box on the floor, then place it back on the desk",
            success_definition=(
                "keep the box in contact with the floor for 1 second, then keep "
                "then keep the box in contact with the desk for 1 second"
            ),
        )
        return meta


# Backward-compatible Python symbol; only the renamed class is registered.
HolodeckPlaceBoxOnFloorThenPickBook = HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk

__all__ = [
    "HolodeckApproachDeskThenOpenLaptop",
    "HolodeckOpenDrawerThenTakeTissueBox",
    "HolodeckOpenDrawerTakeTissueBoxToDesk",
    "HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer",
    "HolodeckApproachDeskThenCloseLaptopMoveCup",
    "HolodeckSugarCubeThenCloseJar",
    "HolodeckOpenJarAddSugarThenCloseJar",
    "HolodeckRemoveFolderFromShelfToDesk",
    "HolodeckPickBookFromSourceToDesk",
    "HolodeckApproachDeskPullChairThenOpenLaptop",
    "HolodeckMoveCupToCenterThenOpenLaptop",
    "HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop",
    "HolodeckCloseLaptopCloseDrawerThenApproachWindow",
    "HolodeckApproachTrashCanThenPickBoxFromFloor",
    "HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk",
    "HolodeckCloseLaptopThenApproachTrashCanThenPickBox",
    "HolodeckApproachWindowThenPickBox",
    "HolodeckApproachWindowThenPickBoxFromFloor",
    "HolodeckCloseLaptopThenApproachTrashCanThenPickBoxFromFloor",
    "HolodeckPullChairThenApproachTrashCanThenPickBox",
    "HolodeckTakeTissueBoxThenPlaceCup",
    "HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop",
    "HolodeckPickBookFromSourceToDeskThenApproachTrashCan",
    "HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan",
    "HolodeckApproachDeskThenIceCubeInGlass",
    "HolodeckCloseJarThenApproachDeskThenApproachTrashCan",
    "HolodeckApproachTrashCanThenApproachDeskThenOpenJar",
    "HolodeckIceCubeThenApproachWindowThenPickBox",
    "HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer",
    "HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat",
    "HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk",
]
