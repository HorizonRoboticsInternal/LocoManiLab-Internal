"""Navigation-led and coupled-transfer Holodeck workflows."""

from __future__ import annotations

import os

import numpy as np
import robocasa
from robocasa.models.objects.objects import MJCFObject

from robocasa.environments.holodeck.tasks.approach_desk import HolodeckApproachDesk
from robocasa.environments.holodeck.tasks.approach_trash_can import (
    HolodeckApproachTrashCan,
)
from robocasa.environments.holodeck.tasks.approach_window import (
    HolodeckApproachWindow,
)
from robocasa.environments.holodeck.tasks.ice_cube_in_glass import (
    HolodeckIceCubeInGlass,
)
from robocasa.environments.holodeck.tasks.laptop import (
    HolodeckCloseLaptop,
    HolodeckOpenLaptop,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.tasks.pick_cup_from_desk import (
    HolodeckPickCupFromDesk,
)
from robocasa.environments.holodeck.tasks.place_cup_on_placemat import (
    HolodeckPlaceCupOnPlacemat,
)
from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.throw_bottle_into_trash_can import (
    HolodeckThrowBottleIntoTrashCan,
)
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robocasa.environments.holodeck.holodeck import normalize_holodeck_scene_name


def _require_supported_scene_ids(scene_ids, supported_scene_ids):
    ids = (scene_ids,) if isinstance(scene_ids, (int, str)) else scene_ids
    requested = {
        normalize_holodeck_scene_name(
            int(scene_id)
            if isinstance(scene_id, str) and scene_id.isdecimal()
            else scene_id
        )
        for scene_id in ids
    }
    supported = {f"train_{scene_id}" for scene_id in supported_scene_ids}
    unsupported = requested - supported
    if unsupported:
        raise ValueError(f"Unsupported Holodeck scenes: {sorted(unsupported)}")


class HolodeckIceCubeThenPlaceCup(
    HolodeckPlaceCupOnPlacemat,
    HolodeckMultiHopComposite,
):
    """Put ice in a cup, then move that cup onto a desk placemat."""

    ABSTRACT = False
    ATOMIC_SKILL = "IceCubeThenPlaceCup"
    TASK_FAMILY = "composite"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    ICE_SPEC_TASK = "ice_cube_then_place_cup"
    ICE_ASSET = HolodeckIceCubeInGlass.ICE_ASSET
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "put_ice_in_cup",
            "_ice_inserted_after_release",
            task_cls=HolodeckIceCubeInGlass,
            hold_steps=5,
            on_enter="_begin_ice_stage",
            metadata={"skill": "pick_place", "object": "ice_cube", "target": "cup"},
        ),
        HolodeckAtomicTask(
            "place_cup_on_placemat",
            "_cup_placed_after_ice",
            task_cls=HolodeckPlaceCupOnPlacemat,
            hold_steps=5,
            on_enter="_begin_cup_stage",
            metadata={"skill": "pick_place", "object": "cup", "target": "placemat"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        # The Objaverse desk collision mesh is open in the two target scenes.
        # Add the same task-local support used by the atomic ice task, but do
        # so only for this composite's newly calibrated layout.
        ice_spec = get_holodeck_task_spec(
            self.scene_name,
            self.ICE_SPEC_TASK,
            required=("table_body", "cup_pose", "ice_pose"),
        )
        remove_scene_bodies = tuple(ice_spec.get("remove_scene_bodies", ()))
        if self.scene_name == "train_1018" and remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured ice-cube composite bodies are missing: "
                    f"{sorted(missing)}"
                )
        HolodeckIceCubeInGlass._add_missing_desk_support(self, ice_spec)
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.ice_task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.ICE_SPEC_TASK,
            required=(
                "table_body",
                "robot_pose",
                "object_scale",
                "cup_pose",
                "ice_pose",
            ),
        )
        if self.scene_name == "train_1018":
            # Scene 1018 uses the other study desk for this composite. Keep
            # the cup, placemat, support calibration, and scene cleanup on
            # that same desk instead of inheriting the standalone cup task's
            # unrelated desk layout.
            self.ice_task_spec = get_holodeck_task_spec(
                self.scene_name,
                self.ICE_SPEC_TASK,
                required=(
                    "table_body",
                    "robot_pose",
                    "object_scale",
                    "cup_pose",
                    "ice_pose",
                    "placemat_scale",
                    "placemat_pose",
                    "target_half_size",
                    "support_z",
                ),
            )
            self.task_spec = self.ice_task_spec
        table_body = self.ice_task_spec["table_body"]
        tables = tuple(
            self.get_fixture(capabilities=("support",), return_all=True) or ()
        )
        if not any(fixture.body_name == table_body for fixture in tables):
            # Objaverse desk exports may not carry the semantic support
            # capability.  ``get_fixture(return_all=True)`` without a query
            # does not enumerate the raw ProcTHOR fixture mapping, so search
            # both registries explicitly by the calibrated body name.
            semantic_fixtures = tuple(
                getattr(self.semantic_scene, "fixtures", {}).values()
            )
            raw_fixtures = tuple(getattr(self, "fixtures", {}).values())
            tables = semantic_fixtures + raw_fixtures
        self.ice_table_fixture = next(
            (fixture for fixture in tables if fixture.body_name == table_body),
            None,
        )
        if self.ice_table_fixture is not None:
            if self.scene_name == "train_1018":
                self.desk_fixture = self.ice_table_fixture
                self.fixture_refs["cup_surface"] = self.ice_table_fixture
                self.fixture_refs["placemat_surface"] = self.ice_table_fixture
            self.fixture_refs["ice_table"] = self.ice_table_fixture
            pose = self.ice_task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.ice_table_fixture is not None)

    def _create_objects(self):
        super()._create_objects()
        # The cup is created by HolodeckPlaceCupOnPlacemat; add only the
        # additional ice object so both stages operate on one shared cup.
        ice = MJCFObject(
            name="ice_cube",
            mjcf_path=os.path.join(robocasa.models.assets_root, self.ICE_ASSET),
            scale=float(self.ice_task_spec["object_scale"]),
        )
        self.objects[ice.name] = ice
        self.model.merge_objects([ice])

    def _reset_internal(self):
        # The placement parent invokes the composite reset guard before this
        # task applies its shared cup-and-ice layout. Validate only after both
        # objects have their calibrated poses.
        self._ice_layout_ready = False
        super()._reset_internal()
        self._place("cup", self.ice_task_spec["cup_pose"])
        self._place("ice_cube", self.ice_task_spec["ice_pose"])
        self.sim.forward()
        self._ice_layout_ready = True
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        if not getattr(self, "_ice_layout_ready", False):
            return
        super()._validate_initial_composite_state()

    def _begin_ice_stage(self):
        pass

    def _ice_in_cup_region(self):
        return HolodeckIceCubeInGlass._ice_in_cup_region(self)

    def _ice_inserted_after_release(self):
        return self._ice_in_cup_region()

    def _cup_placed_after_ice(self):
        # Keep the composite placement stage exactly aligned with the atomic
        # cup-on-placemat primitive, including its one-second hold semantics.
        # The parent ``_post_action`` updates ``_success_hold`` from the same
        # footprint predicate before the composite state machine evaluates
        # this callback.
        return bool(HolodeckPlaceCupOnPlacemat._check_success(self))

    def _begin_cup_stage(self):
        pass

    def _post_action(self, action):
        return super()._post_action(action)

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._ice_inserted_after_release()
            and self._cup_placed_after_ice()
        )

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="put the ice cube in the cup, then move the cup onto the placemat",
            source_object="ice_cube",
            target_object="cup",
            final_target_object="placemat",
            success_definition=(
                "the ice cube must remain in the calibrated cup region, then "
                "the cup must remain in the calibrated placemat region"
            ),
        )
        return meta


class HolodeckApproachDeskThenPullChair(
    HolodeckPullChairFromDesk,
    HolodeckMultiHopComposite,
):
    """Walk to the desk, then pull its calibrated chair out from underneath."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenPullChair"
    TASK_FAMILY = "composite"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    COMPOSITE_SPEC_TASK = "approach_desk_then_pull_chair"
    COMPOSITE_SCENE_IDS = (1197, 1664, 2025)
    NAVIGATION_SPEC_TASK = "approach_desk"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "pull_chair_from_desk",
            "_chair_pulled_after_approach",
            task_cls=HolodeckPullChairFromDesk,
            hold_steps=None,
            on_enter="_begin_pull_stage",
            metadata={"skill": "pull", "object": "chair"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.PUSH_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }:
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
            self.PUSH_SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
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
        self.navigation_desk_fixture = next(
            (
                fixture
                for fixture in self.get_fixture(category="desk", return_all=True) or ()
                if fixture.body_name == self.navigation_spec["desk_body"]
            ),
            None,
        )
        if self.navigation_desk_fixture is None:
            semantic_fixtures = tuple(
                getattr(self.semantic_scene, "fixtures", {}).values()
            )
            raw_fixtures = tuple(getattr(self, "fixtures", {}).values())
            self.navigation_desk_fixture = next(
                (
                    fixture
                    for fixture in semantic_fixtures + raw_fixtures
                    if fixture.body_name == self.navigation_spec["desk_body"]
                ),
                None,
            )
        if self.navigation_desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
        pose = self.navigation_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid() and self.navigation_desk_fixture is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        distance = self.robot_fixture_collision_distance(
            self.navigation_desk_fixture,
            max_distance=self.success_radius + 1.0,
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: composite navigation spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.navigation_desk_fixture,
                max_distance=self.success_radius + 0.25,
            )
            <= self.success_radius
        )

    def _begin_pull_stage(self):
        body_id = self.sim.model.body_name2id(self.chair_body)
        self._chair_stage_start_xy = np.asarray(
            self.sim.data.body_xpos[body_id][:2], dtype=float
        ).copy()

    def _chair_pulled_after_approach(self):
        return HolodeckPullChairFromDesk._chair_has_min_pull_displacement(self)

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._chair_pulled_after_approach()
        )

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, then pull the chair out from under it",
            target_fixture=self.navigation_desk_fixture.stable_id,
            success_definition=(
                "first remain within the calibrated desk collision distance, then "
                f"pull the chair at least {self.MIN_PULL_DISPLACEMENT:.2f} m from "
                "its reset position and keep it there for 1 second"
            ),
        )
        return meta


class HolodeckApproachWindowThenDeskThenPickCup(
    HolodeckPickCupFromDesk,
    HolodeckMultiHopComposite,
):
    """Walk to the window, walk to the desk, then pick up its glass cup."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachWindowThenDeskThenPickCup"
    TASK_FAMILY = "composite"
    SUCCESS_DEBUG_ENABLED = True
    SUPPORTED_SCENE_IDS = (117, 1018, 1664, 2025)
    COMPOSITE_SPEC_TASK = "approach_window_then_desk_then_pick_cup"
    COMPOSITE_SCENE_IDS = (1664, 2025)
    NAVIGATION_SPEC_TASK = "approach_window"
    DESK_SPEC_TASK = "approach_desk"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_window",
            "_window_reached",
            task_cls=HolodeckApproachWindow,
            hold_steps=HolodeckApproachWindow.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "window"},
        ),
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "pick_cup",
            "_cup_lifted_after_navigation",
            task_cls=HolodeckPickCupFromDesk,
            hold_steps=None,
            on_enter="_begin_cup_stage",
            metadata={"skill": "pick", "object": "cup"},
        ),
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        _require_supported_scene_ids(scene_ids, self.SUPPORTED_SCENE_IDS)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        navigation_task = (
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.NAVIGATION_SPEC_TASK
        )
        self.window_spec = get_holodeck_task_spec(
            self.scene_name,
            navigation_task,
            required=("window_body", "robot_pose", "success_radius"),
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
        desk_task = (
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.DESK_SPEC_TASK
        )
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            desk_task,
            required=("desk_body", "success_radius"),
        )
        self.navigation_desk_fixture = next(
            (
                fixture
                for fixture in self.get_fixture(category="desk", return_all=True)
                if fixture.body_name == self.navigation_spec["desk_body"]
            ),
            None,
        )
        if self.navigation_desk_fixture is None:
            calibrated_body = self.navigation_spec["desk_body"]
            self.navigation_desk_fixture = next(
                (
                    fixture
                    for fixture in tuple(
                        getattr(self.semantic_scene, "fixtures", {}).values()
                    )
                    + tuple(getattr(self, "fixtures", {}).values())
                    if fixture.body_name == calibrated_body
                ),
                None,
            )
        if self.window_fixture is not None:
            self.fixture_refs["approach_window"] = self.window_fixture
        if self.navigation_desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
        # Keep the composite's window-navigation start aligned with the atomic
        # window task unless a scene explicitly provides an override.
        pose = self.window_spec.get(
            "composite_robot_pose", self.window_spec["robot_pose"]
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            # Windows are baked collision bodies in Holodeck scenes and are
            # not guaranteed to be present in the semantic fixture registry.
            and self.window_body
            and self.navigation_desk_fixture is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        distance = HolodeckApproachWindow.success_distance(self)
        min_distance = self.window_success_radius + 0.15
        if distance < min_distance:
            raise RuntimeError(
                f"{self.scene_name}: composite navigation spawn is too close to window "
                f"({distance:.3f} m; needs >= {min_distance:.3f} m)"
            )

    def _window_reached(self):
        return bool(
            HolodeckApproachWindow.success_distance(self) <= self.window_success_radius
        )

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.navigation_desk_fixture,
                max_distance=self.desk_success_radius + 0.25,
            )
            <= self.desk_success_radius
        )

    def _begin_cup_stage(self):
        self._cup_stage_initial_z = self._cup_root_z()

    def _cup_lifted_after_navigation(self):
        return bool(self._cup_root_z() > self._cup_stage_initial_z + self.lift_delta)

    def _cup_root_z(self):
        return float(self.sim.data.get_body_xpos(self.objects["cup"].root_body)[2])

    @property
    def window_success_radius(self):
        return float(self.window_spec["success_radius"])

    @property
    def desk_success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._cup_lifted_after_navigation()
        )

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the rightmost window, walk to the desk, then pick up the glass cup",
            target_window=self.window_body,
            target_fixture=self.navigation_desk_fixture.stable_id,
            source_object="cup",
            success_definition=(
                "first reach the calibrated window, then reach the desk, then lift "
                "the glass cup from that desk by the calibrated height"
            ),
        )
        return meta


class HolodeckApproachDeskIceCubePlaceCupThenPickCup(
    HolodeckPlaceCupOnPlacemat,
    HolodeckMultiHopComposite,
):
    """Reach the desk, place the ice-filled cup, then pick up that same cup."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskIceCubePlaceCupThenPickCup"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    ICE_SPEC_TASK = "ice_cube_then_place_cup"
    APPROACH_SPEC_TASK = "approach_desk"
    COMPOSITE_SPEC_TASK = "approach_desk_ice_cube_place_cup_then_pick_cup"
    COMPOSITE_SCENE_IDS = (1197, 1664, 2025)
    LAPTOP_BODY = "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    ICE_ASSET = HolodeckIceCubeInGlass.ICE_ASSET
    CUP_ASSET = HolodeckPickCupFromDesk.CUP_ASSET
    PICK_CUP_LIFT_DELTA = HolodeckPickCupFromDesk.MIN_LIFT_DELTA
    COMPOSITE_LAYOUTS = {
        "train_117": {
            # The robot approaches this desk from -X; keep the ice and target
            # cup on the near side while leaving the placemat beside them.
            "cup_pose": {"pos": [5.52, 4.80, 0.7773], "quat": [1.0, 0.0, 0.0, 0.0]},
            "ice_pose": {"pos": [5.52, 4.45, 0.72108], "quat": [1.0, 0.0, 0.0, 0.0]},
            "placemat_pose": {
                "pos": [5.65, 5.03, 0.7015],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_1018": {
            # Keep all three tabletop objects on the same desk targeted by
            # HolodeckApproachDesk.  The mat, cup, and ice are aligned from
            # near to far so the cup does not block the initial ice grasp.
            "cup_pose": {"pos": [7.55, 0.82, 0.8009], "quat": [1.0, 0.0, 0.0, 0.0]},
            # Put the ice beyond the cup along the placemat-to-cup axis so
            # the cup does not occlude the grasp approach.
            "ice_pose": {"pos": [7.55, 1.16, 0.7248], "quat": [1.0, 0.0, 0.0, 0.0]},
            "placemat_pose": {
                "pos": [7.55, 0.48, 0.7248],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_1197": {
            "cup_pose": {"pos": [5.22, 2.10, 0.8061], "quat": [1.0, 0.0, 0.0, 0.0]},
            "ice_pose": {"pos": [5.30, 1.85, 0.7501], "quat": [1.0, 0.0, 0.0, 0.0]},
            "placemat_pose": {
                "pos": [5.40, 2.35, 0.7303],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_1664": {
            "cup_pose": {"pos": [5.64, 1.85, 0.8257], "quat": [1.0, 0.0, 0.0, 0.0]},
            "ice_pose": {"pos": [5.70, 1.60, 0.7697], "quat": [1.0, 0.0, 0.0, 0.0]},
            "placemat_pose": {
                "pos": [5.76, 2.15, 0.7499],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "train_2025": {
            "cup_pose": {"pos": [5.64, 2.15, 0.8459], "quat": [1.0, 0.0, 0.0, 0.0]},
            "ice_pose": {"pos": [5.64, 1.90, 0.7899], "quat": [1.0, 0.0, 0.0, 0.0]},
            "placemat_pose": {
                "pos": [5.76, 2.40, 0.7704],
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
            "_ice_inserted_after_release",
            task_cls=HolodeckIceCubeInGlass,
            hold_steps=20,
            on_enter="_begin_ice_stage",
            metadata={"skill": "pick_place", "object": "ice_cube", "target": "cup"},
        ),
        HolodeckAtomicTask(
            "place_cup_on_placemat",
            "_cup_placed_after_ice",
            task_cls=HolodeckPlaceCupOnPlacemat,
            hold_steps=5,
            on_enter="_begin_cup_stage",
            metadata={"skill": "pick_place", "object": "cup", "target": "placemat"},
        ),
        HolodeckAtomicTask(
            "pick_cup",
            "_cup_lifted_after_placement",
            task_cls=HolodeckPickCupFromDesk,
            hold_steps=None,
            on_enter="_begin_pick_cup_stage",
            metadata={"skill": "pick", "object": "cup"},
        ),
    )

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        spec_task = (
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.ICE_SPEC_TASK
        )
        self.ice_task_spec = get_holodeck_task_spec(
            self.scene_name,
            spec_task,
            required=(
                "table_body",
                "robot_pose",
                "object_scale",
                "cup_pose",
                "ice_pose",
            ),
        )
        navigation_task = (
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.APPROACH_SPEC_TASK
        )
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            navigation_task,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        if self.scene_name == "train_1018":
            # Keep the remote spawn while targeting the same desk as the
            # calibrated approach_desk task.  Other composites retain their
            # own scene-specific navigation overrides.
            self.navigation_spec["desk_body"] = self.ice_task_spec["table_body"]
            self.navigation_spec["robot_pose"] = {
                "pos": [5.35, 1.25, 0.0],
                "yaw": 0.0,
            }
        layout = self.COMPOSITE_LAYOUTS[self.scene_name]
        self.ice_task_spec["cup_pose"] = layout["cup_pose"]
        self.ice_task_spec["ice_pose"] = layout["ice_pose"]
        self.placemat_task_spec = dict(self.task_spec)
        self.placemat_task_spec["cup_pose"] = layout["cup_pose"]
        self.placemat_task_spec["placemat_pose"] = layout["placemat_pose"]
        self.ice_table_fixture = self.get_fixture(id=self.ice_task_spec["table_body"])
        # The 1018 composite deliberately moves the cup and mat from the
        # standalone placement desk onto the desk used by approach_desk.  Keep
        # the inherited placement primitive bound to that same fixture and its
        # measured collision support height.
        if self.scene_name == "train_1018":
            self.placemat_task_spec["table_body"] = self.ice_task_spec["table_body"]
            self.placemat_task_spec["support_z"] = 0.72347
        self.navigation_desk_fixture = self.get_fixture(
            id=self.navigation_spec["desk_body"]
        )
        self.cup_desk_fixture = self.ice_table_fixture
        if self.scene_name == "train_1018" and self.ice_table_fixture is not None:
            self.desk_fixture = self.ice_table_fixture
        if self.navigation_desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
        if self.ice_table_fixture is not None:
            self.fixture_refs["ice_table"] = self.ice_table_fixture
        if self.cup_desk_fixture is not None:
            self.fixture_refs["cup_pick_surface"] = self.cup_desk_fixture
        pose = self.navigation_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _setup_baked_scene_fixtures(self):
        if self._uses_composite_spec():
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        if self.scene_name == "train_1018":
            removed = self.remove_scene_bodies((self.LAPTOP_BODY,))
            if self.LAPTOP_BODY not in removed:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{self.LAPTOP_BODY}"
                )
        super()._setup_baked_scene_fixtures()

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.navigation_desk_fixture is not None
            and self.ice_table_fixture is not None
            and self.cup_desk_fixture is not None
        )

    def _create_objects(self):
        # HolodeckPlaceCupOnPlacemat reads cup and placemat fields from the
        # active task_spec while creating its two shared objects.
        self.task_spec = self.placemat_task_spec
        super()._create_objects()
        root = robocasa.models.assets_root
        ice = MJCFObject(
            name="ice_cube",
            mjcf_path=os.path.join(root, self.ICE_ASSET),
            scale=float(self.ice_task_spec["object_scale"]),
        )
        self.objects[ice.name] = ice
        self.model.merge_objects([ice])
        self.task_spec = self.ice_task_spec

    def _setup_scene(self):
        super()._setup_scene()
        distance = self.robot_fixture_collision_distance(
            self.navigation_desk_fixture,
            max_distance=self.success_radius + 1.0,
        )
        if not np.isfinite(distance) or distance < self.success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: composite navigation spawn is too close to desk "
                f"({distance:.3f} m; needs >= {self.success_radius + 0.15:.3f} m)"
            )

    def _reset_internal(self):
        # The parent reset places and validates the cup/placemat layout before
        # this composite adds the ice to the shared cup.
        self.task_spec = self.placemat_task_spec
        super()._reset_internal()
        self._place("cup", self.ice_task_spec["cup_pose"])
        self._place("ice_cube", self.ice_task_spec["ice_pose"])
        self.sim.forward()
        self._reset_composite_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()
        self.task_spec = self.ice_task_spec

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.navigation_desk_fixture,
                max_distance=self.success_radius + 0.25,
            )
            <= self.success_radius
        )

    def _begin_ice_stage(self):
        pass

    def _ice_in_cup_region(self):
        return HolodeckIceCubeInGlass._ice_in_cup_region(self)

    def _ice_inserted_after_release(self):
        return HolodeckIceCubeInGlass._ice_in_cup_region(self)

    def _begin_cup_stage(self):
        pass

    def _cup_placed_after_ice(self):
        placement_stage_completed = any(
            task.name == "place_cup_on_placemat" for task in self.completed_atomic_tasks
        )
        return bool(placement_stage_completed or self._success_hold >= self.hold_steps)

    def _begin_pick_cup_stage(self):
        self._cup_stage_initial_z = self._cup_root_z()

    def _cup_root_z(self):
        return float(self.sim.data.get_body_xpos(self.objects["cup"].root_body)[2])

    def _cup_lifted_after_placement(self):
        return bool(
            self._cup_root_z() > self._cup_stage_initial_z + self.PICK_CUP_LIFT_DELTA
        )

    def _post_action(self, action):
        # The cup primitive's post-action predicate reads support_z and other
        # placement fields from its own calibration while the composite keeps
        # the ice calibration active for the shared ice object.
        saved_spec = self.task_spec
        self.task_spec = self.placemat_task_spec
        try:
            ret = super()._post_action(action)
            # self._log_success_conditions()
            return ret
        finally:
            self.task_spec = saved_spec

    @property
    def success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _check_success(self):
        return bool(
            HolodeckMultiHopComposite._check_success(self)
            and self._ice_inserted_after_release()
            and self._cup_placed_after_ice()
            and self._cup_lifted_after_placement()
        )

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "walk to the desk, put the ice cube in the cup, place the cup on "
                "the placemat, then pick up that glass cup"
            ),
            target_fixture=self.navigation_desk_fixture.stable_id,
            source_object="ice_cube",
            target_object="cup",
            final_target_object="cup",
            success_definition=(
                "reach the calibrated desk, keep the ice cube inside the cup, "
                "keep the moved cup fully on the stable placemat, then lift "
                "that same glass cup by more than 30 cm for one second"
            ),
        )
        return meta


class HolodeckApproachTrashCanThenThrowBottle(
    HolodeckThrowBottleIntoTrashCan,
    HolodeckMultiHopComposite,
):
    """Reach the trash can, then throw the empty bottle from the side table into it."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenThrowBottle"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018)
    NAVIGATION_SPEC_TASK = "approach_trash_can"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
        HolodeckAtomicTask(
            "throw_bottle",
            "_throw_success_after_approach",
            task_cls=HolodeckThrowBottleIntoTrashCan,
            hold_steps=None,
            on_enter="_begin_throw_stage",
            metadata={"skill": "pick_place", "object": "bottle", "target": "trash_can"},
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.throw_task_spec = self.task_spec
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        pose = self.navigation_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )
        self.task_spec = self.throw_task_spec

    def _setup_scene(self):
        super()._setup_scene()
        distance = HolodeckApproachTrashCan.success_distance(self)
        if distance < self.trash_can_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: trash-can composite spawn is too close "
                f"({distance:.3f} m; needs >= {self.trash_can_success_radius + 0.15:.3f} m)"
            )

    @property
    def trash_can_success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _trash_can_reached(self):
        return bool(
            HolodeckApproachTrashCan.success_distance(self)
            <= self.trash_can_success_radius
        )

    def _throw_success_after_approach(self):
        return bool(self._bottle_in_bin())

    def _begin_throw_stage(self):
        self._success_hold = 0

    def _reset_internal(self):
        self.task_spec = self.throw_task_spec
        super()._reset_internal()
        self.task_spec = self.throw_task_spec

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the trash can, then throw the empty bottle from the side table into it",
            source_object="bottle",
            source_fixture=self.source_body,
            target_fixture=self.trash_can_body,
            success_definition=(
                "first remain within the calibrated trash-can collision distance, "
                "then keep the bottle on the trash-can bottom support surface "
                "for one second"
            ),
        )
        return meta


class HolodeckApproachWindowThenDesk(HolodeckMultiHopComposite):
    """Reach the rightmost window, then walk to the calibrated desk."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachRightmostWindowThenDesk"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    SUPPORTED_SCENE_IDS = (117, 1018, 1664, 2025)
    # This navigation-only composite does not require the scene-default
    # laptop.  In particular, the calibrated Objaverse desks in 1664 and 2025
    # are only navigation targets here.
    USE_DEFAULT_ASSET_LAYOUT = False
    LAPTOP_CLEARED_SCENES = ("train_1664", "train_2025")
    WINDOW_SPEC_TASK = "approach_window"
    DESK_SPEC_TASK = "approach_desk"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_rightmost_window",
            "_window_reached",
            task_cls=HolodeckApproachWindow,
            hold_steps=HolodeckApproachWindow.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "rightmost_window"},
        ),
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            on_enter="_begin_desk_stage",
            metadata={"skill": "navigation", "target": "desk"},
        ),
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        _require_supported_scene_ids(scene_ids, self.SUPPORTED_SCENE_IDS)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        # The embedded laptops in these two Objaverse exports intersect the
        # desk mesh. This navigation-only task has no laptop stage, so remove
        # only the calibrated laptop root from these task-local scene copies.
        if self.scene_name in self.LAPTOP_CLEARED_SCENES:
            spec = get_holodeck_task_spec(
                self.scene_name,
                self.DESK_SPEC_TASK,
                required=("source_body",),
            )
            source_body = spec["source_body"]
            removed = self.remove_scene_bodies((source_body,))
            if source_body not in removed:
                raise RuntimeError(
                    f"{self.scene_name}: calibrated laptop source is missing: "
                    f"{source_body}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.window_spec = get_holodeck_task_spec(
            self.scene_name,
            self.WINDOW_SPEC_TASK,
            required=("window_body", "robot_pose", "success_radius"),
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
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DESK_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        desk_body = self.navigation_spec["desk_body"]
        # Objaverse desks in train_1664/train_2025 intentionally have an
        # empty semantic category. Resolve by the calibrated baked body name,
        # checking both the root and expanded body-name lists used by the
        # semantic and raw ProcTHOR fixture records.
        candidates = list(self.get_fixture(category="desk", return_all=True) or ())
        candidates.extend(getattr(self.semantic_scene, "fixtures", {}).values())
        candidates.extend(getattr(self, "fixtures", {}).values())
        self.desk_fixture = next(
            (
                fixture
                for fixture in candidates
                if desk_body
                in {
                    getattr(fixture, "body_name", None),
                    *getattr(fixture, "body_names", ()),
                }
            ),
            None,
        )
        if self.window_fixture is not None:
            self.fixture_refs["approach_rightmost_window"] = self.window_fixture
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
        pose = self.window_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        valid = bool(self.window_body and self.desk_fixture is not None)
        if not valid:
            missing = []
            if not self.window_body:
                missing.append("window_body")
            if self.desk_fixture is None:
                missing.append(f"desk_body={self.navigation_spec['desk_body']!r}")
            self._scene_invalid_reason = (
                f"{self.scene_name}: calibrated fixture missing ({', '.join(missing)})"
            )
        return valid

    def _setup_scene(self):
        super()._setup_scene()
        if self.window_body not in self.sim.model.body_names:
            raise RuntimeError(
                f"{self.scene_name}: calibrated rightmost window is missing: "
                f"{self.window_body}"
            )
        distance = HolodeckApproachWindow.success_distance(self)
        if distance < self.window_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: window composite spawn is too close "
                f"({distance:.3f} m; needs >= {self.window_success_radius + 0.15:.3f} m)"
            )
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: window composite spawn has collisions"
            )
        if not self._robot_footprint_is_in_room(radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: window composite spawn is off floor"
            )

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _robot_footprint_is_in_room(self, radius=0.25):
        base = self._robot_base_xy()
        samples = [base]
        samples.extend(
            base + float(radius) * np.array([np.cos(angle), np.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        )
        return all(
            self.room_of_point([point[0], point[1], 0.0]) is not None
            for point in samples
        )

    @property
    def window_success_radius(self):
        return float(self.window_spec["success_radius"])

    @property
    def desk_success_radius(self):
        return float(self.navigation_spec["success_radius"])

    def _window_reached(self):
        return bool(
            HolodeckApproachWindow.success_distance(self) <= self.window_success_radius
        )

    def _begin_desk_stage(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _desk_reached(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy) >= 0.15:
            self._desk_stage_moved = True
        return bool(
            self._desk_stage_moved
            and self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.desk_success_radius + 0.25,
            )
            <= self.desk_success_radius
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the rightmost window, then walk to the desk",
            target_window=self.window_body,
            target_fixture=self.desk_fixture.stable_id,
            success_definition=(
                "first remain within the calibrated rightmost-window collision "
                "distance, then remain within the calibrated desk collision distance"
            ),
        )
        return meta


class HolodeckApproachTrashCanThenThrowBottleThenApproachDesk(
    HolodeckApproachTrashCanThenThrowBottle,
):
    """Throw the side-table bottle into the bin, then return to the desk."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenThrowBottleThenApproachDesk"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SUCCESS_DEBUG_ENABLED = True
    DESK_SPEC_TASK = "approach_desk"
    COMPOSITE_SPEC_TASK = "approach_trash_can_then_throw_bottle_then_approach_desk"
    COMPOSITE_SCENE_IDS = (1197,)
    ATOMIC_TASKS = (
        *HolodeckApproachTrashCanThenThrowBottle.ATOMIC_TASKS,
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached_after_throw",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            on_enter="_begin_return_to_desk",
            metadata={"skill": "navigation", "target": "desk"},
        ),
    )

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        desk_spec_task = (
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.DESK_SPEC_TASK
        )
        self.desk_navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            desk_spec_task,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.desk_fixture = self.get_fixture(id=self.desk_navigation_spec["desk_body"])
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
        # The first navigation stage needs a remote pose that is outside the
        # final desk region too.  The atomic trash-can pose for train_1197 is
        # only 0.992 m from the desk, which would let the return stage start
        # inside its required 0.15 m margin.
        # For scenes without a task-specific composite calibration, keep the
        # remote spawn from the first navigation stage.  The desk atomic pose
        # is an operation-side stance and points at the desk, which makes the
        # initial composite orientation wrong for the trash-can stage.
        pose_spec = (
            self.desk_navigation_spec
            if self._uses_composite_spec()
            else self.navigation_spec
        )
        pose = pose_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.desk_fixture is not None)

    def _setup_scene(self):
        super()._setup_scene()
        distance = self.robot_fixture_collision_distance(
            self.desk_fixture,
            max_distance=self.desk_success_radius + 1.0,
        )
        if not np.isfinite(distance) or distance < self.desk_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: return-to-desk spawn is too close "
                f"({distance:.3f} m; needs >= {self.desk_success_radius + 0.15:.3f} m)"
            )

    @property
    def desk_success_radius(self):
        return float(self.desk_navigation_spec["success_radius"])

    def _begin_return_to_desk(self):
        self._desk_return_start_xy = self._robot_base_xy().copy()
        self._desk_return_moved = False

    def _desk_reached_after_throw(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_return_start_xy) >= 0.15:
            self._desk_return_moved = True
        return bool(
            self._desk_return_moved
            and self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.desk_success_radius + 0.25,
            )
            <= self.desk_success_radius
        )

    def _success_debug_details(self, task):
        if task is None or task.name != "throw_bottle":
            return ""

        valid = self._bottle_in_bin()

        return (
            f"on_bin_floor={valid}, "
            f"success_hold={self._success_hold}/{self.hold_steps}"
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "walk to the trash can, throw the empty bottle from the side table "
                "into it, then return to the desk"
            ),
            target_fixture=self.desk_fixture.stable_id,
            success_definition=(
                "first reach the calibrated trash can, keep the side-table bottle "
                "on its bottom support surface for one second, "
                "then move back to the calibrated desk"
            ),
        )
        return meta


class HolodeckApproachDeskThenOpenLaptopThenApproachWindow(
    HolodeckOpenLaptop,
    HolodeckMultiHopComposite,
):
    """Open the desk laptop, then walk to the rightmost window."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenOpenLaptopThenApproachWindow"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018)
    SPEC_TASK = "open_laptop"
    NAVIGATION_SPEC_TASK = "approach_desk"
    WINDOW_SPEC_TASK = "approach_window"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_desk",
            "_desk_reached",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "open_laptop",
            "_laptop_opened",
            task_cls=HolodeckOpenLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
        HolodeckAtomicTask(
            "approach_rightmost_window",
            "_window_reached_after_laptop",
            task_cls=HolodeckApproachWindow,
            hold_steps=HolodeckApproachWindow.HOLD_STEPS,
            on_enter="_begin_window_stage",
            metadata={"skill": "navigation", "target": "rightmost_window"},
        ),
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        _require_supported_scene_ids(scene_ids, self.SUPPORTED_SCENE_IDS)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.task_spec
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            self.NAVIGATION_SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.desk_fixture = self.get_fixture(id=self.navigation_spec["desk_body"])
        self.window_spec = get_holodeck_task_spec(
            self.scene_name,
            self.WINDOW_SPEC_TASK,
            required=("window_body", "robot_pose", "success_radius"),
        )
        self.window_body = self.window_spec["window_body"]
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
        # The composite begins at the desk, but its initial pose must also be
        # outside the final window success region. Keep that remote spawn
        # explicit in the scene calibration.
        pose = self.window_spec.get("composite_robot_pose")
        if pose is None:
            raise ValueError(
                f"{self.scene_name}/approach_window: composite_robot_pose "
                "is required for desk/laptop/window composites"
            )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.desk_fixture is not None
            and self.window_body
        )

    def _place_robot_for_articulation(self, distance):
        del distance
        self.sim.forward()
        return self.count_body_collisions() == 0

    def _reset_internal(self):
        self._invalid_early_laptop_open = False
        super()._reset_internal()

    def _setup_scene(self):
        super()._setup_scene()
        desk_distance = self.robot_fixture_collision_distance(
            self.desk_fixture,
            max_distance=self.desk_success_radius + 1.0,
        )
        if (
            not np.isfinite(desk_distance)
            or desk_distance < self.desk_success_radius + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: desk composite spawn is too close "
                f"({desk_distance:.3f} m; needs >= {self.desk_success_radius + 0.15:.3f} m)"
            )
        if self.window_body not in self.sim.model.body_names:
            raise RuntimeError(
                f"{self.scene_name}: calibrated rightmost window is missing: "
                f"{self.window_body}"
            )
        window_distance = HolodeckApproachWindow.success_distance(self)
        if window_distance < self.window_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: window composite spawn is too close "
                f"({window_distance:.3f} m; needs >= {self.window_success_radius + 0.15:.3f} m)"
            )

    @property
    def desk_success_radius(self):
        return float(self.navigation_spec["success_radius"])

    @property
    def window_success_radius(self):
        return float(self.window_spec["success_radius"])

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.desk_success_radius + 0.25,
            )
            <= self.desk_success_radius
        )

    def _laptop_opened(self):
        return bool(
            not getattr(self, "_invalid_early_laptop_open", False)
            and RoomOpenArticulation._raw_success(self)
        )

    def _post_action(self, action):
        if (
            self.current_atomic_task is not None
            and self.current_atomic_task.name == "approach_desk"
            and RoomOpenArticulation._raw_success(self)
        ):
            self._invalid_early_laptop_open = True
        return super()._post_action(action)

    def _begin_window_stage(self):
        self._window_stage_start_xy = self._robot_base_xy().copy()
        self._window_stage_moved = False

    def _window_reached_after_laptop(self):
        if np.linalg.norm(self._robot_base_xy() - self._window_stage_start_xy) >= 0.15:
            self._window_stage_moved = True
        return bool(
            self._window_stage_moved
            and HolodeckApproachWindow.success_distance(self)
            <= self.window_success_radius
        )

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, open the laptop, then walk to the rightmost window",
            target_fixture=self.desk_fixture.stable_id,
            target_joint=self.target_joint,
            target_window=self.window_body,
            success_definition=(
                "first reach the calibrated desk, open the specified laptop lid, "
                "then remain within the calibrated rightmost-window distance"
            ),
        )
        return meta


class HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan(
    HolodeckCloseLaptop,
    HolodeckMultiHopComposite,
):
    """Close the desk laptop, then walk to the trash can."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachDeskThenCloseLaptopThenApproachTrashCan"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197)
    SPEC_TASK = "close_laptop"
    NAVIGATION_SPEC_TASK = "approach_desk"
    TRASH_SPEC_TASK = "approach_trash_can"
    COMPOSITE_SPEC_TASK = "approach_desk_then_close_laptop_then_approach_trash_can"
    COMPOSITE_SCENE_IDS = (1197,)
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
            "_laptop_closed",
            task_cls=HolodeckCloseLaptop,
            hold_steps=RoomOpenArticulation.HOLD_STEPS,
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_can_reached_after_laptop",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=HolodeckApproachTrashCan.HOLD_STEPS,
            on_enter="_begin_trash_stage",
            metadata={"skill": "navigation", "target": "trash_can"},
        ),
    )

    def _uses_composite_spec(self):
        return self.scene_name in {
            f"train_{scene_id}" for scene_id in self.COMPOSITE_SCENE_IDS
        }

    def _setup_task_fixtures(self):
        if self._uses_composite_spec():
            # The composite has a scene-specific remote spawn.  Keep the
            # atomic close-laptop calibration for all other scenes.
            self.SPEC_TASK = self.COMPOSITE_SPEC_TASK
        super()._setup_task_fixtures()
        self.laptop_task_spec = self.task_spec
        spec_task = (
            self.COMPOSITE_SPEC_TASK
            if self._uses_composite_spec()
            else self.NAVIGATION_SPEC_TASK
        )
        self.navigation_spec = get_holodeck_task_spec(
            self.scene_name,
            spec_task,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        self.desk_fixture = self.get_fixture(id=self.navigation_spec["desk_body"])
        self.trash_spec = get_holodeck_task_spec(
            self.scene_name,
            spec_task if self._uses_composite_spec() else self.TRASH_SPEC_TASK,
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
        self.trash_can_xml_body = self.find_scene_body(self.trash_can_body)
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        # The composite starts at the desk for its first stage, but must begin
        # outside the final trash-can success region as well.  Keep that
        # second-stage spawn explicit in the scene calibration.
        pose = self.trash_spec.get("desk_composite_robot_pose")
        if pose is None:
            pose = self.trash_spec.get(
                "composite_robot_pose", self.navigation_spec["robot_pose"]
            )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.desk_fixture is not None
            and self.trash_can_xml_body is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        desk_distance = self.robot_fixture_collision_distance(
            self.desk_fixture,
            max_distance=self.desk_success_radius + 1.0,
        )
        if (
            not np.isfinite(desk_distance)
            or desk_distance < self.desk_success_radius + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: desk composite spawn is too close "
                f"({desk_distance:.3f} m; needs >= {self.desk_success_radius + 0.15:.3f} m)"
            )
        trash_distance = HolodeckApproachTrashCan.success_distance(self)
        if trash_distance < self.trash_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: trash-can composite spawn is too close "
                f"({trash_distance:.3f} m; needs >= {self.trash_success_radius + 0.15:.3f} m)"
            )

    def _place_robot_for_articulation(self, distance):
        del distance
        self.sim.forward()
        return self.count_body_collisions() == 0

    def _reset_internal(self):
        self._invalid_early_laptop_close = False
        super()._reset_internal()

    @property
    def desk_success_radius(self):
        return float(self.navigation_spec["success_radius"])

    @property
    def trash_success_radius(self):
        return float(self.trash_spec["success_radius"])

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _desk_reached(self):
        return bool(
            self.robot_fixture_collision_distance(
                self.desk_fixture,
                max_distance=self.desk_success_radius + 0.25,
            )
            <= self.desk_success_radius
        )

    def _laptop_closed(self):
        return bool(
            not getattr(self, "_invalid_early_laptop_close", False)
            and RoomOpenArticulation._raw_success(self)
        )

    def _post_action(self, action):
        if (
            self.current_atomic_task is not None
            and self.current_atomic_task.name == "approach_desk"
            and RoomOpenArticulation._raw_success(self)
        ):
            self._invalid_early_laptop_close = True
        return super()._post_action(action)

    def _begin_trash_stage(self):
        self._trash_stage_start_xy = self._robot_base_xy().copy()
        self._trash_stage_moved = False

    def _trash_can_reached_after_laptop(self):
        if np.linalg.norm(self._robot_base_xy() - self._trash_stage_start_xy) >= 0.15:
            self._trash_stage_moved = True
        return bool(
            self._trash_stage_moved
            and HolodeckApproachTrashCan.success_distance(self)
            <= self.trash_success_radius
        )

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the desk, close the laptop, then walk to the trash can",
            target_fixture=self.desk_fixture.stable_id,
            target_joint=self.target_joint,
            final_target_fixture=self.trash_can_body,
            success_definition=(
                "first reach the calibrated desk, close the specified laptop lid, "
                "then remain within the calibrated trash-can collision distance"
            ),
        )
        return meta


class HolodeckApproachTrashCanThenApproachDeskThenPullChair(
    HolodeckApproachDeskThenPullChair
):
    """Reach the trash can, return to the desk, then pull out its chair."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenApproachDeskThenPullChair"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (117, 1018)
    TRASH_SPEC_TASK = "approach_trash_can"
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
            "_desk_reached_after_trash_can",
            task_cls=HolodeckApproachDesk,
            hold_steps=HolodeckApproachDesk.HOLD_STEPS,
            on_enter="_begin_desk_stage_after_trash_can",
            metadata={"skill": "navigation", "target": "desk"},
        ),
        HolodeckAtomicTask(
            "pull_chair_from_desk",
            "_chair_pulled_after_approach",
            task_cls=HolodeckPullChairFromDesk,
            hold_steps=None,
            on_enter="_begin_pull_stage",
            metadata={"skill": "pull", "object": "chair"},
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
        # The pull calibration identifies the desk that actually owns the
        # chair. Use that same fixture for the preceding desk stage.
        self.navigation_desk_fixture = self.desk_fixture
        self.fixture_refs["approach_desk"] = self.navigation_desk_fixture
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
        # The atomic trash-can pose is close enough to the desk for this
        # composite's return-to-desk stage to begin inside the desk margin.
        # Keep that calibration for the atomic task, but use its explicit
        # remote override when one is provided for the composite.
        pose = self.trash_spec.get(
            "composite_robot_pose", self.trash_spec["robot_pose"]
        )
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _setup_scene(self):
        super()._setup_scene()
        distance = HolodeckApproachTrashCan.success_distance(self)
        if distance < self.trash_success_radius + 0.15:
            raise RuntimeError(
                f"{self.scene_name}: trash-can composite spawn is too close "
                f"({distance:.3f} m; needs >= {self.trash_success_radius + 0.15:.3f} m)"
            )

    @property
    def trash_success_radius(self):
        return float(self.trash_spec["success_radius"])

    def _trash_can_reached(self):
        return bool(
            HolodeckApproachTrashCan.success_distance(self) <= self.trash_success_radius
        )

    def _begin_desk_stage_after_trash_can(self):
        self._desk_stage_start_xy = self._robot_base_xy().copy()
        self._desk_stage_moved = False

    def _desk_reached_after_trash_can(self):
        if np.linalg.norm(self._robot_base_xy() - self._desk_stage_start_xy) >= 0.15:
            self._desk_stage_moved = True
        return bool(self._desk_stage_moved and self._desk_reached())

    def _check_success(self):
        return HolodeckMultiHopComposite._check_success(self)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="walk to the trash can, walk to the desk, then pull the chair out",
            success_definition=(
                "first reach the calibrated trash can, then move to the calibrated "
                f"desk, and finally pull its chair at least "
                f"{self.MIN_PULL_DISPLACEMENT:.2f} m from its reset position "
                "and keep it there for 1 second"
            ),
        )
        return meta


__all__ = [
    "HolodeckIceCubeThenPlaceCup",
    "HolodeckApproachDeskThenPullChair",
    "HolodeckApproachWindowThenDeskThenPickCup",
    "HolodeckApproachDeskIceCubePlaceCupThenPickCup",
    "HolodeckApproachTrashCanThenThrowBottle",
    "HolodeckApproachWindowThenDesk",
    "HolodeckApproachTrashCanThenThrowBottleThenApproachDesk",
    "HolodeckApproachDeskThenOpenLaptopThenApproachWindow",
    "HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan",
    "HolodeckApproachTrashCanThenApproachDeskThenPullChair",
]
