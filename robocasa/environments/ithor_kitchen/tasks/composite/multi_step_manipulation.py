"""Ordered kitchen tasks combining navigation, placement, and articulation."""

from __future__ import annotations

import os

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    IThorKitchenEnv,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
    get_ithor_kitchen_task_specs,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion
from robocasa.utils import object_utils as OU


_FRIDGE_WORKFLOW_SCENES = (13, 24, 25)
_DRAWER_CYCLE_SCENES = (13, 24, 25)
_CABINET_CYCLE_SCENES = (6, 7, 13, 24, 25)
_BOWL_TABLE_SCENES = (13, 24)
_STATUE_CABINET_SCENES = (7, 13)


class _OrderedKitchenWorkflowMixin(IThorMultiStageCompositeMixin):
    """Shared contracts for sequential multi-step kitchen tasks."""

    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_ORDERING = "sequential"
    SUCCESS_HOLD_SECONDS = 1.0
    INTERMEDIATE_HOLD_STEPS = 5
    NAVIGATION_DISTANCE = 0.35
    INITIAL_POSE_JITTER = False
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def _get_task_spec(self):
        key = (self.scene_name, self.SPEC_TASK)
        if getattr(self, "_multi_step_spec_key", None) != key:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=self.REQUIRED_SPEC_FIELDS,
            )
            self._multi_step_spec_key = key
        return self.task_spec

    def _required_stage_hold_steps(self, task):
        if self._atomic_tasks and task is self._atomic_tasks[-1]:
            seconds = float(self._get_task_spec()["hold_seconds"])
            return max(1, int(round(seconds * self.control_freq)))
        return int(task.hold_steps or self.INTERMEDIATE_HOLD_STEPS)

    def _fixture_contact(self, object_name, fixture):
        object_bodies = self._body_tree_ids([self.obj_body_id[object_name]])
        fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in object_bodies and body2 in fixture_bodies) or (
                body2 in object_bodies and body1 in fixture_bodies
            ):
                return True
        return False

    def _robot_near_fixture(self, fixture):
        distance = self.robot_fixture_collision_distance(
            fixture,
            max_distance=2.0,
        )
        threshold = float(
            self._get_task_spec().get("navigation_distance", self.NAVIGATION_DISTANCE)
        )
        return bool(np.isfinite(distance) and distance <= threshold)

    def _arm_condition_after_stage_entry(self, attribute, condition):
        """Require a later-stage condition to become false before it can pass."""

        if getattr(self, attribute, False):
            return bool(condition)
        if not condition:
            setattr(self, attribute, True)
        return False

    def _reset_composite_progress(self):
        super()._reset_composite_progress()
        self._stage_condition_armed = False

    def _arm_active_stage_condition(self):
        self._stage_condition_armed = False

    def _active_stage_condition(self, condition):
        return self._arm_condition_after_stage_entry(
            "_stage_condition_armed", bool(condition)
        )

    def _get_task_lang(self):
        return self.TASK_LANG

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self._get_task_lang(),
            composite_stages=[task.name for task in self._atomic_tasks],
            pickup_is_success_condition=False,
        )
        return meta


class _OrderedArticulationWorkflowMixin(_OrderedKitchenWorkflowMixin):
    """Shared exact-fixture setup for workflows with one articulated target."""

    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)

    def _finalize_articulation_robot_pose(self, _distance):
        # These composites deliberately start away from the articulation target.
        # The room reset performs its standard spawn-clutter clearing after all
        # task objects are placed; collision and Floor-footprint audits therefore
        # belong to that completed reset state rather than this early hook.
        self.sim.forward()
        return True

    def _articulation_is_open(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            >= float(self._get_task_spec()["success_open_fraction"])
        )

    def _articulation_is_closed(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            <= float(self._get_task_spec()["success_close_fraction"])
        )

    def _robot_near_articulation(self):
        return self._robot_near_fixture(self.target_fixture)

    def _close_active_stage_condition(self):
        return self._active_stage_condition(self._articulation_is_closed())


class IThorKitchenStoreAppleInFridge(
    _OrderedArticulationWorkflowMixin,
    SupportContactPlacementMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    """Carry a table apple to the fridge, place it inside, and close the door."""

    ATOMIC_SKILL = "StoreAppleInFridge"
    COMPOSITE_KIND = "navigate_open_place_close"
    SPEC_TASK = "store_apple_in_fridge"
    TASK_LANG = 'take the apple from the table, walk to the fridge, open the fridge door, put the apple inside, then close the fridge door'
    SUPPORTED_SCENE_IDS = _FRIDGE_WORKFLOW_SCENES
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Fridge",)
    TARGET_LANG = "fridge door"
    TARGET_JOINT_ROLES = ("door",)
    REQUIRED_SPEC_FIELDS = (
        "hold_seconds",
        "robot_pose",
        "target_fixture",
        "target_joint",
        "init_open_fraction",
        "success_open_fraction",
        "success_close_fraction",
        "fridge_receptacle_region",
        "apple_pos",
    )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "navigate_to_fridge",
                "_robot_near_articulation",
                task_cls="RootXYNavigation",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            ),
            IThorAtomicTask(
                "open_fridge",
                "_open_active_stage_condition",
                task_cls="IThorKitchenOpenFridge",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
                on_enter="_arm_active_stage_condition",
            ),
            IThorAtomicTask(
                "put_apple_in_fridge",
                "_apple_in_fridge_active_stage",
                task_cls="IThorKitchenAppleInFridge",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
                on_enter="_arm_active_stage_condition",
            ),
            IThorAtomicTask(
                "close_fridge",
                "_fridge_closed_with_apple",
                task_cls="IThorKitchenCloseFridge",
                on_enter="_arm_active_stage_condition",
            ),
        )

    def _get_obj_cfgs(self):
        return [dict(name="apple", obj_groups="apple")]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualStoreAppleInFridgeSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        region_id = self._get_task_spec()["fridge_receptacle_region"]
        self.fridge_receptacle_region = next(
            (
                region
                for region in self.get_region(return_all=True)
                if region.stable_id == region_id
            ),
            None,
        )
        if self.fridge_receptacle_region is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: unknown fridge region "
                f"{region_id!r}"
            )
        if self.fridge_receptacle_region.fixture_id != self.target_fixture.stable_id:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: fridge region does not "
                "belong to target_fixture"
            )
        self.region_refs["fridge_receptacle"] = self.fridge_receptacle_region

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_object_bottom_at(
            "apple",
            self._get_task_spec()["apple_pos"][:2],
            self._get_task_spec()["apple_pos"][2],
        )

    def _apple_in_fridge(self):
        return bool(
            self.object_in_region("apple", self.fridge_receptacle_region)
            and self._fixture_contact("apple", self.target_fixture)
            and self.object_released("apple")
        )

    def _open_active_stage_condition(self):
        return self._active_stage_condition(self._articulation_is_open())

    def _apple_in_fridge_active_stage(self):
        return self._active_stage_condition(self._apple_in_fridge())

    def _fridge_closed_with_apple(self):
        return bool(self._close_active_stage_condition() and self._apple_in_fridge())


class IThorKitchenOpenThenCloseDrawer(
    _OrderedArticulationWorkflowMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    """Open the configured drawer and then close that same drawer."""

    ATOMIC_SKILL = "OpenThenCloseDrawer"
    DIFFICULTY_LEVEL = "L2"
    COMPOSITE_KIND = "articulation_cycle"
    SPEC_TASK = "drawer_open_close_cycle"
    TASK_LANG = 'open the drawer, then close the same drawer'
    SUPPORTED_SCENE_IDS = _DRAWER_CYCLE_SCENES
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Drawer",)
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    REQUIRED_SPEC_FIELDS = (
        "hold_seconds",
        "robot_pose",
        "target_fixture",
        "target_joint",
        "init_open_fraction",
        "success_open_fraction",
        "success_close_fraction",
    )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "open_drawer",
                "_articulation_is_open",
                task_cls="IThorKitchenOpenDrawer",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            ),
            IThorAtomicTask(
                "close_drawer",
                "_close_active_stage_condition",
                task_cls="IThorKitchenCloseDrawer",
                on_enter="_arm_active_stage_condition",
            ),
        )


class _CabinetArticulationCycleMixin(_OrderedArticulationWorkflowMixin):
    """Ordered open/close workflow for the calibrated kitchen-scene cabinet."""

    TARGET_CATEGORIES = ("Cabinet",)
    TARGET_LANG = "cabinet door"
    TARGET_JOINT_ROLES = ("door",)
    REQUIRED_SPEC_FIELDS = (
        "hold_seconds", "robot_pose", "target_fixture", "target_joint",
        "init_open_fraction", "success_open_fraction", "success_close_fraction",
    )


class IThorKitchenOpenThenCloseCabinet(
    _CabinetArticulationCycleMixin, RoomOpenArticulation, IThorKitchenEnv,
):
    """Open the cabinet door to 80%, then close it to 20%."""

    ATOMIC_SKILL = "OpenThenCloseCabinet"
    COMPOSITE_KIND = "articulation_cycle"
    SPEC_TASK = "cabinet_open_close_cycle"
    TASK_LANG = "open the cabinet door, then close the same door"
    SUPPORTED_SCENE_IDS = _CABINET_CYCLE_SCENES
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    BEHAVIOR = "open"

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask("open_cabinet", "_articulation_is_open", task_cls="IThorKitchenOpenCabinet"),
            IThorAtomicTask("close_cabinet", "_close_active_stage_condition", task_cls="IThorKitchenCloseCabinet", on_enter="_arm_active_stage_condition"),
        )


class IThorKitchenCloseThenOpenCabinet(
    _CabinetArticulationCycleMixin, RoomOpenArticulation, IThorKitchenEnv,
):
    """Close the cabinet door to 20%, then open it to 80%."""

    ATOMIC_SKILL = "CloseThenOpenCabinet"
    COMPOSITE_KIND = "articulation_cycle"
    SPEC_TASK = "cabinet_close_open_cycle"
    TASK_LANG = "close the cabinet door, then open the same door"
    SUPPORTED_SCENE_IDS = _CABINET_CYCLE_SCENES
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    BEHAVIOR = "close"

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask("close_cabinet", "_close_active_stage_condition", task_cls="IThorKitchenCloseCabinet"),
            IThorAtomicTask("open_cabinet", "_active_stage_condition_open", task_cls="IThorKitchenOpenCabinet", on_enter="_arm_active_stage_condition"),
        )

    def _active_stage_condition_open(self):
        return self._active_stage_condition(self._articulation_is_open())


class IThorKitchenBowlToSinkThenAppleOnPlate(
    _OrderedKitchenWorkflowMixin,
    SupportContactPlacementMixin,
    IThorKitchenEnv,
):
    """Put a bowl in the sink, walk to the table, and plate an apple."""

    ATOMIC_SKILL = "BowlToSinkThenAppleOnPlate"
    COMPOSITE_KIND = "place_navigate_place"
    SPEC_TASK = "bowl_to_sink_then_apple_on_plate"
    TASK_LANG = 'put the bowl in the sink, then walk to the table and put the apple on the plate'
    SUPPORTED_SCENE_IDS = _BOWL_TABLE_SCENES
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    BOWL_XML = "models/assets/objects/objaverse/bowl/bowl_1/model.xml"
    BOWL_SCALE = 0.65
    APPLE_ON_PLATE_XY_TOL = 0.13
    REQUIRED_SPEC_FIELDS = (
        "hold_seconds",
        "robot_pose",
        "bowl_pos",
        "sink_fixture",
        "sink_region",
        "table_fixture",
        "plate_pos",
        "apple_pos",
    )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "put_bowl_in_sink",
                "_bowl_in_sink",
                task_cls="IThorKitchenBowlToSink",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            ),
            IThorAtomicTask(
                "navigate_to_table",
                "_table_navigation_active_stage",
                task_cls="RootXYNavigation",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
                on_enter="_arm_active_stage_condition",
            ),
            IThorAtomicTask(
                "put_apple_on_plate",
                "_apple_on_plate_with_bowl_retained",
                task_cls="IThorKitchenAppleOnPlate",
                on_enter="_arm_active_stage_condition",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.sink_fixture = self.get_fixture(id=spec["sink_fixture"])
        self.table_fixture = self.get_fixture(id=spec["table_fixture"])
        if self.sink_fixture is None or self.table_fixture is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing configured sink/table"
            )
        region = spec["sink_region"]
        center = np.asarray(region["center"], dtype=float)
        half_size = np.asarray(region["half_size"], dtype=float)
        if center.shape != (3,) or half_size.shape != (3,) or np.any(half_size <= 0):
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid sink_region")
        self.sink_region = SemanticRegion(
            stable_id=f"{self.sink_fixture.stable_id}:region:receptacle:workflow",
            fixture_id=self.sink_fixture.stable_id,
            kind="receptacle",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=half_size,
            _world_center=center,
            _world_yaw=float(region.get("yaw", 0.0)),
        )
        self.fixture_refs.update(sink=self.sink_fixture, table=self.table_fixture)
        self.region_refs["sink_receptacle"] = self.sink_region

    def _get_obj_cfgs(self):
        bowl_xml = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.BOWL_XML)
        )
        return [
            dict(name="bowl", obj_groups=bowl_xml, object_scale=self.BOWL_SCALE),
            dict(name="plate", obj_groups="plate"),
            dict(name="apple", obj_groups="apple"),
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualBowlSinkApplePlateSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        for name, key in (
            ("bowl", "bowl_pos"),
            ("plate", "plate_pos"),
            ("apple", "apple_pos"),
        ):
            pos = np.asarray(spec[key], dtype=float)
            if pos.shape != (3,):
                raise ValueError(
                    f"{self.scene_name}/{self.SPEC_TASK}: {key} must contain XYZ"
                )
            self._set_task_object_bottom_at(name, pos[:2], pos[2])

    def _bowl_in_sink(self):
        return bool(
            self.object_in_region("bowl", self.sink_region, rim_tol=0.04, xy_tol=0.03)
            and self._fixture_contact("bowl", self.sink_fixture)
            and self.object_released("bowl")
        )

    def _robot_near_table(self):
        return self._robot_near_fixture(self.table_fixture)

    def _table_navigation_active_stage(self):
        return self._active_stage_condition(self._robot_near_table())

    def _apple_on_plate(self):
        return bool(
            OU.check_obj_in_receptacle(
                self,
                "apple",
                "plate",
                th=self.APPLE_ON_PLATE_XY_TOL,
            )
            and self.object_released("apple")
        )

    def _apple_on_plate_with_bowl_retained(self):
        return bool(
            self._active_stage_condition(self._apple_on_plate())
            and self._bowl_in_sink()
        )


class IThorKitchenApproachCabinetAndStoreStatue(
    _OrderedArticulationWorkflowMixin,
    SupportContactPlacementMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    """Walk to an open cabinet, store the table statue, and close its door."""

    ATOMIC_SKILL = "ApproachCabinetAndStoreStatue"
    COMPOSITE_KIND = "navigate_place_close"
    SPEC_TASK = "approach_cabinet_and_store_statue"
    TASK_LANG = 'walk to the cabinet, take the statue from the table, put it inside the open cabinet, then close the cabinet door'
    SUPPORTED_SCENE_IDS = _STATUE_CABINET_SCENES
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Cabinet",)
    TARGET_LANG = "cabinet door"
    TARGET_JOINT_ROLES = ("door",)
    STATUE_SCALE = 0.30
    STATUE_XML = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../../models/assets/objects/thor/Miscellaneous Objects/Statue/"
            "Prefabs/Stone_Statue_4/Stone_Statue_4.xml",
        )
    )
    REQUIRED_SPEC_FIELDS = (
        "hold_seconds",
        "robot_pose",
        "target_fixture",
        "target_joint",
        "init_open_fraction",
        "success_close_fraction",
        "cabinet_receptacle_region",
        "statue_pos",
        "statue_quat",
    )

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "navigate_to_cabinet",
                "_robot_near_articulation",
                task_cls="RootXYNavigation",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            ),
            IThorAtomicTask(
                "put_statue_in_cabinet",
                "_statue_in_cabinet_active_stage",
                task_cls="IThorKitchenStatueInCabinet",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
                on_enter="_arm_active_stage_condition",
            ),
            IThorAtomicTask(
                "close_cabinet",
                "_cabinet_closed_with_statue",
                task_cls="IThorKitchenCloseCabinet",
                on_enter="_arm_active_stage_condition",
            ),
        )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        region_id = self._get_task_spec()["cabinet_receptacle_region"]
        self.cabinet_receptacle_region = next(
            (
                region
                for region in self.get_region(return_all=True)
                if region.stable_id == region_id
            ),
            None,
        )
        if self.cabinet_receptacle_region is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: unknown cabinet region "
                f"{region_id!r}"
            )
        self.cabinet_receptacle_fixture = self.get_fixture(
            id=self.cabinet_receptacle_region.fixture_id
        )
        if self.cabinet_receptacle_fixture is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing cabinet receptacle fixture"
            )
        self.region_refs["cabinet_receptacle"] = self.cabinet_receptacle_region

    def _get_obj_cfgs(self):
        return [dict(name="statue", obj_groups=self.STATUE_XML)]

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        statue = LivingRoomStatueObject(
            name="statue",
            source_xml=self.STATUE_XML,
            scale=self.STATUE_SCALE,
        )
        self.objects[statue.name] = statue
        self.model.merge_objects([statue])

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCabinetStatueSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        pos = np.asarray(spec["statue_pos"], dtype=float)
        quat = np.asarray(spec["statue_quat"], dtype=float)
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid statue pose")
        joint = self.objects["statue"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = pos
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        qpos[2] += pos[2] - self.object_bottom_z("statue")
        self.sim.data.set_joint_qpos(joint, qpos)
        joint_id = self.sim.model.joint_name2id(joint)
        dof = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof : dof + 6] = 0.0
        self.sim.forward()

    def _statue_in_cabinet(self):
        return bool(
            self.object_in_region("statue", self.cabinet_receptacle_region)
            and self._fixture_contact("statue", self.cabinet_receptacle_fixture)
            and self.object_released("statue")
        )

    def _statue_in_cabinet_active_stage(self):
        return self._active_stage_condition(self._statue_in_cabinet())

    def _cabinet_closed_with_statue(self):
        return bool(self._close_active_stage_condition() and self._statue_in_cabinet())


# Only concrete workflow classes are public and runnable.
for _implementation_class in (
    _OrderedKitchenWorkflowMixin,
    _OrderedArticulationWorkflowMixin,
    _CabinetArticulationCycleMixin,
):
    REGISTERED_ITHOR_KITCHEN_ENVS.pop(_implementation_class.__name__, None)
    REGISTERED_ENVS.pop(_implementation_class.__name__, None)


__all__ = [
    "IThorKitchenStoreAppleInFridge",
    "IThorKitchenOpenThenCloseDrawer",
    "IThorKitchenOpenThenCloseCabinet",
    "IThorKitchenCloseThenOpenCabinet",
    "IThorKitchenBowlToSinkThenAppleOnPlate",
    "IThorKitchenApproachCabinetAndStoreStatue",
]
