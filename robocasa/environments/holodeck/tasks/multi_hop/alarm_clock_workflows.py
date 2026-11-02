"""Ordered clock/drawer workflows scoped to Holodeck scene 2209."""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.holodeck.tasks.alarm_clock import (
    HolodeckPlaceAlarmClockInDrawer,
)
from robocasa.environments.holodeck.tasks.approach_trash_can import (
    HolodeckApproachTrashCan,
)
from robocasa.environments.holodeck.tasks.articulation import (
    HolodeckCloseDrawer,
    HolodeckOpenDrawer,
)
from robocasa.environments.holodeck.tasks.pick_box_from_floor import (
    HolodeckPickBoxFromFloor,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.tasks.multi_hop.plant_to_trash import (
    HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan,
)
from robocasa.environments.holodeck.tasks.throw_bottle_into_trash_can import (
    HolodeckThrowBottleIntoTrashCan,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckClockDrawerWorkflow(
    HolodeckMultiHopComposite, HolodeckPlaceAlarmClockInDrawer
):
    """Reuse the atomic clock layout without changing its scene calibration."""

    ABSTRACT = True
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (2209,)
    SPEC_TASK = HolodeckPlaceAlarmClockInDrawer.SPEC_TASK
    OPEN_FRACTION = 0.80
    CLOSE_FRACTION = 0.20

    def _setup_baked_scene_fixtures(self):
        self.workflow_spec = get_holodeck_task_spec(
            self.scene_name,
            self.WORKFLOW_SPEC_TASK,
            required=("drawer_initial_fraction",),
        )
        super()._setup_baked_scene_fixtures()

    def _setup_references(self):
        super()._setup_references()
        joint_id = self.sim.model.joint_name2id(self.task_spec["drawer_joint"])
        self._drawer_range = self.sim.model.jnt_range[joint_id].copy()

    def _drawer_fraction(self):
        lower, upper = self._drawer_range
        qpos = float(self.sim.data.get_joint_qpos(self.task_spec["drawer_joint"]))
        return (qpos - lower) / (upper - lower)

    def _reset_task_progress(self):
        lower, upper = self._drawer_range
        qpos = lower + self.workflow_spec["drawer_initial_fraction"] * (upper - lower)
        self.sim.data.set_joint_qpos(self.task_spec["drawer_joint"], qpos)
        self.sim.data.set_joint_qvel(self.task_spec["drawer_joint"], 0.0)
        self.sim.forward()
        self._clock_stage_armed = False
        self._close_stage_armed = False
        super()._reset_task_progress()

    def _clock_placed(self):
        return HolodeckPlaceAlarmClockInDrawer._goal_satisfied(self)

    def _begin_clock_stage(self):
        self._clock_stage_armed = not self._clock_placed()

    def _clock_placed_in_order(self):
        placed = self._clock_placed()
        if not placed:
            self._clock_stage_armed = True
        return bool(
            self._clock_stage_armed
            and placed
            and self._drawer_fraction() >= self.OPEN_FRACTION
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(lang=self.LANG, success_definition=self.SUCCESS_DEFINITION)
        return meta


class HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox(_HolodeckClockDrawerWorkflow):
    """Place the clock, close its drawer, then lift the floor box."""

    ABSTRACT = False
    ATOMIC_SKILL = "PlaceAlarmClockThenCloseDrawerThenPickBox"
    WORKFLOW_SPEC_TASK = "place_alarm_clock_then_close_drawer_then_pick_box"
    LANG = "Put the alarm clock into the open top desk drawer, close the drawer, then pick up the cardboard box from the floor."
    SUCCESS_DEFINITION = (
        "contact the named drawer floor with the clock for 1 second while the "
        "drawer is at least 80% open, then close to at most 20% for 1 second, "
        "then lift the floor box more than 0.30 m for 1 second; the clock must "
        "stay on the drawer floor and the drawer must remain closed"
    )
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "place_alarm_clock",
            "_clock_placed_in_order",
            task_cls=HolodeckPlaceAlarmClockInDrawer,
            on_enter="_begin_clock_stage",
        ),
        HolodeckAtomicTask(
            "close_drawer",
            "_drawer_closed_in_order",
            task_cls=HolodeckCloseDrawer,
            on_enter="_begin_close_stage",
        ),
        HolodeckAtomicTask(
            "pick_box",
            "_box_lifted_in_order",
            task_cls=HolodeckPickBoxFromFloor,
            on_enter="_begin_box_stage",
        ),
    )
    BOX_ASSET = HolodeckPickBoxFromFloor.BOX_ASSET
    BOX_SCALE = HolodeckPickBoxFromFloor.BOX_SCALE
    BOX_HALF_SIZE = HolodeckPickBoxFromFloor.BOX_HALF_SIZE
    _box_root_z = HolodeckPickBoxFromFloor._box_root_z
    _normalized_room_id = staticmethod(HolodeckPickBoxFromFloor._normalized_room_id)

    def _create_objects(self):
        super()._create_objects()
        path = HolodeckPickBoxFromFloor._make_box_xml()
        try:
            box = MJCFObject(name="box", mjcf_path=path, scale=self.BOX_SCALE)
            box._regions["bbox"] = {
                "elem": ET.Element(
                    "geom",
                    pos="0 0 0",
                    size=" ".join(str(v * self.BOX_SCALE) for v in self.BOX_HALF_SIZE),
                )
            }
        finally:
            os.remove(path)
        self.objects["box"] = box
        self.model.merge_objects([box])

    def _reset_task_progress(self):
        clock_spec = self.task_spec
        self.task_spec = self.workflow_spec
        try:
            HolodeckPickBoxFromFloor._set_box_pose(self)
        finally:
            self.task_spec = clock_spec
        # The atomic helper includes the prefab's oversized invisible bbox in
        # its grounding envelope. Ground this composite on physical boxes only.
        box = self.objects["box"]
        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        qpos[2] -= self._box_bottom_z()
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        self._box_initial_z = self._box_root_z()
        super()._reset_task_progress()

    def _box_bottom_z(self):
        model, data = self.sim.model, self.sim.data
        bodies = self._body_tree_ids(
            [model.body_name2id(self.objects["box"].root_body)]
        )
        geoms = self._collidable_geoms(bodies)
        bottoms = []
        for geom in geoms:
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise RuntimeError("Box_19 must use box collision primitives")
            rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
            extent = np.abs(rotation[2]) @ model.geom_size[geom]
            bottoms.append(float(data.geom_xpos[geom, 2] - extent))
        return min(bottoms)

    def _begin_close_stage(self):
        self._close_stage_armed = self._drawer_fraction() >= self.OPEN_FRACTION

    def _drawer_closed_in_order(self):
        fraction = self._drawer_fraction()
        if fraction >= self.OPEN_FRACTION:
            self._close_stage_armed = True
        return bool(
            self._close_stage_armed
            and fraction <= self.CLOSE_FRACTION
            and self._clock_placed()
        )

    def _box_lifted(self):
        return (
            self._box_root_z()
            > self._box_initial_z + HolodeckPickBoxFromFloor.MIN_LIFT_DELTA
        )

    def _begin_box_stage(self):
        self._box_stage_armed = not self._box_lifted()

    def _box_lifted_in_order(self):
        lifted = self._box_lifted()
        if not lifted:
            self._box_stage_armed = True
        return bool(self._box_stage_armed and lifted and self._drawer_closed_in_order())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            source_object="box", box_model=self.BOX_ASSET, box_scale=self.BOX_SCALE
        )
        return meta


class HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan(
    _HolodeckClockDrawerWorkflow
):
    """Open the top drawer, place the clock, then walk to the bin."""

    ABSTRACT = False
    ATOMIC_SKILL = "OpenDrawerThenPlaceAlarmClockThenApproachTrashCan"
    WORKFLOW_SPEC_TASK = "open_drawer_then_place_alarm_clock_then_approach_trash_can"
    LANG = "Open the top desk drawer, put the alarm clock into it, then walk to the trash can and stand nearby."
    SUCCESS_DEFINITION = (
        "open the specified drawer from 30% to at least 80% for 1 second, "
        "then contact its floor with the clock for 1 second, then move at "
        "least 0.15 m and stand within 0.35 m collision distance of the trash "
        "can for 1 second while the clock stays on the drawer floor; early "
        "clock placement must be undone before the placement stage"
    )
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "open_drawer", "_drawer_opened", task_cls=HolodeckOpenDrawer
        ),
        HolodeckAtomicTask(
            "place_alarm_clock",
            "_clock_placed_in_order",
            task_cls=HolodeckPlaceAlarmClockInDrawer,
            on_enter="_begin_clock_stage",
        ),
        HolodeckAtomicTask(
            "approach_trash_can",
            "_trash_reached_in_order",
            task_cls=HolodeckApproachTrashCan,
            on_enter="_begin_trash_stage",
        ),
    )
    # Copy the existing scene-117 bin only into this composite's in-memory XML.
    TRASH_CAN_SOURCE_BODY = (
        HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan.TRASH_CAN_SOURCE_BODY
    )
    TRASH_CAN_PREFIX = "holodeck2209_clock_workflow"
    _robot_base_xy = HolodeckApproachTrashCan._robot_base_xy

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        self.trash_can_body = f"{self.TRASH_CAN_PREFIX}_trash_can"
        HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan._add_scene_117_trash_can(
            self
        )
        HolodeckThrowBottleIntoTrashCan._configure_trash_can_collision(self)

    def _drawer_opened(self):
        return self._drawer_fraction() >= self.OPEN_FRACTION

    def _begin_trash_stage(self):
        self._trash_start_xy = self._robot_base_xy().copy()
        self._trash_stage_armed = (
            self._trash_distance() > self.workflow_spec["success_radius"]
        )

    def _trash_distance(self):
        return HolodeckApproachTrashCan.success_distance(self)

    def _trash_reached_in_order(self):
        distance = self._trash_distance()
        if distance > self.workflow_spec["success_radius"]:
            self._trash_stage_armed = True
        return bool(
            self._trash_stage_armed
            and np.linalg.norm(self._robot_base_xy() - self._trash_start_xy) >= 0.15
            and distance <= self.workflow_spec["success_radius"]
            and self._clock_placed()
        )

    def _required_stage_hold_steps(self, task):
        if task.name == "approach_trash_can":
            return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))
        return super()._required_stage_hold_steps(task)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            trash_can_body=self.trash_can_body,
            trash_can_source_scene="train_117",
            success_radius=self.workflow_spec["success_radius"],
        )
        return meta


__all__ = [
    "HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox",
    "HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan",
]
