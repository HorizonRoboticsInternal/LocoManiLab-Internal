"""Approach the bin beside scene 1477's plant, then lift and place the plant."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.holodeck.holodeck import get_holodeck_scenes_dir
from robocasa.environments.holodeck.tasks.approach_trash_can import (
    HolodeckApproachTrashCan,
)
from robocasa.environments.holodeck.tasks.lift_plant_from_low_cabinet import (
    HolodeckLiftPlantFromLowCabinet,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.tasks.throw_bottle_into_trash_can import (
    HolodeckThrowBottleIntoTrashCan,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan(
    HolodeckMultiHopComposite, HolodeckLiftPlantFromLowCabinet
):
    """Walk to the plant-side bin, lift the plant, and place it inside."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUPPORTED_SCENE_IDS = (1477,)
    SUCCESS_DEBUG_ENABLED = False
    SUCCESS_DEBUG_INTERVAL_SECONDS = 1.0
    # Reuse the atomic plant's source, free body, scale, XY, and orientation.
    SPEC_TASK = HolodeckLiftPlantFromLowCabinet.SPEC_TASK
    WORKFLOW_SPEC_TASK = (
        "approach_trash_can_then_lift_plant_then_place_plant_in_trash_can"
    )
    TRASH_CAN_SOURCE_BODY = "holodeck117_trash_can"
    TRASH_CAN_PREFIX = "holodeck1477_plant_workflow"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_trash_can",
            "_bin_reached",
            task_cls=HolodeckApproachTrashCan,
            hold_steps=5,
            metadata={"skill": "navigation", "target": "plant_side_trash_can"},
        ),
        HolodeckAtomicTask(
            "lift_plant",
            "_plant_lifted_in_order",
            task_cls=HolodeckLiftPlantFromLowCabinet,
            on_enter="_begin_lift_stage",
            metadata={"skill": "lift", "object": "potted_plant"},
        ),
        HolodeckAtomicTask(
            "place_plant_in_trash_can",
            "_plant_in_bin",
            on_enter="_begin_release_stage",
            metadata={"skill": "place", "target": "plant_side_trash_can"},
        ),
    )

    def _get_robot_base_pose_config(self):
        return self.workflow_spec["robot_pose"]

    def _setup_baked_scene_fixtures(self):
        self.workflow_spec = get_holodeck_task_spec(
            self.scene_name,
            self.WORKFLOW_SPEC_TASK,
            required=(
                "trash_can_body",
                "bin_pos",
                "bin_inner_min",
                "bin_inner_max",
                "success_radius",
                "robot_pose",
            ),
        )
        self.trash_can_body = self.workflow_spec["trash_can_body"]
        if self.find_scene_body(self.trash_can_body) is not None:
            raise RuntimeError(f"{self.scene_name}: task bin already exists")
        self._add_scene_117_trash_can()
        HolodeckThrowBottleIntoTrashCan._configure_trash_can_collision(self)
        super()._setup_baked_scene_fixtures()

    def _add_scene_117_trash_can(self):
        """Reuse scene 117's actual bin meshes, materials, scale, and orientation."""
        source_path = Path(get_holodeck_scenes_dir()) / "train_117.robocasa.xml"
        source = ET.parse(source_path).getroot()
        body = deepcopy(
            source.find(f"./worldbody/body[@name='{self.TRASH_CAN_SOURCE_BODY}']")
        )
        if body is None:
            raise RuntimeError("train_117: source trash can is missing")
        assets = {item.get("name"): item for item in source.findall("./asset/*")}
        needed = set()
        pending = [body]
        while pending:
            for element in pending.pop().iter():
                for attribute in ("mesh", "material", "texture"):
                    name = element.get(attribute)
                    if name and name not in needed:
                        needed.add(name)
                        pending.append(assets[name])
        bin_model = ET.Element("mujoco")
        asset = ET.SubElement(bin_model, "asset")
        for item in source.findall("./asset/*"):
            if item.get("name") in needed:
                asset.append(deepcopy(item))
        for element in body.iter():
            name = element.get("name")
            if name:
                element.set(
                    "name", name.replace(self.TRASH_CAN_SOURCE_BODY, "trash_can")
                )
        ET.SubElement(bin_model, "worldbody").append(body)
        self._append_xml_fixture_model(
            bin_model,
            str(source_path.parent),
            self.TRASH_CAN_PREFIX,
            self.workflow_spec["bin_pos"],
            quat=None,
            ensure_inertials=False,
        )

    def _setup_references(self):
        super()._setup_references()
        self._plant_bodies = self._body_tree_ids([self._target_body_id()])
        self._plant_geoms = self._collidable_geoms(self._plant_bodies)
        self._bin_floor_geom = self.sim.model.geom_name2id(
            f"{self.trash_can_body}_inner_floor"
        )

    def _set_target_pose(self):
        super()._set_target_pose()
        self.sim.forward()
        source = self.sim.model.body_name2id(self.task_spec["source_body"])
        source_geoms = self._collidable_geoms(self._body_tree_ids([source]))
        top = self._collision_points(source_geoms)[:, 2].max()
        bottom = self._plant_collision_points()[:, 2].min()
        # The inherited pose leaves a gap over this side table. Correct only
        # this workflow, with a 0.05 mm contact overlap and zero free velocity.
        joint = self._target_joint_name()
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[2] += top - bottom - 0.00005
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()
        if not self.check_contact(
            self._geom_names_for_bodies(self._plant_bodies),
            self._geom_names_for_bodies(self._body_tree_ids([source])),
        ):
            raise RuntimeError(
                f"{self.scene_name}: plant is not supported by side table"
            )

    def _setup_scene(self):
        super()._setup_scene()
        if not self._calibrate_robot_free_joint_height():
            raise RuntimeError(f"{self.scene_name}: cannot calibrate robot height")
        if not HolodeckThrowBottleIntoTrashCan._robot_footprint_is_in_room(self):
            raise RuntimeError(f"{self.scene_name}: robot starts off floor")
        if self.count_body_collisions() != 0:
            raise RuntimeError(f"{self.scene_name}: robot has spawn collision")
        if self.success_distance() <= self.workflow_spec["success_radius"] + 0.15:
            raise RuntimeError(f"{self.scene_name}: robot starts too close to the bin")

    def _reset_task_progress(self):
        super()._reset_task_progress()
        self._lift_stage_armed = False
        self._release_stage_ready = False
        self._plant_debug_last_state = None
        self._success_debug_next_log_time = 0.0

    def _robot_base_xy(self):
        return HolodeckApproachTrashCan._robot_base_xy(self)

    def success_distance(self):
        return HolodeckApproachTrashCan.success_distance(self)

    def _plant_lifted(self):
        return bool(self._plant_root_z() >= self._plant_initial_z + self.MIN_LIFT_DELTA)

    def _bin_reached(self):
        return self.success_distance() <= self.workflow_spec["success_radius"]

    def _begin_lift_stage(self):
        self._lift_stage_armed = not self._plant_lifted()

    def _plant_lifted_in_order(self):
        lifted = self._plant_lifted()
        if not lifted:
            self._lift_stage_armed = True
        return bool(self._lift_stage_armed and lifted)

    def _begin_release_stage(self):
        self._release_stage_ready = self._plant_lifted_in_order()

    def _plant_collision_points(self):
        return self._collision_points(self._plant_geoms)

    def _collision_points(self, geom_ids):
        signs = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        points = []
        for geom in geom_ids:
            aabb = self.sim.model.geom_aabb[geom]
            rot = self.sim.data.geom_xmat[geom].reshape(3, 3)
            points.extend(
                (aabb[:3] + signs * aabb[3:]) @ rot.T + self.sim.data.geom_xpos[geom]
            )
        return np.asarray(points)

    def _plant_contacts_floor(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if contact.geom1 == self._bin_floor_geom:
                other = contact.geom2
            elif contact.geom2 == self._bin_floor_geom:
                other = contact.geom1
            else:
                continue
            if int(self.sim.model.geom_bodyid[other]) in self._plant_bodies:
                return True
        return False

    def _plant_in_bin(self):
        if not self._release_stage_ready:
            return False
        points = self._plant_collision_points()
        origin = self.sim.data.get_body_xpos(self.trash_can_body)
        rotation = self.sim.data.get_body_xmat(self.trash_can_body)
        # This native bin is Y-up in body coordinates, with its original yaw.
        local = (points - origin) @ rotation
        return bool(
            np.all(local >= np.asarray(self.workflow_spec["bin_inner_min"]))
            and np.all(local <= np.asarray(self.workflow_spec["bin_inner_max"]))
            and self._plant_contacts_floor()
        )

    def _required_stage_hold_steps(self, task):
        if task.name == "place_plant_in_trash_can":
            return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))
        return super()._required_stage_hold_steps(task)

    def _check_success(self):
        return super()._check_success() and self._plant_in_bin()

    def _success_debug_details(self, task):
        lift = self._plant_root_z() - self._plant_initial_z
        distance = self.success_distance()
        origin = self.sim.data.get_body_xpos(self.trash_can_body)
        rotation = self.sim.data.get_body_xmat(self.trash_can_body)
        local = (self._plant_collision_points() - origin) @ rotation
        lower, upper = local.min(axis=0), local.max(axis=0)
        below = np.maximum(np.asarray(self.workflow_spec["bin_inner_min"]) - lower, 0)
        above = np.maximum(upper - np.asarray(self.workflow_spec["bin_inner_max"]), 0)
        floor_contact = self._plant_contacts_floor()
        reasons = {
            "approach_trash_can": [
                (distance <= self.workflow_spec["success_radius"], "离桶过远"),
            ],
            "lift_plant": [
                (self._lift_stage_armed, "需先放低提前举起的盆栽"),
                (lift >= self.MIN_LIFT_DELTA, "抬升不足"),
            ],
            "place_plant_in_trash_can": [
                (self._release_stage_ready, "未按顺序完成靠近垃圾桶与抬升"),
                (not np.any(below) and not np.any(above), "盆体超出桶内容纳范围"),
                (floor_contact, "盆栽未接触桶底"),
            ],
        }
        checks = reasons[task.name if task else "place_plant_in_trash_can"]
        blocked = "、".join(reason for ok, reason in checks if not ok)
        vector = lambda values: np.array2string(values, precision=4, separator=",")
        return (
            f"未满足={blocked or '无，等待保持计数/已完成'}；"
            f"抬升={lift:.3f}/{self.MIN_LIFT_DELTA:.3f}m；"
            f"到桶碰撞距离={distance:.3f}/{self.workflow_spec['success_radius']:.3f}m；"
            f"投放阶段就绪={int(self._release_stage_ready)}，桶底接触={int(floor_contact)}；"
            f"盆体局部min={vector(lower)},max={vector(upper)}；"
            f"下界越界={vector(below)},上界越界={vector(above)}m(X/Z水平,Y竖直)"
        )

    def _log_success_conditions(self):
        sim_time = float(self.sim.data.time)
        success = bool(self._check_success())
        state = (self.stage_index, success)
        changed = state != self._plant_debug_last_state
        if sim_time < self._success_debug_next_log_time and not changed:
            return
        self._plant_debug_last_state = state
        self._success_debug_next_log_time = (
            sim_time + self.SUCCESS_DEBUG_INTERVAL_SECONDS
        )
        task = self.current_atomic_task
        target = self._required_stage_hold_steps(task) if task else 0
        print(
            f"[plant1477-debug] 时间={sim_time:.2f}s "
            f"阶段={self.stage_index + 1 if task else len(self._atomic_tasks)}/"
            f"{len(self._atomic_tasks)}({task.name if task else '全部完成'}) "
            f"保持={self._stage_hold_steps}/{target} 任务成功={int(success)}；"
            f"{self._success_debug_details(task)}",
            flush=True,
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = (
            "walk to the trash can beside the potted plant, lift the plant "
            "from the side table, and place it into the trash can"
        )
        meta["source_object"] = self.task_spec["target_body"]
        meta["source_fixture"] = self.task_spec["source_body"]
        meta["target_fixture"] = self.trash_can_body
        meta["trash_can_asset_source"] = f"train_117:{self.TRASH_CAN_SOURCE_BODY}"
        meta["success_definition"] = (
            "in order: stand within 0.35 m collision distance of the plant-side "
            "bin for 5 steps; then lift the plant by 0.30 m for 1 s "
            "(an early lift must be lowered and repeated); pot fully inside the bin "
            "with bottom contact for 1 s; no hand-state checks at any stage"
        )
        return meta


__all__ = ["HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan"]
