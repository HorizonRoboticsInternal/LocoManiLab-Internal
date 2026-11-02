"""Place an ice cube into a glass cup."""

import os
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckIceCubeInGlass(HolodeckEnv):
    """Calibrated tabletop ice-in-glass placement task."""

    ATOMIC_SKILL = "PickPlaceInside"
    TASK_FAMILY = "pick_place"
    CUP_ASSET = "objects/lightwheel/glass_cup/GlassCup014/model.xml"
    ICE_ASSET = "objects/lightwheel/ice_cube/IceCube004/model.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "ice_cube_in_glass"
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    # This task owns its scene cleanup and removes the baked laptop from the
    # task spec where necessary. Do not apply Holodeck's default laptop layout
    # first, which would require removing the same source body a second time.
    USE_DEFAULT_ASSET_LAYOUT = False
    NOISE_FREE_ROBOT_SPAWN = True

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "table_body",
                "robot_pose",
                "object_scale",
                "cup_pose",
                "ice_pose",
            ),
        )
        table_body = self.task_spec["table_body"]
        # Objaverse exports may omit support capabilities on desks. Resolve
        # the calibrated surface by its exact stable body before falling back
        # to the semantic category query.
        semantic_fixtures = getattr(self.semantic_scene, "fixtures", {}).values()
        self.table_fixture = next(
            (fixture for fixture in semantic_fixtures if fixture.body_name == table_body),
            None,
        )
        if self.table_fixture is None:
            self.table_fixture = next(
                (
                    fixture
                    for fixture in getattr(self, "fixtures", {}).values()
                    if fixture.body_name == table_body
                ),
                None,
            )
        if self.table_fixture is None:
            tables = self.get_fixture(category="desk", return_all=True) or ()
            self.table_fixture = next(
                (fixture for fixture in tables if fixture.body_name == table_body),
                None,
            )
        if self.table_fixture is not None:
            self.fixture_refs["table"] = self.table_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _setup_baked_scene_fixtures(self):
        """Add task-local desk support where the Objaverse collision is open."""
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured ice-cube scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        self._add_missing_desk_support(spec)
        super()._setup_baked_scene_fixtures()

    def _add_missing_desk_support(self, spec):
        if self.scene_name not in ("train_1664", "train_2025"):
            return
        body = self.find_scene_body(spec["table_body"])
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing desk for ice-cube support: "
                f"{spec['table_body']}"
            )

        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        from robosuite.utils import transform_utils as T

        body_rot = T.quat2mat(T.convert_quat(body_quat, to="xyzw"))
        local_quat = T.convert_quat(T.mat2quat(body_rot.T), to="wxyz")
        support_top_z = float(
            spec.get("support_top_z", float(spec["support_z"]) + 0.001)
        )
        for name, pose, half_size in (
            ("cup", spec["cup_pose"], (0.09, 0.09)),
            ("ice", spec["ice_pose"], (0.035, 0.035)),
        ):
            geom_name = f"{self.scene_name}_ice_{name}_support"
            if body.find(f"./geom[@name='{geom_name}']") is not None:
                continue
            object_pos = np.asarray(pose["pos"], dtype=float)
            support_center = np.asarray(
                [object_pos[0], object_pos[1], support_top_z - 0.005],
                dtype=float,
            )
            local_pos = body_rot.T @ (support_center - body_pos)
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="box",
                pos=" ".join(str(float(value)) for value in local_pos),
                quat=" ".join(str(float(value)) for value in local_quat),
                size=f"{half_size[0]} {half_size[1]} 0.005",
                contype="8",
                conaffinity="15",
                group="0",
                friction="0.9 0.9 0.001",
                solimp="0.95 0.98 0.001",
                solref="0.004 1",
            )

    def _is_scene_valid(self):
        return self.table_fixture is not None

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        root = robocasa.models.assets_root
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(root, self.CUP_ASSET),
            scale=float(self.task_spec["object_scale"]),
        )
        ice = MJCFObject(
            name="ice_cube",
            mjcf_path=os.path.join(root, self.ICE_ASSET),
            scale=float(self.task_spec["object_scale"]),
        )
        self.objects[cup.name] = cup
        self.objects[ice.name] = ice
        self.model.merge_objects([cup, ice])

    def _reset_internal(self):
        super()._reset_internal()
        self._place(self.objects["cup"], self.task_spec["cup_pose"])
        self._place(self.objects["ice_cube"], self.task_spec["ice_pose"])
        self.sim.forward()
        self._reset_task_progress()

    def _place(self, obj, pose):
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
        if isinstance(qvel_addr, tuple):
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            self.sim.data.qvel[qvel_addr] = 0.0

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._success_hold = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        inserted = self._ice_in_cup_region()
        self._success_hold = self._success_hold + 1 if inserted else 0
        return ret

    def _ice_in_cup_region(self):
        ice = self.sim.data.get_body_xpos(self.objects["ice_cube"].root_body)
        ids = [
            i
            for i in range(self.sim.model.ngeom)
            if "cup" in (self.sim.model.geom_id2name(i) or "")
            and "reg_bbox" in (self.sim.model.geom_id2name(i) or "")
        ]
        if not ids:
            return False
        region_id = ids[0]
        center = self.sim.data.geom_xpos[region_id]
        half_size = self.sim.model.geom_size[region_id]
        return bool(np.all(np.abs(ice - center) <= half_size))

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="pick up an ice cube and put it into the glass cup",
            success_definition=(
                "the ice cube must remain in the calibrated glass target region "
                "for one second"
            ),
        )
        return meta


__all__ = ["HolodeckIceCubeInGlass"]
