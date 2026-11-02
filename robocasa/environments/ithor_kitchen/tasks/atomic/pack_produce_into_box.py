"""iTHOR kitchen: pack apple and banana into a box on a cleared support."""

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.fixtures import Fixture
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class IThorKitchenPackProduceIntoBox(IThorKitchenEnv):
    """Put the apple and banana staged beside a box into the box."""

    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "pack_produce_into_box"

    ITEMS = (("apple", "apple"), ("banana", "banana"))
    BOX_XML = "objects/usd/box/converted"

    ROBOT_SCALE = 1.0
    BOX_SCALE = 0.28
    BOX_COLLISION_SCALE = np.array([0.70, 0.67, 0.95])
    PRODUCE_SCALE = 1.0

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_pack_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "support_region",
                    "box_pos",
                    "box_yaw",
                    "apple_pos",
                    "banana_pos",
                ),
            )
            self._pack_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.usd_box = None
        self.usd_box_semantic = None

        self._add_usd_box([0.0, 0.0, 0.0])
        self.work_surface_fixture, self.work_surface_region = self._make_support_region(
            self._get_task_spec()["support_region"]
        )
        self.fixture_refs["work_surface"] = self.work_surface_fixture
        self.region_refs["work_surface"] = self.work_surface_region
        self._set_box_yaw(float(self._get_task_spec()["box_yaw"]))

    def _make_support_region(self, spec):
        center = np.asarray(spec.get("center"), dtype=float)
        half_xy = np.asarray(spec.get("half_xy"), dtype=float)
        yaw = float(spec.get("yaw", 0.0))
        body_name = spec.get("body_name")
        if center.shape != (3,) or half_xy.shape != (2,) or np.any(half_xy <= 0.0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/support_region: "
                "expected center XYZ and positive half_xy"
            )
        if not isinstance(body_name, str) or not body_name:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/support_region: "
                "body_name is required"
            )
        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:support"
        region = SemanticRegion(
            stable_id=f"{fixture_id}:region:support",
            fixture_id=fixture_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            _world_center=center,
            _world_yaw=yaw,
        )
        fixture = SemanticFixture(
            stable_id=fixture_id,
            name=f"{self.SPEC_TASK}_support",
            category="table",
            source_category="TaskCalibratedSupport",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=f"{self.SPEC_TASK}_support",
            center=center,
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            yaw=yaw,
            front=np.array([np.cos(yaw), np.sin(yaw)]),
            capabilities={"support", "task_calibrated"},
            regions={region.stable_id: region},
        )
        self.semantic_scene.add_fixture(fixture)
        return fixture, region

    def _add_usd_box(self, pos, yaw=0.0):
        self.usd_box = Fixture(
            xml=self.BOX_XML,
            name="usd_box",
            duplicate_collision_geoms=False,
            scale=self.BOX_SCALE,
            rng=self.rng,
            joints=[dict(type="free", damping="0.001")],
        )
        self.usd_box.set_euler([0.0, 0.0, float(yaw)])
        self.usd_box.set_pos(np.asarray(pos, dtype=float))
        for geom in self.usd_box.worldbody.iter("geom"):
            geom_name = geom.get("name") or ""
            if not (geom_name.endswith("box_floor_col") or "box_wall_" in geom_name):
                continue
            geom.set(
                "size",
                array_to_string(
                    string_to_array(geom.get("size")) * self.BOX_COLLISION_SCALE
                ),
            )
            geom.set(
                "pos",
                array_to_string(
                    string_to_array(geom.get("pos")) * self.BOX_COLLISION_SCALE
                ),
            )
        self.furniture["usd_box"] = self.usd_box
        self.fixtures["usd_box"] = self.usd_box
        self.usd_box_semantic = self.register_semantic_fixture_model(
            self.usd_box,
            name="usd_box",
            category="box",
            source_category="Box",
            asset_id="usd_box",
            region_kinds={"receptacle": "receptacle"},
        )

    def _set_box_yaw(self, yaw):
        yaw = float(yaw)
        self.usd_box.set_euler([0.0, 0.0, yaw])
        self.usd_box_semantic.yaw = yaw
        self.usd_box_semantic.front = np.array([np.cos(yaw), np.sin(yaw)])

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.usd_box is not None
            and self.work_surface_region is not None
        )

    def _get_obj_cfgs(self):
        if self.work_surface_region is None:
            return []
        return [
            dict(
                name=name,
                obj_groups=group,
                object_scale=self.PRODUCE_SCALE,
            )
            for name, group in self.ITEMS
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualPackProduceIntoBoxSampler")

    def _before_object_placement(self):
        if self.work_surface_region is None or self.usd_box is None:
            return
        self.clear_support_freejoint_clutter(
            self.work_surface_region,
            self.work_surface_fixture,
            protected_body_roots=self._task_body_roots(),
        )

    def _semantic_placements_are_valid(self):
        return True

    def _reset_internal(self):
        self._success_hold_steps = 0
        super()._reset_internal()
        self.success_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        if self.work_surface_region is None:
            return
        self._place_task_objects()

    def _task_body_roots(self):
        roots = []
        if self.usd_box is not None:
            try:
                roots.append(self.sim.model.body_name2id(self.usd_box.root_body))
            except Exception:
                pass
        roots.extend(
            self.obj_body_id[name]
            for name, _ in self.ITEMS
            if name in getattr(self, "obj_body_id", {})
        )
        return roots

    def _place_task_objects(self):
        spec = self._get_task_spec()
        self._set_free_joint_bottom_at(self.usd_box, spec["box_pos"])
        for name, _ in self.ITEMS:
            self._set_free_joint_bottom_at(self.objects[name], spec[f"{name}_pos"])
        self.sim.forward()

    def _set_free_joint_bottom_at(self, obj, pos):
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: object positions need XYZ"
            )
        joint_name = obj.joints[0]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:2] = pos[:2]
        qpos[2] = float(pos[2] - obj.bottom_offset[2])
        self.sim.data.set_joint_qpos(joint_name, qpos)
        try:
            qvel_addr = self.sim.model.get_joint_qvel_addr(joint_name)
        except Exception:
            qvel_addr = None
        if qvel_addr is not None:
            if isinstance(qvel_addr, tuple):
                self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
            else:
                self.sim.data.qvel[qvel_addr] = 0.0

    def object_inside_box_collision(self, name, margin=0.0):
        root_id = self.sim.model.body_name2id(self.usd_box.root_body)
        root_pos = np.asarray(self.sim.data.body_xpos[root_id], dtype=float)
        root_rot = np.asarray(self.sim.data.body_xmat[root_id], dtype=float).reshape(
            3, 3
        )

        centers = {}
        sizes = {}
        for suffix in (
            "box_floor_col",
            "box_wall_x_pos",
            "box_wall_x_neg",
            "box_wall_y_pos",
            "box_wall_y_neg",
        ):
            geom_id = self.sim.model.geom_name2id(
                f"{self.usd_box.naming_prefix}{suffix}"
            )
            centers[suffix] = root_rot.T @ (self.sim.data.geom_xpos[geom_id] - root_pos)
            sizes[suffix] = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)

        low = np.array(
            [
                centers["box_wall_x_neg"][0] + sizes["box_wall_x_neg"][0],
                centers["box_wall_y_neg"][1] + sizes["box_wall_y_neg"][1],
                centers["box_floor_col"][2] + sizes["box_floor_col"][2],
            ]
        )
        high = np.array(
            [
                centers["box_wall_x_pos"][0] - sizes["box_wall_x_pos"][0],
                centers["box_wall_y_pos"][1] - sizes["box_wall_y_pos"][1],
                min(
                    centers["box_wall_x_pos"][2] + sizes["box_wall_x_pos"][2],
                    centers["box_wall_x_neg"][2] + sizes["box_wall_x_neg"][2],
                    centers["box_wall_y_pos"][2] + sizes["box_wall_y_pos"][2],
                    centers["box_wall_y_neg"][2] + sizes["box_wall_y_neg"][2],
                ),
            ]
        )
        local = root_rot.T @ (self.object_pos(name) - root_pos)
        return bool(np.all(local >= low + margin) and np.all(local <= high - margin))

    def _raw_success(self):
        return all(self.object_inside_box_collision(name) for name, _ in self.ITEMS)

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        required_steps = max(
            1, int(round(self.success_hold_seconds * self.control_freq))
        )
        return self._success_hold_steps >= required_steps

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "put the apple and banana into the box"
        return ep_meta
