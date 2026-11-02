"""Pick the calibrated glass cup from the Holodeck desk."""

import os
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckPickCupFromDesk(HolodeckEnv):
    """Lift a GlassCup014 clear of the calibrated desk surface."""

    ATOMIC_SKILL = "PickObject"
    TASK_FAMILY = "pick"
    CUP_ASSET = "objects/lightwheel/glass_cup/GlassCup014/model.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "pick_cup_from_desk"
    HOLD_SECONDS = 1.0
    MIN_LIFT_DELTA = 0.30
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
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
                "cup_pose",
                "object_scale",
            ),
        )
        table_body = self.task_spec["table_body"]
        # Objaverse exports may leave objectType/category empty. Resolve the
        # calibrated desk by its exact stable body in both semantic and raw
        # fixture registries before using the category query as a fallback.
        semantic_fixtures = getattr(self.semantic_scene, "fixtures", {}).values()
        self.desk_fixture = next(
            (
                fixture
                for fixture in semantic_fixtures
                if fixture.body_name == table_body
            ),
            None,
        )
        if self.desk_fixture is None:
            self.desk_fixture = next(
                (
                    fixture
                    for fixture in getattr(self, "fixtures", {}).values()
                    if fixture.body_name == table_body
                ),
                None,
            )
        if self.desk_fixture is None:
            desks = self.get_fixture(category="desk", return_all=True) or ()
            self.desk_fixture = next(
                (fixture for fixture in desks if fixture.body_name == table_body),
                None,
            )
        if self.desk_fixture is not None:
            self.fixture_refs["cup_surface"] = self.desk_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _setup_baked_scene_fixtures(self):
        """Apply task-local scene cleanup before semantic parsing."""
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured cup scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        self._add_missing_desk_support(spec)
        super()._setup_baked_scene_fixtures()

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

    def _add_missing_desk_support(self, spec):
        """Add the calibrated cup support used by the other desk tasks."""
        if self.scene_name not in ("train_1664", "train_2025"):
            return
        body = self.find_scene_body(spec["table_body"])
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing desk for cup support: "
                f"{spec['table_body']}"
            )

        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        from robosuite.utils import transform_utils as T

        body_rot = T.quat2mat(T.convert_quat(body_quat, to="xyzw"))
        local_rot = body_rot.T
        local_quat = T.convert_quat(T.mat2quat(local_rot), to="wxyz")
        support_top_z = self._support_top_z(spec)
        pose = np.asarray(spec["cup_pose"]["pos"], dtype=float)
        support_center = np.asarray(
            [pose[0], pose[1], support_top_z - 0.005], dtype=float
        )
        local_pos = body_rot.T @ (support_center - body_pos)
        geom_name = f"{self.scene_name}_pick_cup_support"
        if body.find(f"./geom[@name='{geom_name}']") is not None:
            return
        ET.SubElement(
            body,
            "geom",
            name=geom_name,
            type="box",
            pos=" ".join(str(float(value)) for value in local_pos),
            quat=" ".join(str(float(value)) for value in local_quat),
            size="0.09 0.09 0.005",
            contype="8",
            conaffinity="15",
            group="0",
            friction="0.9 0.9 0.001",
            solimp="0.95 0.98 0.001",
            solref="0.004 1",
        )

    def _support_top_z(self, spec):
        """Resolve support height when a composite spec reuses cup layout only."""
        support_top_z = spec.get("support_top_z")
        if support_top_z is not None:
            return float(support_top_z)

        support_z = spec.get("support_z")
        if support_z is not None:
            return float(support_z) + 0.001

        canonical_spec = get_holodeck_task_spec(
            self.scene_name,
            "pick_cup_from_desk",
            required=("support_z",),
        )
        return float(
            canonical_spec.get(
                "support_top_z", float(canonical_spec["support_z"]) + 0.001
            )
        )

    def _is_scene_valid(self):
        return self.desk_fixture is not None

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(robocasa.models.assets_root, self.CUP_ASSET),
            scale=float(self.task_spec["object_scale"]),
        )
        self.objects[cup.name] = cup
        self.model.merge_objects([cup])

    def _reset_internal(self):
        super()._reset_internal()
        if not self._robot_footprint_is_in_room(radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: cup task robot starts off the room floor"
            )
        cup = self.objects["cup"]
        qpos = self.sim.data.get_joint_qpos(cup.joints[0]).copy()
        pose = self.task_spec["cup_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(cup.joints[0], qpos)
        self.sim.data.set_joint_qvel(cup.joints[0], np.zeros(6))
        self.sim.forward()
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: cup task reset has robot collision"
            )
        self._cup_initial_z = self._cup_root_z()
        self._success_hold = 0

    def _cup_root_z(self):
        return float(self.sim.data.get_body_xpos(self.objects["cup"].root_body)[2])

    def _post_action(self, action):
        ret = super()._post_action(action)
        cup_z = self._cup_root_z()
        lifted = cup_z > self._cup_initial_z + self.lift_delta
        self._success_hold = (
            self._success_hold + 1
            if lifted
            else 0
        )
        return ret

    @property
    def lift_delta(self):
        return float(self.MIN_LIFT_DELTA)

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "pick up the glass cup from the desk and lift it"
        meta["success_definition"] = (
            f"the cup root rises more than {self.lift_delta:.2f} m above its "
            f"reset height for {self.HOLD_SECONDS:.1f} s"
        )
        return meta


__all__ = ["HolodeckPickCupFromDesk"]
