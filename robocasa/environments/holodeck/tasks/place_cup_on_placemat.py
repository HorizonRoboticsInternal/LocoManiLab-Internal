"""Place the calibrated cup on a placemat on the Holodeck desk."""

import os
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.utils import transform_utils as T


class HolodeckPlaceCupOnPlacemat(HolodeckEnv):
    """Pick up the desk cup and release it fully on the placemat."""

    ATOMIC_SKILL = "PickPlaceOnSurface"
    TASK_FAMILY = "pick_place"
    CUP_ASSET = "objects/lightwheel/glass_cup/GlassCup014/model.xml"
    PLACEMAT_ASSET = "objects/lightwheel/placemat/Placemat001/model.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "place_cup_on_placemat"
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    NOISE_FREE_ROBOT_SPAWN = True
    # Allow a small visual placement error around the mat boundary. The cup
    # must still remain within 5 cm of the mat's measured bbox.
    PLACEMAT_EDGE_TOLERANCE = 0.05
    SUPPORT_Z_TOLERANCE = 0.035
    MIN_MANIPULATION_DISPLACEMENT = 0.10
    SURFACE_Z_TOLERANCE = 0.035
    MAX_LINEAR_SPEED = 0.08
    MAX_ANGULAR_SPEED = 0.30

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
                "cup_scale",
                "cup_pose",
                "placemat_scale",
                "placemat_pose",
                "target_half_size",
                "support_z",
            ),
        )
        # Some Holodeck Objaverse exports leave objectType/category empty even
        # when the calibrated body is a desk. Resolve the task surface by its
        # explicit body name so scene validity does not depend on that label.
        table_body = self.task_spec["table_body"]
        self.desk_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == table_body
            ),
            None,
        )
        if self.desk_fixture is not None:
            self.fixture_refs["cup_surface"] = self.desk_fixture
            self.fixture_refs["placemat_surface"] = self.desk_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _setup_baked_scene_fixtures(self):
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

    def _add_missing_desk_support(self, spec):
        """Restore the desktop collision under the calibrated placemat.

        The Objaverse desk has a visual tabletop extending past the matching
        collision mesh in these two scenes.  A task-local support keeps the
        visible placement physically supported without changing the baked XML.
        """
        if self.scene_name not in ("train_1664", "train_2025"):
            return
        body = self.find_scene_body(spec["table_body"])
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing desk for placemat support: "
                f"{spec['table_body']}"
            )
        # XML body poses use MuJoCo's wxyz convention.  Orient the child box
        # so its axes are world-aligned despite the desk's rotated root body.
        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        body_rot = T.quat2mat(T.convert_quat(body_quat, to="xyzw"))
        local_rot = body_rot.T
        local_quat = T.convert_quat(T.mat2quat(local_rot), to="wxyz")
        # Keep the proxy just above the imperfect baked collision surface so
        # contact-margin resolution cannot pull a free object through it.
        support_top_z = float(spec["support_z"]) + 0.001
        for name, pose, half_size in (
            ("cup", spec["cup_pose"], (0.09, 0.09)),
            ("placemat", spec["placemat_pose"], (0.20, 0.147)),
        ):
            geom_name = f"{self.scene_name}_{name}_support"
            if body.find(f"./geom[@name='{geom_name}']") is not None:
                continue
            support_pos = np.asarray(pose["pos"], dtype=float)
            support_center = np.asarray(
                [support_pos[0], support_pos[1], support_top_z - 0.005],
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
        return self.desk_fixture is not None

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        root = robocasa.models.assets_root
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(root, self.CUP_ASSET),
            scale=float(self.task_spec["cup_scale"]),
        )
        placemat = MJCFObject(
            name="placemat",
            mjcf_path=os.path.join(root, self.PLACEMAT_ASSET),
            scale=float(self.task_spec["placemat_scale"]),
        )
        self.objects[cup.name] = cup
        self.objects[placemat.name] = placemat
        self.model.merge_objects([cup, placemat])

    def _place(self, name, pose):
        obj = self.objects[name]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
        if isinstance(qvel_addr, tuple):
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            self.sim.data.qvel[qvel_addr] = 0.0

    def _reset_internal(self):
        super()._reset_internal()
        self._place("placemat", self.task_spec["placemat_pose"])
        self._place("cup", self.task_spec["cup_pose"])
        self.sim.forward()
        self._initial_placemat_pos = self.object_pos("placemat").copy()
        self._initial_cup_pos = self.object_pos("cup").copy()
        self._success_hold = 0
        if not self._initial_layout_valid():
            raise RuntimeError(f"{self.scene_name}: cup placemat reset is invalid")

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
        """Validate the calibrated support height without requiring a contact event."""
        return bool(
            abs(self.object_bottom_z(name) - float(self.task_spec["support_z"]))
            <= self.SUPPORT_Z_TOLERANCE
        )

    # These helpers remain for the tissue-box-and-cup composite. The atomic
    # task's success predicate below intentionally does not use them.
    def _object_top_z(self, name):
        return float(np.max(self.object_bbox_points(name)[:, 2]))

    def _placemat_is_stable(self):
        placemat = self.objects["placemat"]
        placemat_mat = np.asarray(
            self.sim.data.get_body_xmat(placemat.root_body), dtype=float
        ).reshape(3, 3)
        return bool(
            np.linalg.norm(
                self.object_pos("placemat")[:2] - self._initial_placemat_pos[:2]
            )
            <= 0.025
            and float(placemat_mat[2, 2]) >= 0.98
            and abs(
                self.object_bottom_z("placemat") - float(self.task_spec["support_z"])
            )
            <= self.SUPPORT_Z_TOLERANCE
            and self._object_is_supported_by_desk("placemat")
        )

    def _cup_is_stable(self):
        cup = self.objects["cup"]
        cup_mat = np.asarray(
            self.sim.data.get_body_xmat(cup.root_body), dtype=float
        ).reshape(3, 3)
        surface_gap = self.object_bottom_z("cup") - self._object_top_z("placemat")
        return bool(
            float(cup_mat[2, 2]) >= 0.90
            and abs(surface_gap) <= self.SURFACE_Z_TOLERANCE
            and np.linalg.norm(
                self.sim.data.get_joint_qvel(self.objects["cup"].joints[0])[:3]
            )
            <= self.MAX_LINEAR_SPEED
            and np.linalg.norm(
                self.sim.data.get_joint_qvel(self.objects["cup"].joints[0])[3:]
            )
            <= self.MAX_ANGULAR_SPEED
        )

    def _cup_was_manipulated(self):
        displacement = np.linalg.norm(
            self.object_pos("cup")[:2] - self._initial_cup_pos[:2]
        )
        vertical_lift = self.object_pos("cup")[2] - self._initial_cup_pos[2]
        return bool(
            displacement >= self.MIN_MANIPULATION_DISPLACEMENT
            or vertical_lift >= self.MIN_MANIPULATION_DISPLACEMENT
        )

    def _cup_on_placemat(self):
        placemat = self.objects["placemat"]
        placemat_pos = np.asarray(
            self.sim.data.get_body_xpos(placemat.root_body), dtype=float
        )
        placemat_mat = np.asarray(
            self.sim.data.get_body_xmat(placemat.root_body), dtype=float
        ).reshape(3, 3)
        placemat_local = (
            self.object_bbox_points("placemat") - placemat_pos
        ) @ placemat_mat
        cup_local = (self.object_bbox_points("cup") - placemat_pos) @ placemat_mat
        placemat_min = np.min(placemat_local[:, :2], axis=0)
        placemat_max = np.max(placemat_local[:, :2], axis=0)
        cup_min = np.min(cup_local[:, :2], axis=0)
        cup_max = np.max(cup_local[:, :2], axis=0)
        tolerance = float(self.PLACEMAT_EDGE_TOLERANCE)
        inside_xy = bool(
            np.all(cup_min >= placemat_min - tolerance)
            and np.all(cup_max <= placemat_max + tolerance)
        )
        # Restore the scene117 atomic evaluation rule; composites keep contact.
        if type(self) is HolodeckPlaceCupOnPlacemat and self.scene_name == "train_117":
            return inside_xy
        return bool(inside_xy and self._cup_contacts_placemat())

    def _cup_contacts_placemat(self):
        """Return whether any cup geom is in contact with the placemat."""
        cup_bodies = self._body_tree_ids([self.obj_body_id["cup"]])
        placemat_bodies = self._body_tree_ids([self.obj_body_id["placemat"]])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in cup_bodies and body2 in placemat_bodies) or (
                body2 in cup_bodies and body1 in placemat_bodies
            ):
                return True
        return False

    def _post_action(self, action):
        valid = self._cup_on_placemat()
        self._success_hold = self._success_hold + 1 if valid else 0
        ret = super()._post_action(action)
        return ret

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
            lang="pick up the cup from the desk and place it on the placemat",
            source_object="cup",
            source_model=self.CUP_ASSET,
            target_object="placemat",
            target_model=self.PLACEMAT_ASSET,
            success_definition=(
                "the cup footprint must remain within the calibrated placemat "
                "target region and contact the placemat for one second"
            ),
        )
        if type(self) is HolodeckPlaceCupOnPlacemat and self.scene_name == "train_117":
            meta["success_definition"] = (
                "the cup footprint must remain within the calibrated placemat "
                "target region for one second"
            )
        return meta


__all__ = ["HolodeckPlaceCupOnPlacemat"]
