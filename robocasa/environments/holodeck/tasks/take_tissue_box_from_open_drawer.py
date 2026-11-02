"""Holodeck task for taking a tissue box out of an open drawer."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_atomic.pick_place import resolve_task_support_region
from robocasa.models.arenas.molmospaces_arena import inline_defaults
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robocasa.models.scenes.thor_semantics import SemanticRegion
from robosuite.utils import transform_utils as T


class _HolodeckDrawerTissueBoxMixin:
    """Shared calibrated drawer, tissue-box, and desk checks."""

    DESTINATION_REGION_REQUIRED = True
    DESTINATION_TABLE_SPEC_TASK = None
    DESTINATION_SUPPORT_SPEC_TASK = None

    TISSUE_BOX_ASSET = (
        "objects/thor/Miscellaneous Objects/TissueBox/Prefabs/"
        "Tissue_Box_1/Tissue_Box_1.xml"
    )
    TISSUE_BOX_SCALE = 0.5
    HOLD_SECONDS = 1.0
    HOLD_STEPS = 5  # Fallback used by declarative composite metadata.
    # The imported tissue-box visual bbox extends about 4.6 mm below its
    # lowest collidable primitive.  Raise the task-local desk contact plane by
    # 6 mm so a box whose bbox is placed at the calibrated desk height also
    # produces a stable MuJoCo contact (with a small overlap margin).
    DESTINATION_SUPPORT_CONTACT_OFFSET = 0.006
    MIN_LIFT_DELTA = 0.30
    SUCCESS_DEBUG_INTERVAL_SECONDS = 2.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    START_DRAWER_CLOSED = False

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        destination_spec_task = getattr(self, "DESTINATION_TABLE_SPEC_TASK", None)
        required = (
            "target_fixture",
            "target_joint",
            "robot_pose",
            "lang",
        )
        if destination_spec_task is None:
            required += ("target_table_body",)
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=required,
        )
        target_table_body = self.task_spec.get("target_table_body")
        if destination_spec_task is not None:
            destination_spec = get_holodeck_task_spec(
                self.scene_name,
                destination_spec_task,
                required=("desk_body",),
            )
            target_table_body = destination_spec["desk_body"]
        self.target_table_fixture = self.get_fixture(
            id=target_table_body
        )
        if self.target_table_fixture is not None:
            self.fixture_refs["destination_desk"] = self.target_table_fixture
        if self.target_fixture is not None:
            self.fixture_refs["drawer"] = self.target_fixture

    def _setup_baked_scene_fixtures(self):
        if getattr(self, "DESTINATION_SUPPORT_SPEC_TASK", None) is not None:
            self._add_missing_destination_desk_support()
        super()._setup_baked_scene_fixtures()

    def _add_missing_destination_desk_support(self):
        """Restore the calibrated desk support for placement workflows.

        The 1664/2025 Objaverse desk is the same desk used by
        ``HolodeckApproachDesk``, but its collision mesh does not produce a
        semantic support region.  Reuse the measured tabletop bounds from the
        desk manipulation calibration for a task-local support patch.
        """
        if self.scene_name not in ("train_1664", "train_2025"):
            return

        desk_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DESTINATION_TABLE_SPEC_TASK,
            required=("desk_body",),
        )
        support_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DESTINATION_SUPPORT_SPEC_TASK,
            required=("desk_body", "tabletop_bounds_xy", "support_z"),
        )
        if support_spec["desk_body"] != desk_spec["desk_body"]:
            raise RuntimeError(
                f"{self.scene_name}: tissue-box support calibration does not "
                "target the approach-desk body"
            )

        body = self.find_scene_body(desk_spec["desk_body"])
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing approach-desk body for tissue-box support: "
                f"{desk_spec['desk_body']}"
            )

        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        body_rot = T.quat2mat(T.convert_quat(body_quat, to="xyzw"))
        local_quat = T.convert_quat(T.mat2quat(body_rot.T), to="wxyz")
        tabletop_bounds = np.asarray(support_spec["tabletop_bounds_xy"], dtype=float)
        if tabletop_bounds.shape != (2, 2) or np.any(
            tabletop_bounds[1] <= tabletop_bounds[0]
        ):
            raise RuntimeError(
                f"{self.scene_name}: invalid calibrated desk tabletop bounds"
            )
        support_top_z = float(support_spec["support_z"]) + 0.001
        support_contact_z = (
            support_top_z + self.DESTINATION_SUPPORT_CONTACT_OFFSET
        )
        support_pos = tabletop_bounds.mean(axis=0)
        support_center = np.asarray(
            [support_pos[0], support_pos[1], support_contact_z - 0.005],
            dtype=float,
        )
        local_pos = body_rot.T @ (support_center - body_pos)
        geom_name = f"{self.scene_name}_tissue_box_support"
        if body.find(f"./geom[@name='{geom_name}']") is not None:
            return
        half_size = (tabletop_bounds[1] - tabletop_bounds[0]) / 2.0
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

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        xml_path = self._make_tissue_box_xml()
        try:
            tissue_box = MJCFObject(
                name="tissue_box",
                mjcf_path=xml_path,
                scale=self.TISSUE_BOX_SCALE,
                density=120,
            )
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.objects[tissue_box.name] = tissue_box
        self.model.merge_objects([tissue_box])

    @classmethod
    def _make_tissue_box_xml(cls):
        """Flatten the THOR tissue-box prefab into a free-joint object."""
        source = os.path.join(robocasa.models.assets_root, cls.TISSUE_BOX_ASSET)
        tree = ET.parse(source)
        root = tree.getroot()
        inline_defaults(root)
        worldbody = root.find("worldbody")
        body = worldbody.find("body") if worldbody is not None else None
        if body is None:
            raise ValueError(f"{source}: missing tissue-box root body")
        worldbody.remove(body)
        for child in list(body):
            if child.tag == "joint":
                body.remove(child)
        body.set("name", "object")
        for geom in body.iter("geom"):
            if geom.get("group") == "4":
                geom.set("group", "0")
            elif geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                geom.set("group", "1")
        ET.SubElement(
            body,
            "geom",
            name="reg_bbox",
            type="box",
            pos="0 -0.002 0",
            size="0.11 0.075 0.06",
            contype="0",
            conaffinity="0",
            group="1",
            rgba="0 0 0 0",
        )
        wrapper = ET.SubElement(worldbody, "body")
        wrapper.append(body)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", dir=os.path.dirname(source), delete=False
        ) as file:
            tree.write(file, encoding="utf-8", xml_declaration=True)
            return file.name

    def _setup_scene(self):
        super()._setup_scene()
        if self.target_table_fixture is None:
            raise RuntimeError(f"{self.scene_name}: calibrated destination desk is missing")

        joint_id = self.sim.model.joint_name2id(self.target_joint)
        drawer_body = self.sim.model.body_id2name(
            int(self.sim.model.jnt_bodyid[joint_id])
        )
        drawer_regions = [
            region
            for region in self.get_region(
                fixture=self.target_fixture, kind="receptacle", return_all=True
            )
            if region.parent_body == drawer_body
        ]
        if not drawer_regions:
            raise RuntimeError(f"{self.scene_name}: target drawer has no receptacle region")
        self.drawer_region = max(
            drawer_regions,
            key=lambda region: (float(np.prod(region.half_size)), region.stable_id),
        )
        self.region_refs["drawer_contents"] = self.drawer_region

        support_regions = self.get_region(
            fixture=self.target_table_fixture, kind="support", return_all=True
        )
        if not support_regions:
            # A few exported desks expose a shallow tabletop receptacle instead
            # of a support site. Convert only that local region for placement.
            shallow = self.get_region(
                fixture=self.target_table_fixture, kind="receptacle", return_all=True
            )
            support_regions = []
            for region in shallow:
                try:
                    support_regions.append(resolve_task_support_region(region))
                except ValueError:
                    continue
        if not support_regions and self.DESTINATION_SUPPORT_SPEC_TASK is not None:
            support_regions = [self._configured_destination_support_region()]
        if not support_regions:
            if self.DESTINATION_REGION_REQUIRED:
                raise RuntimeError(
                    f"{self.scene_name}: destination desk has no support surface"
                )
            self.destination_region = None
            return
        self.destination_region = max(
            support_regions,
            key=lambda region: (float(np.prod(region.half_size[:2])), region.stable_id),
        )
        self.region_refs["destination_desk"] = self.destination_region

    def _configured_destination_support_region(self):
        support_spec = get_holodeck_task_spec(
            self.scene_name,
            self.DESTINATION_SUPPORT_SPEC_TASK,
            required=("tabletop_bounds_xy", "support_z"),
        )
        tabletop_bounds = np.asarray(support_spec["tabletop_bounds_xy"], dtype=float)
        if tabletop_bounds.shape != (2, 2) or np.any(
            tabletop_bounds[1] <= tabletop_bounds[0]
        ):
            raise RuntimeError(
                f"{self.scene_name}: invalid calibrated desk tabletop bounds"
            )
        support_pos = tabletop_bounds.mean(axis=0)
        support_top_z = float(support_spec["support_z"]) + 0.001
        half_size = (tabletop_bounds[1] - tabletop_bounds[0]) / 2.0
        return SemanticRegion(
            stable_id=(
                f"{self.target_table_fixture.stable_id}:region:"
                "tissue_box_support"
            ),
            fixture_id=self.target_table_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_size[0], half_size[1], 0.02]),
            _world_center=np.array(
                [support_pos[0], support_pos[1], support_top_z], dtype=float
            ),
            _world_yaw=0.0,
        )

    def _set_tissue_box_in_drawer(self):
        center, _ = self.drawer_region.world_pose(self.sim)
        joint = self.objects["tissue_box"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        # Search vertically because imported object roots do not necessarily
        # coincide with their visual bbox center.
        for z in np.arange(float(center[2]) + 0.08, float(center[2]) - 0.12, -0.002):
            qpos[:3] = [float(center[0]), float(center[1]), float(z)]
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
            self.sim.forward()
            if self._tissue_box_in_drawer():
                return
        raise RuntimeError(f"{self.scene_name}: no stable tissue-box pose inside drawer")

    def _tissue_box_in_drawer(self):
        return bool(
            self.object_in_region(
                "tissue_box", self.drawer_region, rim_tol=0.015, xy_tol=0.01
            )
            and self._object_contacts_fixture("tissue_box", self.target_fixture)
        )

    def _object_contacts_fixture(self, name, fixture, penetration_tolerance=0.003):
        object_bodies = self._body_tree_ids([self.obj_body_id[name]])
        fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in object_bodies and body_2 in fixture_bodies) or (
                body_2 in object_bodies and body_1 in fixture_bodies
            ):
                distances.append(float(contact.dist))
        return bool(distances and min(distances) >= -float(penetration_tolerance))

    def _reset_internal(self):
        super()._reset_internal()
        # ``sim.reset()`` restores the compiled robot qpos.  Re-apply the
        # task calibration because articulated/composite setup may have
        # selected or temporarily used another pose during the prior episode.
        pose = self._articulation_robot_pose()
        if pose is not None:
            pos, yaw = pose
            if not (
                self._set_robot_free_pose(np.asarray(pos, dtype=float)[:2], float(yaw))
                and self._calibrate_robot_free_joint_height()
            ):
                raise RuntimeError(
                    f"{self.scene_name}: failed to restore calibrated tissue-box spawn"
                )
            self.robot_base_pos = np.asarray(pos, dtype=float).copy()
            self.robot_base_ori = np.array([0.0, 0.0, float(yaw)])
            self.sim.forward()
        # ``sim.reset()`` restores the compiled XML qpos, not the task-specific
        # qpos applied by ``RoomOpenArticulation._setup_scene()``. Reapply the
        # calibrated drawer state on every reset so later episodes match the
        # first one.
        self.sim.data.set_joint_qpos(
            self.target_joint, self._initial_joint_qpos(self.target_semantic_joint)
        )
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)
        self.sim.forward()
        if (
            self.START_DRAWER_CLOSED
            and not np.isclose(
                self.target_semantic_joint.fraction(self.sim),
                self.OPEN_INIT_FRACTION,
            )
        ):
            raise RuntimeError(
                f"{self.scene_name}: drawer composite does not start at its "
                "calibrated opening fraction"
            )
        self._set_tissue_box_in_drawer()
        self._initial_tissue_box_in_drawer = self._tissue_box_in_drawer()
        if not self._initial_tissue_box_in_drawer:
            raise RuntimeError(f"{self.scene_name}: tissue box reset is outside drawer")
        center, yaw = self.drawer_region.world_pose(self.sim)
        tissue_box_pos = self.object_pos("tissue_box")
        delta = np.asarray(tissue_box_pos[:2], dtype=float) - np.asarray(center[:2])
        c, s = np.cos(-yaw), np.sin(-yaw)
        self._drawer_tissue_box_local_offset = np.array(
            [
                c * delta[0] - s * delta[1],
                s * delta[0] + c * delta[1],
                tissue_box_pos[2] - center[2],
            ],
            dtype=float,
        )
        self._tissue_box_grasped_after_open = False
        self._tissue_box_removed_after_open_state = False
        self._invalid_pre_open_grasp = False
        # A calibrated pose can leave a closed hand touching the object at
        # spawn.  Do not treat that contact as the task's grasp action; require
        # one observed release before a grasp can arm the removal predicate.
        initially_grasped = False
        grasp_probe = getattr(self, "g1_hand_grasping_object", None)
        if callable(grasp_probe):
            initially_grasped = bool(grasp_probe("tissue_box"))
        self._tissue_box_was_ungrasped_after_reset = not initially_grasped
        self._invalid_spawn_grasp = initially_grasped
        self._success_hold = 0
        self._success_debug_step = 0
        self._tissue_box_initial_z = self._tissue_box_root_z()

    def _follow_tissue_box_with_drawer(self):
        """Keep the loose tissue box on the drawer while it is opening."""
        center, yaw = self.drawer_region.world_pose(self.sim)
        local = self._drawer_tissue_box_local_offset
        c, s = np.cos(yaw), np.sin(yaw)
        world_xy = np.asarray(center[:2]) + np.array(
            [c * local[0] - s * local[1], s * local[0] + c * local[1]]
        )
        qpos = self.sim.data.get_joint_qpos(self.objects["tissue_box"].joints[0]).copy()
        qpos[:3] = [world_xy[0], world_xy[1], float(center[2]) + local[2]]
        self.sim.data.set_joint_qpos(self.objects["tissue_box"].joints[0], qpos)
        self.sim.data.set_joint_qvel(self.objects["tissue_box"].joints[0], np.zeros(6))
        self.sim.forward()

    def _drawer_open_enough(self, fraction):
        return float(fraction) >= float(self.SUCCESS_OPEN_FRACTION)

    def _tissue_box_following_allowed(self):
        """Allow drawer-following only before the composite leaves open stage."""
        if getattr(self, "_tissue_box_removed_after_open_state", False):
            return False
        if getattr(self, "_tissue_box_removed", False):
            return False
        active = getattr(self, "current_atomic_task", None)
        return active is None or active.name == "open_drawer"

    def _tissue_box_root_z(self):
        return float(
            self.sim.data.get_body_xpos(self.objects["tissue_box"].root_body)[2]
        )

    def _tissue_box_lifted(self):
        return bool(
            self._tissue_box_root_z()
            > self._tissue_box_initial_z + float(self.MIN_LIFT_DELTA)
        )

    @property
    def hold_steps(self):
        control_freq = getattr(self, "control_freq", None)
        if control_freq is None:
            return max(1, int(self.HOLD_STEPS))
        return max(1, int(round(self.HOLD_SECONDS * control_freq)))

    def _post_action(self, action):
        # The articulation parent also updates ``_success_hold`` for its drawer
        # predicate. Preserve this task's value so its independent lift hold
        # count is neither reset nor contaminated by the parent condition.
        tissue_box_success_hold = self._success_hold
        ret = super()._post_action(action)
        fraction = self.target_semantic_joint.fraction(self.sim)
        grasped = self.g1_hand_grasping_object("tissue_box")
        if (
            getattr(self, "_invalid_spawn_grasp", False)
            and not grasped
            and self._tissue_box_in_drawer()
        ):
            # A spawn grasp is recoverable only if the box is released before
            # it leaves the drawer.  Otherwise this episode is invalid.
            self._invalid_spawn_grasp = False
            self._tissue_box_was_ungrasped_after_reset = True
        elif not grasped and not getattr(self, "_invalid_spawn_grasp", False):
            self._tissue_box_was_ungrasped_after_reset = True
        if not self._drawer_open_enough(fraction) and grasped:
            self._invalid_pre_open_grasp = True
        if (
            not self._drawer_open_enough(fraction)
            and not grasped
            and not self._tissue_box_grasped_after_open
            and self._tissue_box_following_allowed()
        ):
            self._follow_tissue_box_with_drawer()
        tissue_box_root_z = self._tissue_box_root_z()
        height_threshold = self._tissue_box_initial_z + self.MIN_LIFT_DELTA
        lifted = self._tissue_box_lifted()
        self._tissue_box_removed_after_open_state = lifted
        self._success_hold = tissue_box_success_hold + 1 if lifted else 0
        self._success_debug_step = getattr(self, "_success_debug_step", 0) + 1
        debug_interval = max(
            1,
            int(
                round(
                    self.SUCCESS_DEBUG_INTERVAL_SECONDS
                    * getattr(self, "control_freq", self.HOLD_STEPS)
                )
            ),
        )
        success = self._check_success()
        # if self._success_debug_step % debug_interval == 0:
        #     root_pos = np.asarray(
        #         self.sim.data.get_body_xpos(self.objects["tissue_box"].root_body),
        #         dtype=float,
        #     )
        #     print(
        #         f"[holodeck tissue-box success] scene={self.scene_name} "
        #         f"root_pos={np.round(root_pos, 4).tolist()} "
        #         f"initial_root_z={self._tissue_box_initial_z:.4f} "
        #         f"height_delta={tissue_box_root_z - self._tissue_box_initial_z:.4f} "
        #         f"threshold_z={height_threshold:.4f} lifted={lifted} "
        #         f"hold={self._success_hold}/{self.hold_steps} success={success}",
        #         flush=True,
        #     )
        if isinstance(ret, tuple) and len(ret) == 3:
            # The parent computes reward before this task updates its hold
            # counter; publish the post-update value to avoid a one-step lag.
            ret = (float(success), ret[1], ret[2])
        return ret

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.task_spec["lang"],
            object_model=self.TISSUE_BOX_ASSET,
            object_scale=self.TISSUE_BOX_SCALE,
            drawer_fixture=self.target_fixture.stable_id,
            drawer_joint=self.target_joint,
            drawer_region=self.drawer_region.stable_id,
            destination_fixture=self.target_table_fixture.stable_id,
            destination_region=(
                self.destination_region.stable_id
                if self.destination_region is not None
                else None
            ),
            composite_stages=list(self.COMPOSITE_STAGES),
            lift_delta=float(self.MIN_LIFT_DELTA),
            success_definition=(
                f"the tissue box root rises more than {self.MIN_LIFT_DELTA:.2f} m "
                f"above its reset height for {self.HOLD_SECONDS:.1f} s"
            ),
        )
        return meta


class HolodeckTakeTissueBoxFromOpenDrawer(
    _HolodeckDrawerTissueBoxMixin, RoomOpenArticulation, HolodeckEnv
):
    """Take the tissue box out of the already-open office drawer."""

    ATOMIC_SKILL = "TakeTissueBoxFromOpenDrawer"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_STAGES = (
        "start with the calibrated drawer open and the tissue box inside",
        "grasp and remove the tissue box from the open drawer",
    )
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Dresser",)
    TARGET_LANG = "dresser drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUCCESS_JOINT_SELECTION = "target"
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "take_tissue_box_from_open_drawer"
    OPEN_STAND_DISTANCE = 0.35
    # This atomic skill only removes the box; placement composites retain the
    # mixin's strict destination-support requirement.
    DESTINATION_REGION_REQUIRED = False

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)


__all__ = ["HolodeckTakeTissueBoxFromOpenDrawer"]
