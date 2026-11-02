"""Push the movable office chair in Holodeck scenes to the desk."""

import os
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import new_site

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckPushChairToDesk(HolodeckEnv):
    """Push a calibrated movable office chair into place at a desk."""

    ATOMIC_SKILL = "PushObject"
    TASK_FAMILY = "push"
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "push_chair_to_desk"
    CHAIR_ASSET = (
        "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_Furniture/"
        "OfficeChair/Prefabs/RoboTHOR_office_chair_allak/"
        "RoboTHOR_office_chair_allak.xml"
    )
    CHAIR_SPAWN_NAME = "holodeck_task_chair"
    TARGET_SITE_NAME = "push_chair_target_region"
    HOLD_SECONDS = 1.0
    # Keep the grippers just outside the chair collision envelope at reset.
    # The previous 0.55 m placement spawned the hands inside the chair in
    # train_1018, making the free body appear immovable at normal speed.
    PUSH_STANDOFF = 0.67
    # Treat the office chair as a light caster chair.  MuJoCo uses the first
    # friction coefficient for both tangential directions, so this makes
    # forward/backward and lateral sliding equally easy.
    CHAIR_MASS = 0.5
    CHAIR_INERTIA = 0.04
    CHAIR_CONTACT_FRICTION = (0.005, 0.001, 0.0001)

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        # A composite can enter this shared initializer through a sibling
        # task (for example PlaceBox -> Pull -> Push).  Let the pull subclass
        # select its chair calibration instead of relying on the composite's
        # first MRO-visible SPEC_TASK.
        spec_task = getattr(self, "PUSH_SPEC_TASK", self.SPEC_TASK)
        spec = get_holodeck_task_spec(self.scene_name, spec_task)
        remove_bodies = tuple(spec.get("remove_scene_bodies", ()))
        if remove_bodies:
            removed = self.remove_scene_bodies(remove_bodies)
            missing = set(remove_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured chair blockers are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec_task = getattr(self, "PUSH_SPEC_TASK", self.SPEC_TASK)
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            spec_task,
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
        self._ensure_task_chair_fixture()
        self.chair_body = self.task_spec["chair_body"]
        self.chair_joint = self.task_spec["chair_joint"]
        self._configure_chair_dynamics()
        self.target_xy = np.asarray(self.task_spec["target_xy"], dtype=float)
        self.target_tolerance = np.asarray(
            self.task_spec["target_tolerance"], dtype=float
        )
        chairs = self.get_fixture(category="chair", return_all=True)
        self.chair_fixture = next(
            (fixture for fixture in chairs if fixture.body_name == self.chair_body),
            None,
        )
        # Objaverse desks in train_1664/train_2025 have an empty semantic
        # category, so resolve the calibrated body directly instead of relying
        # on category inference or fixture-name aliases.
        desk_body = self.task_spec["desk_body"]
        self.desk_fixture = next(
            (
                fixture
                for fixture in getattr(self.semantic_scene, "fixtures", {}).values()
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
        if self.chair_fixture is not None:
            self.fixture_refs["push_chair"] = self.chair_fixture
        if self.desk_fixture is not None:
            self.fixture_refs["target_desk"] = self.desk_fixture
        if bool(self.task_spec.get("show_target_region", False)):
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.TARGET_SITE_NAME,
                    type="box",
                    pos=[*self.target_xy, 0.012],
                    size=[*self.target_tolerance, 0.01],
                    rgba=[0.1, 0.9, 0.2, 0.35],
                    group=0,
                )
            )
        robot_pose = self.task_spec["robot_pose"]
        # Derive the base heading from the actual push direction.  Keeping the
        # heading in sync with the calibrated positions prevents a stale YAML
        # yaw from making the robot face away from the chair.
        start_xy = np.asarray(self.task_spec["chair_start_pose"]["pos"][:2], dtype=float)
        push_delta = self.target_xy - start_xy
        push_norm = float(np.linalg.norm(push_delta))
        if push_norm > 1e-6:
            push_axis = push_delta / push_norm
            robot_xy = start_xy - push_axis * self.PUSH_STANDOFF
            robot_yaw = float(np.arctan2(push_axis[1], push_axis[0]))
        else:
            robot_xy = np.asarray(robot_pose["pos"], dtype=float)[:2]
            robot_yaw = float(robot_pose["yaw"])
        robot_pos = np.asarray(robot_pose["pos"], dtype=float).copy()
        robot_pos[:2] = robot_xy
        self.set_robot_base_pose(
            robot_pos,
            robot_yaw,
        )

    def _ensure_task_chair_fixture(self):
        """Spawn the task-local free-jointed chair for scenes without one."""
        configured_body = self.task_spec["chair_body"]
        if self.find_scene_body(configured_body) is not None:
            return
        asset = self.task_spec.get(
            "chair_asset", HolodeckPushChairToDesk.CHAIR_ASSET
        )
        pose = self.task_spec["chair_start_pose"]
        fixture = self.create_scene_fixture(
            xml_path=os.path.join(robocasa.models.assets_root, asset),
            pos=np.asarray(pose["pos"], dtype=float),
            quat=np.asarray(pose["quat"], dtype=float),
            name=self.task_spec.get(
                "chair_spawn_name", HolodeckPushChairToDesk.CHAIR_SPAWN_NAME
            ),
            category="chair",
            source_category="OfficeChair",
            asset_id=self.task_spec.get(
                "chair_asset_id", os.path.splitext(os.path.basename(asset))[0]
            ),
            ensure_inertials=True,
        )
        body = self.find_scene_body(fixture.body_name)
        joint = body.find("./joint[@type='free']") if body is not None else None
        if body is None or joint is None or not joint.get("name"):
            raise RuntimeError(
                f"{self.scene_name}: spawned chair asset has no free joint"
            )
        collision_local_y_offset = float(
            self.task_spec.get("chair_collision_local_y_offset", 0.0)
        )
        if collision_local_y_offset:
            for geom in body.iter("geom"):
                if (
                    geom.get("class") != "__DYNAMIC_MJT__"
                    and "PrimitiveCollider" not in (geom.get("name") or "")
                ):
                    continue
                geom_pos = [
                    float(value) for value in geom.get("pos", "0 0 0").split()
                ]
                # Chair_313_1 is authored with local +Y as its upright axis.
                geom_pos[1] += collision_local_y_offset
                geom.set("pos", " ".join(str(value) for value in geom_pos))
        # The source chair XML includes a red receptacle-collider site for
        # dataset annotation. It is not part of this manipulation task and
        # would otherwise render as a conspicuous red square in the scene.
        for site in body.iter("site"):
            site.set("rgba", "0 0 0 0")
            site.set("group", "0")
        # The XML helper prefixes names to avoid collisions. Keep the runtime
        # calibration aligned with those final names.
        self.task_spec["chair_body"] = fixture.body_name
        self.task_spec["chair_joint"] = joint.get("name")

    def _configure_chair_dynamics(self):
        """Make the baked chair responsive to physical pushing.

        Holodeck exports use the structural-furniture defaults for chair
        collision geoms (high friction and no explicit root inertial).  That
        makes the free-jointed chair require more force than the humanoid arm
        can apply.  Configure only this task's target chair before MuJoCo
        compiles the arena; other scene furniture remains unchanged.
        """
        body = self.find_scene_body(self.chair_body)
        if body is None:
            return
        joint = body.find("./joint")
        if joint is None or joint.get("type") != "free":
            return
        joint.set("damping", "0.0")
        joint.set("frictionloss", "0.0")
        inertial = body.find("./inertial")
        if inertial is None:
            inertial = ET.Element("inertial")
            body.insert(0, inertial)
        inertial.attrib.update(
            pos="0 0.45 0",
            mass=str(self.CHAIR_MASS),
            diaginertia="{0} {0} {0}".format(self.CHAIR_INERTIA),
        )
        chair_geoms = []
        for geom in body.iter("geom"):
            if (
                geom.get("class") != "__DYNAMIC_MJT__"
                and "PrimitiveCollider" not in (geom.get("name") or "")
            ):
                continue
            # The chair is a caster chair; keep floor contact low-friction so
            # the G1 can push it without overcoming structural-furniture drag.
            geom.set("friction", " ".join(map(str, self.CHAIR_CONTACT_FRICTION)))
            chair_geoms.append(geom.get("name"))
        # MuJoCo >= 3.0 hard-codes the "max" contact-friction combination, so
        # the per-geom friction above is ignored against the scene floor
        # (friction 0.9): pushing from hand height would tip the chair over
        # instead of sliding it.  Override each chair<->floor contact with an
        # explicit pair so the low friction actually applies.
        for geom_name in chair_geoms:
            if not geom_name:
                continue
            for floor_name in HolodeckPushChairToDesk._floor_geom_names(self):
                ET.SubElement(
                    HolodeckPushChairToDesk._contact_element(self),
                    "pair",
                    {
                        "geom1": floor_name,
                        "geom2": geom_name,
                        "friction": " ".join(
                            map(str, self.CHAIR_CONTACT_FRICTION)
                        ),
                    },
                )

    def _contact_element(self):
        contact = self.mujoco_arena.root.find("contact")
        if contact is None:
            contact = ET.SubElement(self.mujoco_arena.root, "contact")
        return contact

    def _floor_geom_names(self):
        return [
            geom.get("name")
            for geom in self.mujoco_arena.root.iter("geom")
            if geom.get("type") == "plane" and geom.get("name")
        ]

    def _is_scene_valid(self):
        return bool(self.chair_fixture is not None and self.desk_fixture is not None)

    def _setup_scene(self):
        super()._setup_scene()
        if self.chair_body not in self.sim.model.body_names:
            raise RuntimeError(
                f"{self.scene_name}: calibrated chair body is missing: "
                f"{self.chair_body}"
            )
        self._set_initial_chair_state()
        self.sim.forward()
        if self._success_position_condition():
            raise RuntimeError(
                f"{self.scene_name}: chair reset already satisfies the position condition"
            )
        self._reset_task_progress()

    def _reset_internal(self):
        super()._reset_internal()
        self._set_initial_chair_state()
        self.sim.forward()
        self._reset_task_progress()

    def _set_initial_chair_state(self, task_spec=None):
        spec = self.task_spec if task_spec is None else task_spec
        pose = spec["chair_start_pose"]
        qpos = np.concatenate(
            (
                np.asarray(pose["pos"], dtype=float),
                np.asarray(pose["quat"], dtype=float),
            )
        )
        self.sim.data.set_joint_qpos(self.chair_joint, qpos)
        self.sim.data.set_joint_qvel(self.chair_joint, np.zeros(6))

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._success_hold_steps = 0
        self._position_debug_steps = 0

    def _initial_jitter_extra_joints(self):
        joints = list(super()._initial_jitter_extra_joints())
        if getattr(self, "sim", None) is not None:
            joints.append(self.chair_joint)
        return tuple(joints)

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        # self._position_debug_steps += 1
        # if self._position_debug_steps % max(1, int(self.control_freq)) == 0:
        #     body_id = self.sim.model.body_name2id(self.chair_body)
        #     chair_xy = np.asarray(self.sim.data.body_xpos[body_id][:2], dtype=float)
        #     print(
        #         f"[holodeck chair] scene={self.scene_name} "
        #         f"xy={np.round(chair_xy, 3).tolist()} "
        #         f"target={np.round(self.target_xy, 3).tolist()}",
        #         flush=True,
        #     )
        if self._success_position_condition():
            self._success_hold_steps += 1
        else:
            self._success_hold_steps = 0
        # The parent computes reward before this primitive updates its hold
        # counter. Publish the post-update state so env.step() and teleop
        # logging report success on the step that completes the hold interval.
        success = bool(self._check_success())
        info["task_success"] = success
        return float(success), done, info

    def _chair_in_target_region(self, task_spec=None):
        del task_spec
        body_id = self.sim.model.body_name2id(self.chair_body)
        chair_xy = np.asarray(self.sim.data.body_xpos[body_id][:2], dtype=float)
        # Reaching the calibrated desk region is the placement condition.
        # The hold counter in _post_action filters transient crossings.
        return bool(
            np.all(np.abs(chair_xy - self.target_xy) <= self.target_tolerance)
        )

    def _success_position_condition(self):
        """Return whether the chair satisfies this task's position predicate."""
        return self._chair_in_target_region()

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold_steps >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "push the office chair in front of the desk"
        meta["success_definition"] = (
            "move the chair center into the calibrated desk target region and "
            "keep it there continuously for 1 second"
        )
        return meta


__all__ = ["HolodeckPushChairToDesk"]
