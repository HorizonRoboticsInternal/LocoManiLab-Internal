"""Scene 2209 alarm-clock lift and placement tasks."""

import copy
import itertools
import math

import mujoco
import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckAlarmClockTask(HolodeckEnv):
    """Share one native clock's physical configuration across the two tasks."""

    ABSTRACT = True
    SUPPORTED_SCENE_IDS = (2209,)
    TARGET_BODY = "alarmclock_6ea5bab9e39833dfe9a00a0f58a36c75_1_0_0"
    # Native front-to-back grasp span is 0.084263 m. Uniform scaling gives
    # 0.074994 m while retaining the bells, feet and collision geometry.
    CLOCK_SCALE = 0.89
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    NOISE_FREE_ROBOT_SPAWN = True

    def __init__(self, *args, scene_ids=None, **kwargs):
        self._success_hold = 0
        super().__init__(
            *args,
            scene_ids=self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids,
            **kwargs,
        )

    def _setup_baked_scene_fixtures(self):
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("source_body", "clock_pose", "robot_pose"),
        )
        clock = self.find_scene_body(self.TARGET_BODY)
        if clock is None:
            raise RuntimeError(f"{self.scene_name}: missing native alarm clock")
        asset = self.mujoco_arena.root.find("asset")
        for name in sorted(
            {g.get("mesh") for g in clock.iter("geom") if g.get("mesh")}
        ):
            mesh = asset.find(f"./mesh[@name='{name}']")
            if mesh is None:
                raise RuntimeError(f"{self.scene_name}: missing clock mesh {name}")
            clone = copy.deepcopy(mesh)
            clone.set("name", f"{name}_holodeck2209_clock")
            scale = np.fromstring(clone.get("scale", "1 1 1"), sep=" ")
            clone.set("scale", " ".join(map(str, scale * self.CLOCK_SCALE)))
            asset.append(clone)
            for geom in clock.iter("geom"):
                if geom.get("mesh") == name:
                    geom.set("mesh", clone.get("name"))
        for element in clock.iter():
            if element is clock:
                continue
            for attr in ("pos", "size"):
                value = element.get(attr)
                if value and element.tag in ("body", "geom", "site"):
                    values = np.fromstring(value, sep=" ") * self.CLOCK_SCALE
                    element.set(attr, " ".join(map(str, values)))
        pose = self.task_spec["clock_pose"]
        clock.set("pos", " ".join(map(str, pose["pos"])))
        clock.set("quat", " ".join(map(str, pose["quat"])))
        if "drawer_joint" in self.task_spec:
            joint = self.mujoco_arena.worldbody.find(
                f".//joint[@name='{self.task_spec['drawer_joint']}']"
            )
            if joint is None or joint.get("type") != "slide":
                raise RuntimeError(f"{self.scene_name}: missing sliding desk drawer")
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        if self.find_scene_body(self.task_spec["source_body"]) is None:
            raise RuntimeError(f"{self.scene_name}: missing clock support")
        pose = self.task_spec["robot_pose"]
        pos = np.array(pose["pos"], dtype=float)
        robot_model = self.robots[0].robot_model
        if type(robot_model).__name__ == "G1":
            # G1's mobile base already elevates the physical pelvis. Cancel
            # set_base_xpos's second subtraction only for these new tasks.
            # G1FloatingBody / G1FixedLowerBody already use bottom_offset=0.
            pos += robot_model.bottom_offset
        self.set_robot_base_pose(pos, pose["yaw"])

    def _setup_references(self):
        super()._setup_references()
        model, data = self.sim.model, self.sim.data
        self._clock_body_id = model.body_name2id(self.TARGET_BODY)
        self._clock_body_ids = self._body_tree_ids([self._clock_body_id])
        self._clock_geoms = self._geom_names_for_bodies(self._clock_body_ids)
        joint_id = int(model.body_jntadr[self._clock_body_id])
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("alarm clock must have a free joint")
        self._clock_joint = model.joint_id2name(joint_id)
        self._hand_geoms = {
            side: self._geom_names_for_bodies(
                self._body_ids_matching((f"{side}_hand",))
            )
            for side in ("left", "right")
        }
        # Cache the native clock's complete local bounds, including its bells
        # and feet. Unlike individual rotated geom AABBs, this envelope remains
        # meaningful when projected into the drawer's moving site frame.
        self.sim.forward()
        points = []
        for geom_id in self._geom_ids_for_bodies(self._clock_body_ids):
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise RuntimeError("expected native alarm clock mesh geometry")
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[start : start + count])
            rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            points.append(vertices @ rotation.T + data.geom_xpos[geom_id])
        root_rotation = np.asarray(data.body_xmat[self._clock_body_id]).reshape(3, 3)
        local = (
            np.concatenate(points) - data.body_xpos[self._clock_body_id]
        ) @ root_rotation
        lower, upper = local.min(axis=0), local.max(axis=0)
        self._clock_local_bounds = np.array(list(itertools.product(*zip(lower, upper))))
        if "drawer_joint" in self.task_spec:
            self._drawer_site_id = model.site_name2id(self.task_spec["drawer_site"])
            self._drawer_bottom_id = model.geom_name2id(
                self.task_spec["drawer_bottom_geom"]
            )

    def _clock_points(self):
        data = self.sim.data
        rotation = np.asarray(data.body_xmat[self._clock_body_id]).reshape(3, 3)
        return (
            self._clock_local_bounds @ rotation.T + data.body_xpos[self._clock_body_id]
        )

    def _clock_grasped(self):
        return any(
            self.g1_hand_closed(side)
            and self.check_contact(self._hand_geoms[side], self._clock_geoms)
            for side in ("left", "right")
        )

    def _reset_internal(self):
        self._success_hold = 0
        super()._reset_internal()
        pose = self.task_spec["clock_pose"]
        self.sim.data.set_joint_qpos(
            self._clock_joint, np.array([*pose["pos"], *pose["quat"]])
        )
        self.sim.data.set_joint_qvel(self._clock_joint, np.zeros(6))
        if "drawer_joint" in self.task_spec:
            self.sim.data.set_joint_qpos(
                self.task_spec["drawer_joint"], self.task_spec["drawer_open_qpos"]
            )
            self.sim.data.set_joint_qvel(self.task_spec["drawer_joint"], 0.0)
        self.sim.forward()
        samples = [self._robot_base_xy()]
        samples.extend(
            self._robot_base_xy() + 0.25 * np.array([np.cos(a), np.sin(a)])
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        )
        if any(self.room_of_point([*xy, 0.0]) is None for xy in samples):
            raise RuntimeError(
                f"{self.scene_name}: alarm-clock task robot starts off the floor"
            )
        if self.count_body_collisions() or self._clock_grasped():
            raise RuntimeError(
                f"{self.scene_name}: alarm-clock task has a spawn collision"
            )
        self._clock_initial_z = float(self.sim.data.body_xpos[self._clock_body_id][2])

    @property
    def hold_steps(self):
        return max(1, int(math.ceil(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        self._success_hold = self._success_hold + 1 if self._goal_satisfied() else 0
        return super()._post_action(action)

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.LANG,
            target_body=self.TARGET_BODY,
            object_model="Alarm_Clock_4",
            object_scale=self.CLOCK_SCALE,
            grasp_span=0.084263 * self.CLOCK_SCALE,
            source_body=self.task_spec["source_body"],
            hold_seconds=self.HOLD_SECONDS,
        )
        return meta


class HolodeckPickAlarmClockFromSofa(_HolodeckAlarmClockTask):
    """Lift the alarm clock on the pink sofa by 30 cm for one second."""

    ATOMIC_SKILL = "PickObject"
    TASK_FAMILY = "pick"
    SPEC_TASK = "pick_alarm_clock_from_sofa"
    MIN_LIFT_DELTA = 0.30
    # This notebook falls off the desk after reset; omit it only for this task.
    HIDDEN_SCENE_BODIES = {
        **HolodeckEnv.HIDDEN_SCENE_BODIES,
        "train_2209": (
            *HolodeckEnv.HIDDEN_SCENE_BODIES["train_2209"],
            "obja_e573304fce364cf299027494fc1afede_1_0_0",
        ),
    }
    LANG = "Pick up the alarm clock from the pink sofa, lift it at least 30 cm. "

    def _goal_satisfied(self):
        height = float(self.sim.data.body_xpos[self._clock_body_id][2])
        return bool(height - self._clock_initial_z >= self.MIN_LIFT_DELTA)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lift_delta=self.MIN_LIFT_DELTA,
            success_definition=(
                "the alarm clock rises at least 0.30 m above its reset "
                "height for 1 second"
            ),
        )
        return meta


class HolodeckPlaceAlarmClockInDrawer(_HolodeckAlarmClockTask):
    """Place the desk's alarm clock in its initially open top drawer."""

    ATOMIC_SKILL = "PlaceObject"
    TASK_FAMILY = "place"
    SPEC_TASK = "place_alarm_clock_in_drawer"
    LANG = "Put the alarm clock on the desk into the open top drawer of the same desk."

    def _goal_satisfied(self):
        # Only continuous contact with the named drawer floor counts. Grasp
        # history, hand state, containment and velocity are not prerequisites.
        bottom = self.sim.model.geom_id2name(self._drawer_bottom_id)
        return bool(self.check_contact([bottom], self._clock_geoms))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            drawer_body=self.task_spec["drawer_body"],
            drawer_joint=self.task_spec["drawer_joint"],
            drawer_region=self.task_spec["drawer_site"],
            success_definition=(
                "the alarm clock contacts the top drawer support surface "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["HolodeckPickAlarmClockFromSofa", "HolodeckPlaceAlarmClockInDrawer"]
