"""Scene 3: place the native wine bottle on a stool beside the robot."""

import math

import numpy as np

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec


class IThorKitchenWineBottleOntoStool(IThorKitchenEnv):
    """Move the counter's wine bottle onto the stool on the robot's other side."""

    SUPPORTED_SCENE_IDS = (3,)
    SPEC_TASK = "wine_bottle_onto_stool"
    ATOMIC_SKILL = "PickPlaceWineBottle"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False

    STOOL_BODY = "stool_80b0f89d3b5199daba395c1cfedbc901_1_0_0"
    STOOL_JOINT = STOOL_BODY + "_jntfree_0"
    STOOL_SEAT_GEOM = "stool_80b0f89d3b5199daba395c1cfedbc901_1_1_0_collision_1"
    BOTTLE_BODY = "winebottle_201eb4050e30f10e57909e4c975910e9_1_0_0"
    BOTTLE_JOINT = BOTTLE_BODY + "_jntfree_0"
    CUP_BODY = "cup_dae0e5a693f3ef8e56f0000c68e4dafe_1_0_0"
    REMOVED_BODY_NAMES = (CUP_BODY,)

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_kitchen_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "stool_xy",
                "bottle_xy",
                "hold_seconds",
                "height_tolerance",
                "footprint_tolerance",
                "max_linear_speed",
                "max_angular_speed",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        removed = self.remove_scene_bodies(self.REMOVED_BODY_NAMES)
        missing = set(self.REMOVED_BODY_NAMES) - removed
        if missing:
            raise RuntimeError(
                f"{self.scene_name}: could not remove K02 scene bodies "
                f"{sorted(missing)}"
            )

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        protected = {
            self.sim.model.body_name2id(self.STOOL_BODY),
            self.sim.model.body_name2id(self.BOTTLE_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        stool_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.STOOL_BODY)]
        )
        bottle_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.BOTTLE_BODY)]
        )
        self._stool_geom_ids = self._collidable_geoms(stool_bodies)
        self._bottle_geom_ids = self._collidable_geoms(bottle_bodies)
        self._seat_geom_id = self.sim.model.geom_name2id(self.STOOL_SEAT_GEOM)

    @staticmethod
    def _box_signs():
        return np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )

    def _geom_points(self, geom_ids):
        model, data = self.sim.model._model, self.sim.data._data
        points = []
        for geom_id in geom_ids:
            if int(model.geom_type[geom_id]) == 7:
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                local = model.mesh_vert[start : start + count]
            else:
                bounds = model.geom_aabb[geom_id]
                local = bounds[:3] + self._box_signs() * bounds[3:]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom_id])
        return np.concatenate(points)

    def _place_body_xy(self, joint_name, xy):
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:2] = np.asarray(xy, dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        start, end = self.sim.model.get_joint_qvel_addr(joint_name)
        self.sim.data.qvel[start:end] = 0.0
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        self._place_body_xy(self.STOOL_JOINT, spec["stool_xy"])
        self._place_body_xy(self.BOTTLE_JOINT, spec["bottle_xy"])

    def _seat_points(self):
        return self._geom_points([self._seat_geom_id])

    @staticmethod
    def _footprint_on_seat(bottle_points, seat_points, tolerance):
        bottle_points = np.asarray(bottle_points, dtype=float)
        seat_points = np.asarray(seat_points, dtype=float)
        lower = seat_points[:, :2].min(axis=0) - float(tolerance)
        upper = seat_points[:, :2].max(axis=0) + float(tolerance)
        return bool(
            np.all(bottle_points[:, :2] >= lower)
            and np.all(bottle_points[:, :2] <= upper)
        )

    @staticmethod
    def _stable(velocity, max_linear_speed, max_angular_speed):
        velocity = np.asarray(velocity, dtype=float)
        return bool(
            np.linalg.norm(velocity[:3]) <= float(max_linear_speed)
            and np.linalg.norm(velocity[3:]) <= float(max_angular_speed)
        )

    def _bottle_contacts_stool(self):
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair.intersection(self._bottle_geom_ids) and pair.intersection(
                self._stool_geom_ids
            ):
                return float(contact.dist) >= -0.003
        return False

    def _raw_success(self):
        spec = self._get_task_spec()
        bottle_points = self._geom_points(self._bottle_geom_ids)
        seat_points = self._seat_points()
        start, end = self.sim.model.get_joint_qvel_addr(self.BOTTLE_JOINT)
        return bool(
            self._footprint_on_seat(
                bottle_points, seat_points, spec["footprint_tolerance"]
            )
            and abs(float(bottle_points[:, 2].min()) - float(seat_points[:, 2].max()))
            <= float(spec["height_tolerance"])
            and self._bottle_contacts_stool()
            and self._stable(
                self.sim.data.qvel[start:end],
                spec["max_linear_speed"],
                spec["max_angular_speed"],
            )
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._success_hold_steps = 0

    def _post_action(self, action):
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        required = math.ceil(
            float(self._get_task_spec()["hold_seconds"]) * self.control_freq
        )
        return self._success_hold_steps >= required

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="place the wine bottle from the counter onto the stool beside you",
            task_id=type(self).__name__,
            target_entity="FloorPlan3:Wine_Bottle_1",
            target_body=self.BOTTLE_BODY,
            target_stool="FloorPlan3:Stool_2_3_1",
            removed_scene_bodies=list(self.REMOVED_BODY_NAMES),
            success_definition=(
                "the wine bottle footprint is on the stool seat, contacting it "
                "at seat height and remaining stable continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorKitchenWineBottleOntoStool"]
