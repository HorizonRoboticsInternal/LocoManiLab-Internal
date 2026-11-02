"""ProcTHOR task: carry an object between adjacent rooms."""

import numpy as np

from robocasa.environments.procthor.procthor import ProcTHOREnv
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.placement_samplers import UniformRandomSampler


def _room_label(room_type):
    chars = []
    for i, ch in enumerate(str(room_type)):
        if i > 0 and ch.isupper() and not str(room_type)[i - 1].isupper():
            chars.append(" ")
        chars.append(ch.lower())
    return "".join(chars)


class ProcTHORRoomToRoomCarry(ProcTHOREnv):
    """Pick up a box of food in one room and place it in an adjacent room."""

    ATOMIC_SKILL = "CrossRoomCarry"
    DIFFICULTY_LEVEL = "L3"
    TASK_FAMILY = "carry"
    SCENE_ID = 0
    SOURCE_NAME = "boxed_food"
    SOURCE_LANG = "box of food"
    SOURCE_SCALE = 1.0  # default — spawn points may override
    SURFACE_MARGIN = 0.03

    def __init__(self, *args, scene_ids=SCENE_ID, **kwargs):
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _add_scene_task_fixtures(self):
        for config_name in self._scene_task_spec.get("task_fixture_configs", []):
            self.add_task_fixtures_from_config(config_name)

    # ------------------------------------------------------------------ #
    # Spawn‑point selection (primary path)
    # ------------------------------------------------------------------ #

    def _pick_spawn_point(self):
        """Return a randomly chosen spawn‑point dict for the current scene, or
        None if no valid task-spec spawn points are registered."""
        pts = self._scene_task_spec.get("spawn_points", [])
        if not pts:
            return None
        rooms, _ = self.get_room_layout()
        valid = [pt for pt in pts if self.spawn_point_matches_rooms(pt, rooms)]
        if not valid:
            return None
        return valid[int(self.rng.integers(len(valid)))]

    # ------------------------------------------------------------------ #
    # Task initialisation
    # ------------------------------------------------------------------ #

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        task_spec = self.load_task_spec("room_to_room_carry")
        success_spec = task_spec.get("success", {})
        self._scene_task_spec = task_spec.get("scene_specs", {}).get(
            self.scene_name, {}
        )
        if not self._scene_task_spec:
            raise RuntimeError(
                f"{self.scene_name}: no scene spec for {type(self).__name__}. "
                "Add an entry to procthor_task_specs/room_to_room_carry.yaml."
            )
        self.remove_scene_bodies(
            set(self._scene_task_spec.get("remove_scene_bodies", []))
        )
        self._add_scene_task_fixtures()

        spawn = self._pick_spawn_point()
        if spawn is None:
            raise RuntimeError(
                f"{self.scene_name}: no valid spawn point for {type(self).__name__}. "
                "Check procthor_task_specs/room_to_room_carry.yaml."
            )

        self._spawn_override = spawn
        robot_yaw = float(spawn.get("robot_yaw", 0.0))
        self.set_robot_base_pose(spawn["robot_pos"], robot_yaw)
        self.source_room = spawn["source_room"]
        self.target_room = spawn["target_room"]
        self.source_surface = (
            np.array(spawn["source_surface_center"], dtype=float),
            np.array(spawn["source_surface_half"], dtype=float),
        )
        self.target_surface = (
            np.array(spawn["target_surface_center"], dtype=float),
            np.array(spawn["target_surface_half"], dtype=float),
        )
        self.target_is_surface = bool(spawn.get("target_is_surface", True))
        self.target_is_trash_can = bool(spawn.get("target_is_trash_can", False))
        self.target_xy_tol = np.array(
            success_spec.get("target_xy_tol", [0.12, 0.22]), dtype=float
        )
        self.target_table_z_tol = float(success_spec.get("target_table_z_tol", 0.08))
        self.release_dist = float(success_spec.get("release_dist", 0.35))
        self.trash_can_pos = np.array(
            spawn.get("trash_can_pos", spawn["target_surface_center"]), dtype=float
        )
        self.trash_can_half_height = float(
            success_spec.get("trash_can_half_height", 0.27)
        )
        self.trash_can_success_radius = float(
            success_spec.get("trash_can_success_radius", 0.14)
        )
        self.trash_can_top_tol = float(success_spec.get("trash_can_top_tol", 0.03))

        rooms, _ = self.get_room_layout()
        self.source_room_type = rooms.get(self.source_room, {}).get(
            "type", f"room {self.source_room}"
        )
        self.target_room_type = rooms.get(self.target_room, {}).get(
            "type", f"room {self.target_room}"
        )

        self.surface = self.source_surface

    # ------------------------------------------------------------------ #
    # Object placement
    # ------------------------------------------------------------------ #

    def _get_obj_cfgs(self):
        scale = self.SOURCE_SCALE
        spawn = getattr(self, "_spawn_override", None)
        if spawn is not None and "object_scale" in spawn:
            scale = float(spawn["object_scale"])
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups="boxed_food",
                object_scale=scale,
                placement=dict(margin=self.SURFACE_MARGIN),
            )
        ]

    def _build_placement_initializer(self):
        center, half_xy = self.source_surface
        margin = self.SURFACE_MARGIN
        hx = max(float(half_xy[0]) - margin, 0.0)
        hy = max(float(half_xy[1]) - margin, 0.0)
        sampler = SequentialCompositeSampler(name="RoomToRoomCarrySampler")
        sampler.append_sampler(
            UniformRandomSampler(
                name=f"{self.SOURCE_NAME}_Sampler",
                mujoco_objects=self.objects[self.SOURCE_NAME],
                x_range=[-hx, hx],
                y_range=[-hy, hy],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=center,
                z_offset=0.01,
            )
        )
        return sampler

    # ------------------------------------------------------------------ #
    # Success
    # ------------------------------------------------------------------ #

    def _check_obj_in_trash_can(self):
        bbox = self.object_bbox_points(self.SOURCE_NAME)
        bbox_center = np.mean(bbox, axis=0)
        bottom_z = float(np.min(bbox[:, 2]))
        top_z = float(np.max(bbox[:, 2]))
        can_bottom_z = float(self.trash_can_pos[2] - self.trash_can_half_height)
        can_top_z = float(self.trash_can_pos[2] + self.trash_can_half_height)
        center_dist = float(np.linalg.norm(bbox_center[:2] - self.trash_can_pos[:2]))

        inside_xy = center_dist <= self.trash_can_success_radius
        center_inside_height = can_bottom_z + 0.04 <= bbox_center[2] <= can_top_z
        bottom_below_rim = bottom_z <= can_top_z - self.trash_can_top_tol
        not_sticking_out_too_high = top_z <= can_top_z + self.trash_can_top_tol
        return bool(
            inside_xy
            and center_inside_height
            and bottom_below_rim
            and not_sticking_out_too_high
        )

    def _check_success(self):
        not_holding = self.object_released(self.SOURCE_NAME, th=self.release_dist)
        if getattr(self, "target_is_trash_can", False):
            return bool(self._check_obj_in_trash_can() and not_holding)

        obj_pos = self.object_pos(self.SOURCE_NAME)
        target_center, _ = self.target_surface
        planar = np.abs(obj_pos[:2] - target_center[:2])
        on_target_xy = np.all(planar <= self.target_xy_tol)
        on_target_z = (
            abs(self.object_bottom_z(self.SOURCE_NAME) - float(target_center[2]))
            <= self.target_table_z_tol
        )

        return bool(on_target_xy and on_target_z and not_holding)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["source_room"] = self.source_room
        ep_meta["target_room"] = self.target_room
        ep_meta["source_surface"] = [float(v) for v in self.source_surface[0]]
        ep_meta["target_surface"] = [float(v) for v in self.target_surface[0]]
        ep_meta["target_is_surface"] = self.target_is_surface
        ep_meta["target_is_trash_can"] = getattr(self, "target_is_trash_can", False)
        if ep_meta["target_is_trash_can"]:
            target_phrase = "in the trash can there"
        elif self.target_is_surface:
            target_phrase = "on the surface there"
        else:
            target_phrase = "at the target spot there"
        ep_meta["lang"] = (
            f"pick up the {self.SOURCE_LANG} from the "
            f"{_room_label(self.source_room_type)} and carry it into the adjacent "
            f"{_room_label(self.target_room_type)}, then place it {target_phrase}"
        )
        return ep_meta
