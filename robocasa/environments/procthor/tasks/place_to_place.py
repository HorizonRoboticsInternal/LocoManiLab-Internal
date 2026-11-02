"""Shared helpers for simple ProcTHOR place-to-place tasks."""

import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.procthor.procthor import ProcTHOREnv
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


def room_label(room_type):
    out = []
    for i, ch in enumerate(str(room_type)):
        if i and ch.isupper() and not str(room_type)[i - 1].isupper():
            out.append(" ")
        out.append(ch.lower())
    return "".join(out)


class ProcTHORPlaceToPlaceTask(ProcTHOREnv):
    TABLE_HALF = np.array([0.45, 0.32])
    TABLE_HEIGHT = 0.92
    GRID_STEP = 0.45
    ROBOT_STAND_DISTANCE = 0.95
    ROBOT_SAFE_HALF = 0.04
    TABLE_WALL_CLEARANCE = 0.10
    ITEM_JITTER = 0.12
    HOLD_STEPS = 5
    RELEASE_DIST = 0.35
    REQUIRE_RELEASE = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("obj_registries", ("aigen", "objaverse", "lightwheel"))
        super().__init__(*args, **kwargs)

    # ---- room / floor placement -------------------------------------------------

    def _pick_room_pair(self):
        rooms, edges = self.get_room_layout()
        if len(rooms) < 2:
            return None, None
        pairs = [(a, b) for a, b in edges if a in rooms and b in rooms and a != b]
        if not pairs:
            ids = sorted(rooms)
            pairs = [(a, b) for a in ids for b in ids if a != b]
        return max(
            pairs, key=lambda pair: self._room_area(pair[0]) + self._room_area(pair[1])
        )

    def _room_area(self, room_id):
        poly = np.asarray(self.get_room_layout()[0][room_id]["poly"], dtype=float)
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _free_point(
        self,
        room_id,
        margin,
        require_robot_space=False,
        require_table_space=False,
        safe_half=0.0,
        ignore_body_names=(),
    ):
        for point in sorted(
            self._room_grid(room_id, margin),
            key=lambda p: np.linalg.norm(p - self._room_center(room_id)),
        ):
            if (
                self._point_is_clear(point, ignore_body_names=ignore_body_names)
                and (
                    not require_robot_space
                    or self._robot_space_is_clear(point, room_id, ignore_body_names)
                )
                and (
                    not require_table_space
                    or self._table_space_is_clear(point, room_id)
                )
                and self._safe_box_in_room(point, room_id, safe_half)
            ):
                return point
        return None

    def _room_grid(self, room_id, margin):
        poly = np.asarray(self.get_room_layout()[0][room_id]["poly"], dtype=float)
        lo, hi = np.min(poly, axis=0), np.max(poly, axis=0)
        for x in np.arange(lo[0] + margin, hi[0] - margin + 1e-6, self.GRID_STEP):
            for y in np.arange(lo[1] + margin, hi[1] - margin + 1e-6, self.GRID_STEP):
                point = np.array([x, y])
                if self._in_poly(point, poly):
                    yield point

    def _point_is_clear(self, point, ignore_body_names=()):
        ignore = set(ignore_body_names)
        for fixture in self.semantic_scene.get_fixture(return_all=True):
            if fixture.body_name in ignore:
                continue
            if np.linalg.norm(np.asarray(fixture.center[:2]) - point) < 0.45:
                return False
        return True

    def _table_front(self, table_xy, room_id):
        front = self._room_center(room_id) - np.asarray(table_xy)
        norm = np.linalg.norm(front)
        return front / norm if norm > 1e-6 else np.array([1.0, 0.0])

    def _robot_space_is_clear(self, table_xy, room_id, ignore_body_names=()):
        robot_xy = (
            np.asarray(table_xy)
            + self._table_front(table_xy, room_id) * self.ROBOT_STAND_DISTANCE
        )
        return self._point_is_clear(
            robot_xy, ignore_body_names=ignore_body_names
        ) and self._safe_box_in_room(robot_xy, room_id, self.ROBOT_SAFE_HALF)

    def _table_space_is_clear(self, table_xy, room_id):
        return self._safe_rect_in_room(
            table_xy, room_id, self.TABLE_HALF + self.TABLE_WALL_CLEARANCE
        )

    def _safe_box_in_room(self, center_xy, room_id, half):
        if half <= 0:
            return True
        return self._safe_rect_in_room(center_xy, room_id, np.array([half, half]))

    def _safe_rect_in_room(self, center_xy, room_id, half_xy):
        poly = np.asarray(self.get_room_layout()[0][room_id]["poly"], dtype=float)
        corners = (
            np.asarray(center_xy) + np.array([sx * half_xy[0], sy * half_xy[1]])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )
        return all(self._in_poly(corner, poly) for corner in corners)

    def _room_center(self, room_id):
        return np.mean(np.asarray(self.get_room_layout()[0][room_id]["poly"]), axis=0)

    @staticmethod
    def _in_poly(point, poly):
        x, y = float(point[0]), float(point[1])
        inside, j = False, len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                if x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                    inside = not inside
            j = i
        return inside

    # ---- generated tables / object placement -----------------------------------

    def _add_table(self, body_name, label, room_id):
        body = ET.SubElement(self.mujoco_arena.worldbody, "body", name=body_name)
        geoms = [
            ("top", [0, 0, self.TABLE_HEIGHT - 0.04], [*self.TABLE_HALF, 0.04]),
            (
                "leg_0",
                [0.40, 0.27, self.TABLE_HEIGHT / 2],
                [0.035, 0.035, self.TABLE_HEIGHT / 2],
            ),
            (
                "leg_1",
                [-0.40, 0.27, self.TABLE_HEIGHT / 2],
                [0.035, 0.035, self.TABLE_HEIGHT / 2],
            ),
            (
                "leg_2",
                [0.40, -0.27, self.TABLE_HEIGHT / 2],
                [0.035, 0.035, self.TABLE_HEIGHT / 2],
            ),
            (
                "leg_3",
                [-0.40, -0.27, self.TABLE_HEIGHT / 2],
                [0.035, 0.035, self.TABLE_HEIGHT / 2],
            ),
        ]
        for name, pos, size in geoms:
            for suffix, attrs in (
                ("", {}),
                ("_visual", {"group": "1", "contype": "0", "conaffinity": "0"}),
            ):
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{body_name}_{name}{suffix}",
                    type="box",
                    pos=" ".join(map(str, pos)),
                    size=" ".join(map(str, size)),
                    rgba="0.45 0.31 0.18 1",
                    friction="0.9 0.9 0.001",
                    **attrs,
                )

        stable = f"{self.scene_name}:task:{label}_table"
        region = SemanticRegion(
            stable_id=f"{stable}:region:support",
            fixture_id=stable,
            kind="support",
            parent_body=body_name,
            local_center=np.array([0.0, 0.0, self.TABLE_HEIGHT]),
            half_size=np.array([*self.TABLE_HALF, 0.02]),
        )
        fixture = SemanticFixture(
            stable_id=stable,
            name=body_name,
            category="table",
            source_category="TaskTable",
            room_id=room_id,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=body_name,
            center=np.zeros(3),
            half_size=np.array([*self.TABLE_HALF, self.TABLE_HEIGHT / 2]),
            yaw=0.0,
            front=np.array([1.0, 0.0]),
            capabilities={"support"},
            regions={region.stable_id: region},
            local_center=np.array([0.0, 0.0, self.TABLE_HEIGHT / 2]),
        )
        self.semantic_scene.add_fixture(fixture)
        if label == "source":
            self.source_fixture, self.source_region = fixture, region
        else:
            self.target_fixture, self.target_region = fixture, region
        return fixture, region

    def _move_body(self, body_name, xy, z=0.0):
        self.sim.model.body_pos[self.sim.model.body_name2id(body_name)] = [
            xy[0],
            xy[1],
            z,
        ]

    def _update_table_semantics(self, fixture, region, table_xy, room_id):
        fixture.center = np.array([table_xy[0], table_xy[1], self.TABLE_HEIGHT / 2])
        fixture.front = self._table_front(table_xy, room_id)
        region._world_center = np.array([table_xy[0], table_xy[1], self.TABLE_HEIGHT])

    def _sample_region_offset(self, region, near_robot=False):
        half = np.asarray(region.half_size[:2], dtype=float)
        if not near_robot:
            return np.array(
                [
                    self.rng.uniform(-self.ITEM_JITTER, self.ITEM_JITTER),
                    self.rng.uniform(-self.ITEM_JITTER, self.ITEM_JITTER),
                ]
            )

        center, yaw = region.world_pose(self.sim)
        delta = np.asarray(self.robot_base_pos[:2], dtype=float) - center[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_robot = np.array(
            [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]]
        )
        axis = int(np.argmax(np.abs(local_robot)))
        sign = 1.0 if local_robot[axis] >= 0.0 else -1.0
        ranges = [[-half[0], half[0]], [-half[1], half[1]]]
        ranges[axis] = [0.0, sign * half[axis]]
        lo, hi = sorted(ranges[axis])
        ranges[axis] = [lo, hi]
        return np.array(
            [
                self.rng.uniform(ranges[0][0], ranges[0][1]),
                self.rng.uniform(ranges[1][0], ranges[1][1]),
            ]
        )

    def _place_object_on_region(self, region, z_offset=0.16, near_robot=False):
        center, yaw = region.world_pose(self.sim)
        offset_xy = self._sample_region_offset(region, near_robot=near_robot)
        c, s = np.cos(yaw), np.sin(yaw)
        world_offset_xy = np.array(
            [
                c * offset_xy[0] - s * offset_xy[1],
                s * offset_xy[0] + c * offset_xy[1],
            ]
        )
        qpos = self.sim.data.get_joint_qpos(
            self.objects[self.SOURCE_NAME].joints[0]
        ).copy()
        qpos[:3] = center + np.array([world_offset_xy[0], world_offset_xy[1], z_offset])
        self.sim.data.set_joint_qpos(self.objects[self.SOURCE_NAME].joints[0], qpos)
        self.sim.forward()

    def _place_robot_by_fixture(self, fixture):
        target = fixture.world_center(self.sim)
        front = np.asarray(fixture.front, dtype=float)
        front = front / (np.linalg.norm(front) + 1e-9)
        xy = target[:2] + front * self.ROBOT_STAND_DISTANCE
        yaw = float(np.arctan2(target[1] - xy[1], target[0] - xy[0]))
        if self._set_robot_free_pose(xy, yaw):
            self._calibrate_robot_free_joint_height()
            self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
            self.robot_base_ori = np.array([0.0, 0.0, yaw])
            return True
        return False

    # ---- success ---------------------------------------------------------------

    def object_on_target_table(self, z_above=0.12, require_release=None):
        bbox = self.object_bbox_points(self.SOURCE_NAME)
        center = np.mean(bbox, axis=0)
        bottom = float(np.min(bbox[:, 2]))
        target_center, _ = self.target_region.world_pose(self.sim)
        half = self.target_region.half_size
        if not (
            abs(float(center[0] - target_center[0])) <= float(half[0])
            and abs(float(center[1] - target_center[1])) <= float(half[1])
            and self.TABLE_HEIGHT - 0.08 <= bottom <= self.TABLE_HEIGHT + z_above
        ):
            return False
        if self.REQUIRE_RELEASE if require_release is None else require_release:
            return self.object_released(self.SOURCE_NAME, th=self.RELEASE_DIST)
        return True

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._raw_success() else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS


__all__ = ["ProcTHORPlaceToPlaceTask", "room_label"]
