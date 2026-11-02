"""Tabletop cylinder/socket insertion task."""

import numpy as np

from robosuite.models.objects import CompositeObject, CylinderObject
from robosuite.utils.mjcf_utils import array_to_string

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures.counter import Counter


class TabletopCylinderSocket(LivingRoomEnv):
    COUNTER_SIZE = (0.45, 0.90, 0.78)
    COUNTER_POS = (0.8, 0.0)

    SOCKET_OFFSET = (-0.08, 0.12)
    CYLINDER_OFFSET = (-0.13, -0.16)
    SOCKET_INNER_RADIUS = 0.034
    SOCKET_OUTER_RADIUS = 0.050
    SOCKET_HALF_HEIGHT = 0.035
    SOCKET_SEGMENTS = 16

    CYLINDER_RADIUS = 0.024
    CYLINDER_HALF_HEIGHT = 0.080
    CYLINDER_Z_CLEARANCE = 0.006

    INSERT_XY_TOL = 0.012
    INSERT_BOTTOM_RANGE = (-0.006, 0.030)
    INSERT_VERTICAL_DOT = 0.92
    RELEASE_DISTANCE = 0.12
    HOLD_SECONDS = 1.0

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 7)
        kwargs.setdefault("style_ids", 1)
        kwargs.setdefault("robot_base_pos", (0.30, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()

        counter = Counter(
            name="table",
            size=list(self.COUNTER_SIZE),
            top_texture="textures/marble/marble_2.png",
            base_texture="textures/wood/warm_wood_grain.png",
            rng=self.rng,
        )
        counter_z = -counter._regions["main"]["p0"][2]
        counter.set_pos((self.COUNTER_POS[0], self.COUNTER_POS[1], counter_z))
        counter.set_euler((0.0, 0.0, 0.0))
        self.furniture["table"] = counter

        self.socket = self._make_socket(counter)
        self.furniture["cylinder_socket"] = self.socket
        self.fixtures = self.furniture

    def _make_socket(self, counter):
        geom_types = []
        geom_sizes = []
        geom_locations = []
        geom_quats = []
        geom_names = []
        geom_rgbas = []
        obj_types = []

        wall_half_thickness = (self.SOCKET_OUTER_RADIUS - self.SOCKET_INNER_RADIUS) / 2
        wall_half_length = self.SOCKET_OUTER_RADIUS * np.sin(
            np.pi / self.SOCKET_SEGMENTS
        )
        wall_center_radius = self.SOCKET_INNER_RADIUS + wall_half_thickness
        for idx in range(self.SOCKET_SEGMENTS):
            angle = idx * 2 * np.pi / self.SOCKET_SEGMENTS
            geom_types.append("box")
            geom_sizes.append(
                [wall_half_thickness, wall_half_length, self.SOCKET_HALF_HEIGHT]
            )
            geom_locations.append(
                [
                    wall_center_radius * np.cos(angle),
                    wall_center_radius * np.sin(angle),
                    0.0,
                ]
            )
            geom_quats.append([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
            geom_names.append(f"socket_wall_{idx}")
            geom_rgbas.append([0.56, 0.56, 0.60, 1.0])
            obj_types.append("all")

        geom_types.append("cylinder")
        geom_sizes.append([self.SOCKET_INNER_RADIUS * 0.92, 0.001])
        geom_locations.append([0.0, 0.0, -self.SOCKET_HALF_HEIGHT - 0.002])
        geom_quats.append(None)
        geom_names.append("socket_hole_shadow")
        geom_rgbas.append([0.035, 0.035, 0.040, 1.0])
        obj_types.append("visual")

        socket = CompositeObject(
            name="cylinder_socket",
            total_size=[
                self.SOCKET_OUTER_RADIUS,
                self.SOCKET_OUTER_RADIUS,
                self.SOCKET_HALF_HEIGHT,
            ],
            geom_types=geom_types,
            geom_sizes=geom_sizes,
            geom_locations=geom_locations,
            geom_quats=geom_quats,
            geom_names=geom_names,
            geom_rgbas=geom_rgbas,
            obj_types=obj_types,
            locations_relative_to_center=True,
            joints=None,
            duplicate_collision_geoms=False,
        )

        table_top_z = self._table_top_z(counter)
        socket._obj.set(
            "pos",
            array_to_string(
                (
                    self.COUNTER_POS[0] + self.SOCKET_OFFSET[0],
                    self.COUNTER_POS[1] + self.SOCKET_OFFSET[1],
                    table_top_z + self.SOCKET_HALF_HEIGHT,
                )
            ),
        )
        socket._obj.set("euler", array_to_string((0.0, 0.0, 0.0)))
        return socket

    def _setup_furniture_references(self):
        self.table = self.register_furniture_ref("table", dict(id="table"))

    def _create_objects(self):
        self.object_cfgs = []
        self.objects = {}
        cylinder = CylinderObject(
            name="insert_cylinder",
            size=[self.CYLINDER_RADIUS, self.CYLINDER_HALF_HEIGHT],
            rgba=[0.90, 0.26, 0.18, 1.0],
            density=100,
            friction=[1.0, 0.01, 0.001],
            joints=[dict(type="free", damping="0.0005")],
            duplicate_collision_geoms=False,
        )
        self.objects[cylinder.name] = cylinder
        self.model.merge_objects([cylinder])

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold = 0
        if hasattr(self, "sim") and self.sim is not None:
            self._place_cylinder()

    def _table_top_z(self, table=None):
        table = table if table is not None else self.table
        return float(table.pos[2] + table._regions["main"]["pz"][2])

    def _world_from_table_offset(self, offset, z):
        return np.array(
            [
                self.COUNTER_POS[0] + float(offset[0]),
                self.COUNTER_POS[1] + float(offset[1]),
                float(z),
            ]
        )

    def _socket_center(self):
        return self._world_from_table_offset(
            self.SOCKET_OFFSET,
            self._table_top_z(),
        )

    def _place_cylinder(self):
        cylinder = self.objects["insert_cylinder"]
        qpos = self.sim.data.get_joint_qpos(cylinder.joints[0]).copy()
        qpos[:3] = self._world_from_table_offset(
            self.CYLINDER_OFFSET,
            self._table_top_z() + self.CYLINDER_HALF_HEIGHT + self.CYLINDER_Z_CLEARANCE,
        )
        qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos(cylinder.joints[0], qpos)
        joint_id = self.sim.model.joint_name2id(cylinder.joints[0])
        qvel_addr = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        self.sim.forward()

    def _cylinder_center(self):
        body_id = self.sim.model.body_name2id(self.objects["insert_cylinder"].root_body)
        return np.array(self.sim.data.body_xpos[body_id])

    def _cylinder_axis(self):
        body_id = self.sim.model.body_name2id(self.objects["insert_cylinder"].root_body)
        return np.array(self.sim.data.body_xmat[body_id]).reshape(3, 3)[:, 2]

    def _eef_positions(self):
        site_ids = self.robots[0].eef_site_id
        site_ids = site_ids.values() if isinstance(site_ids, dict) else [site_ids]
        return [np.array(self.sim.data.site_xpos[site_id]) for site_id in site_ids]

    def _cylinder_released(self):
        center = self._cylinder_center()
        return all(
            np.linalg.norm(eef_pos - center) > self.RELEASE_DISTANCE
            for eef_pos in self._eef_positions()
        )

    def _cylinder_inserted(self):
        center = self._cylinder_center()
        socket_center = self._socket_center()
        bottom_z = center[2] - self.CYLINDER_HALF_HEIGHT
        bottom_depth = bottom_z - self._table_top_z()
        vertical = abs(float(np.dot(self._cylinder_axis(), np.array([0.0, 0.0, 1.0]))))
        return bool(
            np.linalg.norm(center[:2] - socket_center[:2]) <= self.INSERT_XY_TOL
            and self.INSERT_BOTTOM_RANGE[0]
            <= bottom_depth
            <= self.INSERT_BOTTOM_RANGE[1]
            and vertical >= self.INSERT_VERTICAL_DOT
            and self._cylinder_released()
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._cylinder_inserted() else 0
        return ret

    def _check_success(self):
        return self._success_hold >= int(round(self.control_freq * self.HOLD_SECONDS))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "pick up the upright cylinder and insert it into the cylindrical socket"
        return ep_meta


__all__ = ["TabletopCylinderSocket"]
