"""Tabletop charger/socket insertion prototype."""

import os

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import Fixture
from robocasa.models.fixtures.counter import Counter


class TabletopChargerSocket(LivingRoomEnv):
    COUNTER_SIZE = (0.45, 0.90, 0.78)
    COUNTER_POS = (0.8, 0.0)
    CHARGER_XML = "usd/charger/converted/model.xml"
    SOCKET_XML = "usd/socket/converted/model.xml"
    CHARGER_SCALE = 1.5
    SOCKET_SCALE = 1.5
    SOCKET_OFFSET = (-0.10, 0.12)
    CHARGER_OFFSET = (-0.13, -0.15)
    SOCKET_SITES = ("socket_0_site", "socket_1_site", "socket_2_site")
    PRONG_HALF_SPACING = 0.00587 * CHARGER_SCALE
    INSERT_RADIAL_TOL = 0.008 * CHARGER_SCALE
    INSERT_DEPTH_RANGE = (0.001 * CHARGER_SCALE, 0.025 * CHARGER_SCALE)
    INSERT_ALIGN_DOT = 0.82

    def __init__(self, *args, **kwargs):
        if kwargs.get("layout_ids") is None:
            kwargs["layout_ids"] = 7
        if kwargs.get("style_ids") is None:
            kwargs["style_ids"] = 1
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
        self.socket = self._make_socket_fixture(counter)
        self.furniture["socket"] = self.socket
        self.fixtures = self.furniture

    def _make_socket_fixture(self, counter):
        socket = Fixture(
            xml=os.path.join("objects", self.SOCKET_XML),
            name="socket",
            duplicate_collision_geoms=False,
            joints=None,
            scale=self.SOCKET_SCALE,
        )
        table_top_z = counter.pos[2] + counter._regions["main"]["pz"][2]
        socket_z = table_top_z - socket.bottom_offset[2] + 0.001
        socket.set_pos(
            (
                self.COUNTER_POS[0] + self.SOCKET_OFFSET[0],
                self.COUNTER_POS[1] + self.SOCKET_OFFSET[1],
                socket_z,
            )
        )
        socket.set_euler((0.0, 0.0, np.pi / 2))
        return socket

    def _setup_furniture_references(self):
        self.table = self.register_furniture_ref("table", dict(id="table"))

    def _get_obj_cfgs(self):
        return [
            dict(
                name="charger",
                obj_groups="charger",
                info=dict(mjcf_path=self.CHARGER_XML),
                object_scale=self.CHARGER_SCALE,
                placement=dict(
                    fixture=self.table,
                    size=(0.12, 0.16),
                    offset=self.CHARGER_OFFSET,
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            )
        ]

    def sample_object(self, groups, **kwargs):
        if (
            isinstance(groups, str)
            and groups.endswith(self.CHARGER_XML)
            and os.path.exists(groups)
        ):
            scale = kwargs.get("object_scale", None) or 1.0
            return (
                dict(
                    mjcf_path=groups,
                    scale=scale,
                ),
                dict(
                    mjcf_path=groups,
                    cat="charger",
                    groups=[],
                    groups_containing_sampled_obj=[],
                    split=None,
                ),
            )
        return super().sample_object(groups, **kwargs)

    def _site_xpos(self, obj_name, site_suffix):
        for site_id in range(self.sim.model.nsite):
            site_name = self.sim.model.site_id2name(site_id) or ""
            if obj_name in site_name and site_suffix in site_name:
                return np.array(self.sim.data.site_xpos[site_id])
        return None

    def _site_xmat(self, obj_name, site_suffix):
        for site_id in range(self.sim.model.nsite):
            site_name = self.sim.model.site_id2name(site_id) or ""
            if obj_name in site_name and site_suffix in site_name:
                return np.array(self.sim.data.site_xmat[site_id]).reshape(3, 3)
        return None

    def _check_success(self):
        left_tip = self._site_xpos("charger", "left_prong_tip_site")
        right_tip = self._site_xpos("charger", "right_prong_tip_site")
        left_base = self._site_xpos("charger", "left_prong_base_site")
        right_base = self._site_xpos("charger", "right_prong_base_site")
        prong_sites = (left_tip, right_tip, left_base, right_base)
        if any(site is None for site in prong_sites):
            return False

        prongs = ((left_base, left_tip), (right_base, right_tip))

        def inserted(tip, target, socket_normal):
            delta = tip - target
            depth = -float(np.dot(delta, socket_normal))
            radial = float(np.linalg.norm(delta + depth * socket_normal))
            return (
                radial < self.INSERT_RADIAL_TOL
                and self.INSERT_DEPTH_RANGE[0] < depth < self.INSERT_DEPTH_RANGE[1]
            )

        def aligned(base, tip, socket_normal):
            axis = tip - base
            axis /= np.linalg.norm(axis) + 1e-8
            return float(np.dot(axis, -socket_normal)) > self.INSERT_ALIGN_DOT

        for site_name in self.SOCKET_SITES:
            socket_center = self._site_xpos("socket", site_name)
            socket_mat = self._site_xmat("socket", site_name)
            if socket_center is None or socket_mat is None:
                continue

            socket_normal = socket_mat[:, 2]
            socket_tangent = socket_mat[:, 0]
            targets = (
                socket_center - socket_tangent * self.PRONG_HALF_SPACING,
                socket_center + socket_tangent * self.PRONG_HALF_SPACING,
            )

            direct = inserted(left_tip, targets[0], socket_normal) and inserted(
                right_tip, targets[1], socket_normal
            )
            swapped = inserted(left_tip, targets[1], socket_normal) and inserted(
                right_tip, targets[0], socket_normal
            )
            if (direct or swapped) and all(
                aligned(base, tip, socket_normal) for base, tip in prongs
            ):
                return True

        return False

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the charger and insert it into the socket"
        return ep_meta
