"""StoreMugInCabinet: pick the mug off the counter (among a couple of distractor
objects), carry it a short distance to the storage cabinet, open the cabinet
door and put the mug inside.

The "table" is a kitchen ``Counter`` -- a procedurally generated worktop whose
collision is clean axis-aligned boxes that match the visual exactly (unlike the
living-room furniture meshes, whose auto-generated *full bounding box* collision
boxed the robot in). It is the native object-placement surface kitchen tasks use
(``placement=dict(fixture=counter, ...)``), at a comfortable standing height,
directly in FRONT of the robot. The cabinet is a kitchen ``SingleCabinet`` (a
real single-hinged-door storage cabinet with interior shelves) standing on the
floor directly BEHIND the robot, its door turned to face the robot. So the robot
spawns BETWEEN the two, facing the counter: it picks the mug in front, turns
around and walks the short distance to the cabinet behind. The scene is
deliberately a bit cluttered (mug + a can + an apple on the counter) so the pick
is a real choice, not the only object.

Success (object-position-based, so the task is complete and real even before
grasping/PICO is wired up): the cabinet door is OPEN *and* the mug is INSIDE the
cabinet's interior. Scene: layout 6 (an empty room) -- the counter + cabinet are
built by the task -> light for fast rendering.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures.cabinets import SingleCabinet
from robocasa.models.fixtures.counter import Counter
import robocasa.utils.object_utils as OU


class StoreMugInCabinet(LivingRoomEnv):
    # counter used as the pick surface (kitchen Counter, clean box collision)
    COUNTER_SIZE = (0.45, 0.85, 0.72)   # x-depth (shallow, toward robot), y-width
    #                                     (wide, left-right), z-height (lowered for
    #                                     an easier reach)
    COUNTER_POS = (0.8, 0.0)            # planar (x, y); directly in FRONT of the
    #                                     robot (+x) -- the robot faces it
    # storage cabinet (kitchen SingleCabinet) placed on the floor by the task
    CABINET_SIZE = (0.45, 0.40, 0.90)   # width, depth, height (m) -- tall enough
    # that the top-positioned handle sits ~0.8m (a comfortable reach standing)
    CABINET_POS = (-0.9, 0.0)           # planar (x, y); directly BEHIND the robot
    #                                     (-x), door turned to face the robot, so
    #                                     the robot stands BETWEEN counter and
    #                                     cabinet (~1.7 m apart) facing the counter

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 6)
        kwargs.setdefault("style_ids", 1)
        # robot at the origin facing +x: it stands BETWEEN the counter (in front,
        # +x, which it faces) and the cabinet (behind, -x).
        kwargs.setdefault("robot_base_pos", (0.2, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()

        # Build a kitchen Counter as the pick surface (origin_table). Its
        # collision is clean procedural boxes matching the visual, and its top
        # geom is a native object-placement region. Rest it on the floor (body
        # origin is centered -> lift half its height).
        counter = Counter(
            name="origin_table",
            size=list(self.COUNTER_SIZE),
            top_texture="textures/marble/marble_2.png",
            base_texture="textures/wood/warm_wood_grain.png",
            rng=self.rng,
        )
        counter_z = -counter._regions["main"]["p0"][2]
        counter.set_pos((self.COUNTER_POS[0], self.COUNTER_POS[1], counter_z))
        counter.set_euler((0.0, 0.0, 0.0))
        self.furniture["origin_table"] = counter

        # Stand a kitchen SingleCabinet on the floor as the storage cabinet,
        # directly behind the robot. Its body origin is centered, so rest it by
        # lifting half its height (the main region's -z corner).
        cab = SingleCabinet(
            size=list(self.CABINET_SIZE), name="cabinet",
            orientation="right",
            # the cabinet XML's default housing texture (wood1.png) isn't in the
            # asset set, so point at a real wood texture.
            texture="textures/wood/warm_wood_grain.png",
            # default handle_vpos is "bottom" (handle too low for a standing
            # humanoid) -> put it at the top of the door.
            panel_config={"handle_vpos": "top"},
            rng=self.rng,
        )
        rest_z = -cab._regions["main"]["p0"][2]
        cab.set_pos((self.CABINET_POS[0], self.CABINET_POS[1], rest_z))
        # default front (door) faces -y; the cabinet sits behind the robot (-x),
        # so rotate +90 deg about z to turn the door to +x, facing the robot.
        cab.set_euler((0.0, 0.0, np.pi / 2))
        self.furniture["cabinet"] = cab
        self.cabinet = cab
        self.fixtures = self.furniture

    def _setup_furniture_references(self):
        self.origin_table = self.register_furniture_ref(
            "origin_table", dict(id="origin_table")
        )

    def _get_obj_cfgs(self):
        # the mug (target) plus a couple of distractors on the counter top.
        # Counter top is ~0.45 (x, depth) x 0.85 (y, width). Sample in a shallow
        # band (0.12 x 0.40) pushed hard toward the near (robot-facing) edge
        # (offset -0.15 in local x) so the objects sit right at the front edge,
        # easy for the robot to reach without leaning over the counter.
        place = dict(
            fixture=self.origin_table, size=(0.12, 0.40), offset=(-0.15, 0.0)
        )
        return [
            dict(name="mug", obj_groups="mug", placement=place),
            dict(name="distractor_can", obj_groups="can", placement=place),
            dict(name="distractor_apple", obj_groups="apple", placement=place),
        ]

    def _setup_scene(self):
        # The cabinet hinge needs noslip iterations to hold the door at the
        # opened angle (frictionloss > gravity torque). Robosuite's scene merge
        # drops the fixture's <option>, so set it here after the sim is built.
        super()._setup_scene()
        self.sim.model.opt.noslip_iterations = max(
            int(self.sim.model.opt.noslip_iterations), 10
        )

    def cabinet_door_open(self):
        return bool(self.cabinet.is_open(self))

    def mug_inside_cabinet(self):
        return bool(OU.obj_inside_of(self, "mug", self.cabinet, partial_check=True))

    def _check_success(self):
        # door opened AND the mug placed inside the cabinet interior
        return self.cabinet_door_open() and self.mug_inside_cabinet()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick up the mug from the table, carry it to the cabinet, open the "
            "cabinet door and put the mug inside"
        )
        return ep_meta
