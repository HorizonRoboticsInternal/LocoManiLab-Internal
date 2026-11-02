"""TabletopPickPlace: the simplest living-room manipulation task -- a stationary,
upper-body-only pick-and-place.

The robot spawns facing a single counter directly in front of it and never
walks: a mug sits on one side of the counter top and a plate on the other, both
within arm's reach. The task is to pick up the mug and place it on the plate.
Because there is no locomotion, no turning and no squatting, the horizon is
short and the visual scene is fixed -- the easiest target for VLA data
collection, training and on-policy deployment validation.

Unlike the object-position-only living-room tasks (CarryObjectToTable,
SquatPickPlaceTVStand), success here is a real *manipulation* condition --
``check_obj_in_receptacle`` (the mug is in contact with the plate and centered
over it) -- so it requires actual grasping. Collect demos with PICO VR
upper-body teleop (``--g1-control pico-manager --pico-source real``); keyboard
locomotion control cannot complete it.

The pick/place surface is a kitchen ``Counter`` (the same clean axis-aligned box
collision StoreMugInCabinet uses, matching the visual exactly) built by the task
in ``_setup_model``; the scene itself (layout 7) is an empty room, so rendering
is light.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures.counter import Counter
import robocasa.utils.object_utils as OU


class TabletopPickPlace(LivingRoomEnv):
    # counter used as the single pick/place surface (kitchen Counter, clean box
    # collision). x-depth (shallow, toward the robot), y-width (wide, left-right),
    # z-height (a comfortable standing reach for the floating-body G1).
    COUNTER_SIZE = (0.45, 0.90, 0.78)
    # planar (x, y); directly in FRONT of the robot (+x), which it faces.
    COUNTER_POS = (0.8, 0.0)
    # left/right separation (counter-local +y / -y) of the mug and the plate.
    # Small so they sit close together -- a short pick->place motion. They still
    # don't overlap: the plate's wide sampling box is almost entirely eaten by
    # the boundary buffer (see below), so the plate barely moves from +SIDE_OFFSET,
    # and the mug's tight box keeps it near -SIDE_OFFSET.
    SIDE_OFFSET = 0.15
    # near-edge bias (counter-local x; -x is the robot-facing front edge, at
    # -COUNTER_SIZE[0]/2 = -0.225). The plate stays only slightly biased (its
    # ~0.25 m footprint is wide, so its box must exceed it and it can't be pushed
    # hard to the edge). The mug is pushed much closer to the front edge so it
    # sits near the lip, easy to reach -- with its tight x box (below) the mug
    # edge still stays a safe margin inside -0.225, so it never overhangs/falls.
    EDGE_OFFSET = -0.05
    MUG_EDGE_OFFSET = -0.14
    # plate near-edge bias: pushed as far toward the front edge as its wide
    # footprint allows. With the tightened plate x box below the plate center
    # barely wanders, so this sits the plate close to the robot without its rim
    # overhanging the -0.225 lip.
    PLATE_EDGE_OFFSET = -0.085
    # shrink the mug below its native size for an easier one-hand grasp.
    MUG_SCALE = 0.7
    # per-object sampling-box size (counter-local x-depth, y-width). x/y must each
    # exceed the object's footprint or the boundary buffer inverts the range. The
    # plate's box must stay > its ~0.25 m footprint, but the buffer then leaves it
    # only ~+/-0.03 m of wander (so it stays put). The mug gets a tight box so it
    # stays close to the front edge (small x) and on its own side (small y).
    PICK_SAMPLE_SIZE = (0.10, 0.12)
    # plate x tightened toward its footprint (~0.25 m) so the boundary buffer
    # leaves it almost no x wander -> it stays put at PLATE_EDGE_OFFSET near the
    # edge (box must stay > footprint or the range inverts).
    PLATE_SAMPLE_SIZE = (0.27, 0.30)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 7)
        kwargs.setdefault("style_ids", 1)
        # robot at the origin facing +x, standing just in front of the counter.
        kwargs.setdefault("robot_base_pos", (0.25, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()

        # Build a kitchen Counter as the pick/place surface. Its collision is
        # clean procedural boxes matching the visual, and its top geom is a native
        # object-placement region. Rest it on the floor (body origin is centered
        # -> lift by the -z corner of its main region).
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
        self.fixtures = self.furniture

    def _setup_furniture_references(self):
        self.table = self.register_furniture_ref("table", dict(id="table"))

    def _get_obj_cfgs(self):
        # mug on one half of the counter top, plate on the other half. Small
        # sample boxes + opposite-side offsets keep them apart and reachable.
        return [
            dict(
                name="mug",
                obj_groups="mug",
                object_scale=self.MUG_SCALE,
                placement=dict(
                    fixture=self.table,
                    size=self.PICK_SAMPLE_SIZE,
                    offset=(self.MUG_EDGE_OFFSET, -self.SIDE_OFFSET),
                ),
            ),
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.table,
                    size=self.PLATE_SAMPLE_SIZE,
                    offset=(self.PLATE_EDGE_OFFSET, self.SIDE_OFFSET),
                ),
            ),
        ]

    def _check_success(self):
        # the mug has been placed on the plate (in contact + centered over it)
        return bool(OU.check_obj_in_receptacle(self, "mug", "plate"))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the mug and place it on the plate"
        return ep_meta
