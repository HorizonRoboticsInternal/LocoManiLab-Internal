"""The maintained fixed-room collection tasks.

The data-collection line originally settled on 48 tasks -- 33 atomic, 15
composite -- one fixed room per space, and verified most of them by actually teleoperating them
(`tasks_to_collect_0804_sorted.md` on `main`). That list, not this repo's older
one, is what the pipeline is now aimed at.

⚠️ SEPARATE REGISTRY ON PURPOSE. The older adapters were written against a task
set that is now set aside, and mixing the two would make it impossible to say
which set a number came from. `goals_for` here resolves ONLY the maintained
members of that collection set; the old
`adapters.goals_for` is untouched and still resolves its own.

⚠️ FIXED ROOMS. Each space is pinned to one floor plan -- kitchen 13, livingroom
209, bedroom 329, bathroom 401 -- because iThor resamples the room on every
reset, and that resampling is the largest noise source in every measurement this
project makes: the same ask measured 0.107 m in one room and 0.628 in the next,
purely because the two spawn the robot at different headings. `SCENE_ID` below
is what the driver should pin.

The collection notes travel with the table because they are measurements
somebody already paid for -- "盘子拿不起来" (the plate cannot be picked up) is
the same conclusion the hand geometry gives independently: the fingers span
0.140 m open and the plate is 0.31 m across.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rl.kimodo.heft.adapters import (
    CLOSE,
    DELIVERABLE_REACH,
    STANDING_REACH,
    HOLD,
    OPEN,
    RELEASE,
    SceneGoals,
    Waypoint,
    _approach,
    _articulation,
    _object_pos,
    _pick_place,
    _room_lift,
)

#: One room per space. The collection line's choice, adopted rather than
#: re-derived: their numbers and ours are only comparable in the same room.
SCENE_ID = {"kitchen": 13, "livingroom": 209, "bedroom": 329, "bathroom": 401}

#: env -> (room, atomic|composite). Straight from the collection list.
TASKS: dict[str, tuple[str, str]] = {
    # ---- kitchen 13 ----
    "IThorKitchenApproachFridge": ("kitchen", "atomic"),
    "IThorKitchenOpenFridge": ("kitchen", "atomic"),
    "IThorKitchenCloseFridge": ("kitchen", "atomic"),
    "IThorKitchenOpenCabinet": ("kitchen", "atomic"),
    "IThorKitchenCloseCabinet": ("kitchen", "atomic"),
    "IThorKitchenOpenDrawer": ("kitchen", "atomic"),
    "IThorKitchenCloseDrawer": ("kitchen", "atomic"),
    "IThorKitchenOpenOven": ("kitchen", "atomic"),
    "IThorKitchenCloseOven": ("kitchen", "atomic"),
    "IThorKitchenAppleOnPlate": ("kitchen", "atomic"),
    "IThorKitchenSpongeToSink": ("kitchen", "atomic"),
    "IThorKitchenApproachOpenFridge": ("kitchen", "composite"),
    "IThorKitchenApproachOpenOven": ("kitchen", "composite"),
    "IThorKitchenOpenDrawerTakeApple": ("kitchen", "composite"),
    "IThorKitchenDrawerPnP": ("kitchen", "composite"),
    "IThorKitchenAppleCabinetPnP": ("kitchen", "composite"),
    "IThorKitchenAppleBananaPlateSwap": ("kitchen", "composite"),
    "IThorKitchenPackProduceIntoBox": ("kitchen", "composite"),
    "IThorKitchenBananaPlateToTable": ("kitchen", "composite"),
    "IThorKitchenCoffeeDrinkToTrash": ("kitchen", "composite"),
    "IThorKitchenBowlStack": ("kitchen", "composite"),
    # ---- livingroom 209 ----
    "IThorLivingRoomApproachSofa": ("livingroom", "atomic"),
    "IThorLivingRoomApproachTable": ("livingroom", "atomic"),
    "IThorLivingRoomOpenDrawer": ("livingroom", "atomic"),
    "IThorLivingRoomCloseLaptop": ("livingroom", "atomic"),
    "IThorLivingRoomLiftStatue": ("livingroom", "atomic"),
    "IThorLivingRoomBasketToSofa": ("livingroom", "atomic"),
    "IThorLivingRoomCoffeeDrinkToBin": ("livingroom", "composite"),
    # ---- bedroom 329 ----
    "IThorBedroomApproachBed": ("bedroom", "atomic"),
    "IThorBedroomApproachDesk": ("bedroom", "atomic"),
    "IThorBedroomOpenTopDrawer": ("bedroom", "atomic"),
    "IThorBedroomCloseTopDrawer": ("bedroom", "atomic"),
    "IThorBedroomCloseLaptop": ("bedroom", "atomic"),
    "IThorBedroomLiftBasket": ("bedroom", "atomic"),
    "IThorBedroomCanIntoBasket": ("bedroom", "atomic"),
    "IThorBedroomApproachDeskAndCloseLaptop": ("bedroom", "composite"),
    "IThorBedroomBedThenDesk": ("bedroom", "composite"),
    # ---- bathroom 401 ----
    "IThorBathroomApproachGarbageCan": ("bathroom", "atomic"),
    "IThorBathroomApproachSink": ("bathroom", "atomic"),
    "IThorBathroomApproachToilet": ("bathroom", "atomic"),
    "IThorBathroomOpenToiletLid": ("bathroom", "atomic"),
    "IThorBathroomCloseToiletLid": ("bathroom", "atomic"),
    "IThorBathroomSoapToLowerShelf": ("bathroom", "atomic"),
    "IThorBathroomToiletPaperRollIntoBin": ("bathroom", "atomic"),
    "IThorBathroomPlungeToilet": ("bathroom", "atomic"),
    "IThorBathroomApproachToiletAndOpenToiletLid": ("bathroom", "composite"),
    "IThorBathroomApproachToiletAndCloseToiletLid": ("bathroom", "composite"),
}

#: What the collection line already found out by trying. Kept because a task
#: somebody failed to teleoperate is unlikely to be one Kimodo completes, and
#: because two of these were independently re-derived here at some cost.
KNOWN_TROUBLE = {
    "IThorKitchenBananaPlateToTable": "plate cannot be picked up (0.31 m across "
                                      "against a 0.140 m hand span); walk too far",
    "IThorKitchenAppleCabinetPnP": "too hard",
    "IThorKitchenBowlStack": "bowls hard to pick, not collected",
    "IThorKitchenCloseOven": "robot bounces off when bending to the oven door",
    "IThorKitchenSpongeToSink": "sponge falls; sink too far; oven opens itself",
    "IThorLivingRoomCoffeeDrinkToBin": "too far",
}

#: Tasks that do not BUILD, with what was actually seen. Separate from
#: `KNOWN_TROUBLE` (which is the collection line's judgement about difficulty)
#: because these are not opinions -- the env raises before a policy ever runs,
#: and an adapter for them can never be exercised.
#:
#: ⚠️ Each was checked on THREE seeds in its pinned room, because a placement
#: failure can be one unlucky draw: `IThorKitchenDrawerPnP` raises "no
#: collision-free semantic object placement" on seed 0 in FloorPlan13 and builds
#: fine on seeds 1 and 2, so it is NOT in this table. The three below fail on
#: every seed tried.
CANNOT_BUILD = {
    "IThorBathroomPlungeToilet":
        "FileNotFoundError: models/assets/objects/ithor/plunger/model.xml -- the "
        "asset is not in this repo and not on origin/main either",
    "IThorKitchenPackProduceIntoBox":
        "FileNotFoundError: models/assets/objects/usd/box/converted/model.xml",
    # ⚠️ Re-measured on 2026-08-12, because the task was RENAMED (it was
    # `IThorLivingRoomPillowToSofa`) and rewritten at the same time -- carrying
    # the old verdict over under a new name would have been an unverified claim.
    # It still does not build, but the failure is a different one now:
    "IThorLivingRoomBasketToSofa":
        "RuntimeError: FloorPlan209: no collision-free semantic object placement "
        "after 12 attempts -- seeds 0-3, all four identical (was 'Probabilities "
        "contain NaN' before the rewrite)",
}

_REG: dict[str, Callable] = {}


def register(*names):
    def deco(fn):
        for n in names:
            _REG[n] = fn
        return fn
    return deco


def scene_id_for(env_name: str) -> int | None:
    """Return the task's fixed collection floor plan, if it is maintained."""
    room = TASKS.get(env_name, (None, None))[0]
    return SCENE_ID.get(room) if room else None


def goals_for(env) -> SceneGoals:
    """Resolve only maintained collection tasks; never fall back silently."""
    name = type(getattr(env, "unwrapped", env)).__name__
    if name not in TASKS:
        raise SystemExit(
            f"[adapters48] {name!r} is not one of the maintained collection tasks. The "
            f"older set lives in `adapters.py` and is resolved separately; "
            f"mixing them would make a number impossible to attribute.")
    if name not in _REG:
        raise SystemExit(
            f"[adapters48] {name!r} is in the collection list but has no adapter "
            f"yet ({TASKS[name][1]}, {TASKS[name][0]}).")
    return _REG[name](env)


def registered() -> list[str]:
    return sorted(_REG)


def missing() -> list[str]:
    return sorted(t for t in TASKS if t not in _REG)


# --------------------------------------------------------------------------
# The families that already have working adapters, delegated rather than
# duplicated. ⚠️ Delegation, not a copy: these were tuned against measurements
# (the arc stepping, the reach-aware contact point, the hand choice) and a
# second copy would drift away from them silently.
# --------------------------------------------------------------------------
register(
    "IThorKitchenApproachFridge", "IThorLivingRoomApproachSofa",
    "IThorLivingRoomApproachTable", "IThorBedroomApproachBed",
    "IThorBedroomApproachDesk", "IThorBathroomApproachGarbageCan",
    "IThorBathroomApproachSink", "IThorBathroomApproachToilet",
)(_approach)

register(
    "IThorKitchenOpenFridge", "IThorKitchenCloseFridge",
    "IThorKitchenOpenCabinet", "IThorKitchenCloseCabinet",
    "IThorKitchenOpenDrawer", "IThorKitchenCloseDrawer",
    "IThorKitchenOpenOven", "IThorKitchenCloseOven",
    "IThorKitchenApproachOpenFridge", "IThorKitchenApproachOpenOven",
    "IThorLivingRoomOpenDrawer", "IThorLivingRoomCloseLaptop",
    "IThorBedroomOpenTopDrawer", "IThorBedroomCloseTopDrawer",
    "IThorBedroomCloseLaptop", "IThorBedroomApproachDeskAndCloseLaptop",
    "IThorBathroomOpenToiletLid", "IThorBathroomCloseToiletLid",
    "IThorBathroomApproachToiletAndOpenToiletLid",
    "IThorBathroomApproachToiletAndCloseToiletLid",
)(_articulation)

register("IThorLivingRoomLiftStatue", "IThorBedroomLiftBasket")(_room_lift)

register("IThorKitchenAppleOnPlate", "IThorKitchenBananaPlateToTable")(_pick_place)


@register("IThorBedroomBedThenDesk")
def _bedroom_bed_then_desk(env) -> SceneGoals:
    """Visit the task's exact bed and desk fixtures in their declared order."""
    waypoints = []
    for fixture, label in (
        (env.first_fixture, "approach bed"),
        (env.second_fixture, "approach desk"),
    ):
        centre = np.asarray(fixture.world_center(env.sim), dtype=np.float64)
        waypoints.append(
            Waypoint(
                np.array([centre[0], centre[1], 0.9]),
                phase=OPEN,
                label=label,
                approach_only=True,
                arrive_radius=1.0,
                arrive_test=lambda fixture=fixture: bool(env._near_fixture(fixture)),
            )
        )
    return SceneGoals(
        waypoints=waypoints,
        lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")),
    )


# --------------------------------------------------------------------------
# Pick and place, the largest missing family. Eleven tasks that all mean "take
# this and put it there" and expose it eleven slightly different ways, so the
# destination is RESOLVED at runtime rather than assumed from an attribute name.
# --------------------------------------------------------------------------

#: Region names that mean "this is where the object has to end up". Matched as
#: substrings, lower-cased, in this order -- `region_refs` uses task-specific
#: names (`trash_receptacle`, `target_shelf`, `apple_bowl_surface`, `trash_bin`)
#: and there is no shared vocabulary to look up.
DEST_HINTS = ("target", "trash", "bin", "receptacle", "basin", "bathtub",
              "bowl", "sofa", "tub", "dest")

#: How far above the destination the hand is asked to hold the object, metres.
#: Clear of the surface without asking for a lift the arm has to fight.
PLACE_CLEARANCE = 0.10


#: Which object to pick, for the tasks that place more than one and do not say.
#: ⚠️ The only hand annotation in this module, and it is here because the
#: alternative is guessing: a receptacle task places both a source and target,
#: and picking the bowl would be a different task that also scores zero.
SOURCE_BY_TASK = {
    "IThorBedroomCanIntoBasket": "can",
    "IThorKitchenPackProduceIntoBox": "apple",
}


def _source_object(env) -> str:
    by_task = SOURCE_BY_TASK.get(type(getattr(env, "unwrapped", env)).__name__)
    objs = list(getattr(env, "objects", {}) or {})
    if by_task and by_task in objs:
        return by_task
    name = getattr(env, "SOURCE_NAME", None)
    if name and name in objs:
        return str(name)
    if len(objs) == 1:
        return objs[0]
    raise SystemExit(
        f"[adapters48] {type(env).__name__}: cannot tell which of {objs} is the "
        f"object to pick up. Set SOURCE_NAME on the task or add it to the table "
        f"here -- guessing would drive at the wrong thing and still report a "
        f"number.")


def _destination(env, source: str) -> np.ndarray:
    """Where the object has to end up, world. Raises rather than guessing."""
    regions = getattr(env, "region_refs", {}) or {}
    for key in DEST_HINTS:
        for rname, region in regions.items():
            if key in str(rname).lower() and "source" not in str(rname).lower():
                centre, _ = region.world_pose(env.sim)
                return np.asarray(centre, dtype=np.float64)
    # A second OBJECT is a destination too: "apple in bowl" names the bowl.
    others = [o for o in (getattr(env, "objects", {}) or {}) if o != source]
    if len(others) == 1:
        return _object_pos(env, others[0])
    for attr in ("BASIN_CENTER_XY", "BIN_TARGET_XY"):
        xy = getattr(env, attr, None)
        if xy is not None:
            here = _object_pos(env, source)
            return np.array([float(xy[0]), float(xy[1]), float(here[2])])
    raise SystemExit(
        f"[adapters48] {type(env).__name__}: no destination found. Regions were "
        f"{sorted(regions)}, objects {sorted(getattr(env, 'objects', {}) or {})}. "
        f"Add a hint to DEST_HINTS or a case here.")


@register(
    "IThorLivingRoomBasketToSofa", "IThorBedroomCanIntoBasket",
    "IThorBathroomSoapToLowerShelf",
    "IThorKitchenCoffeeDrinkToTrash", "IThorLivingRoomCoffeeDrinkToBin",
    "IThorKitchenSpongeToSink",
    "IThorKitchenPackProduceIntoBox",
)
def _pick_place48(env) -> SceneGoals:
    """Grasp the source object, carry it over the destination, let go.

    Three waypoints rather than two: the release is its own, because the grasp
    state machine only opens on a RELEASE phase and a place that never releases
    scores as a carry that happens to be in the right spot.

    ⚠️ The destination is asked for 0.10 m ABOVE its centre. A region's world
    pose is the surface, and asking the hand to arrive exactly on it drives the
    held object through the table it is being placed on.

    ⚠️ Every one of these is a real grasp, and the hand spans 0.09-0.14 m: the
    collection line could not pick the plate up either. Expect zeros from the
    ones whose object is outside that band; the adapter's job is to make the
    attempt legible, not to promise it succeeds.
    """
    src = _source_object(env)
    here = _object_pos(env, src)
    dest = _destination(env, src)
    over = dest + np.array([0.0, 0.0, PLACE_CLEARANCE])

    # ⚠️ Whether the carry needs FEET is a property of the distance, not of the
    # word "carry". These tasks range from same-workspace placement to a can
    # carried across the bedroom: marking them all
    # stationary asks the robot to place something it cannot reach, and marking
    # them all walking spends h27's 0.21 m root error on a 6 cm move.
    root = np.asarray(env.sim.data.body_xpos[
        __import__("mujoco").mj_name2id(
            env.sim.model._model,
            __import__("mujoco").mjtObj.mjOBJ_BODY, "mobilebase0_support")],
        dtype=np.float64)
    walk_to_place = bool(np.linalg.norm(over[:2] - root[:2])
                         > DELIVERABLE_REACH)
    return SceneGoals(waypoints=[
        Waypoint(here, phase=CLOSE, label=f"grasp {src}", grasp_object=src),
        Waypoint(over, phase=HOLD, label=f"carry {src}",
                 needs_root=walk_to_place, grasp_object=src),
        Waypoint(over, phase=RELEASE, label=f"release {src}", needs_root=False,
                 grasp_object=src),
    ], lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")),
        notes={"source": src,
               "dest": [round(float(v), 3) for v in dest],
               "carry_m": round(float(np.linalg.norm(dest[:2] - here[:2])), 3),
               "walk_to_place": walk_to_place})


@register("IThorBathroomToiletPaperRollIntoBin")
def _native_toilet_roll_to_bin(env) -> SceneGoals:
    """Drive the hand through the native fixture's source and bin positions.

    Native iTHOR fixtures are not entries in ``env.objects``, so the scripted
    object-grasp tracker cannot bind them. The task itself remains the judge of
    contact, carry, release, and final containment.
    """

    here = np.asarray(env.source_fixture.world_center(env.sim), dtype=np.float64)
    centre, _ = env.bin_region.world_pose(env.sim)
    over = np.asarray(centre, dtype=np.float64) + np.array(
        [0.0, 0.0, PLACE_CLEARANCE]
    )
    root = np.asarray(env._robot_base_xy(), dtype=np.float64)
    walk_to_place = bool(np.linalg.norm(over[:2] - root[:2]) > DELIVERABLE_REACH)
    return SceneGoals(
        waypoints=[
            Waypoint(here, phase=CLOSE, label="grasp toilet paper roll"),
            Waypoint(
                over,
                phase=HOLD,
                label="carry toilet paper roll",
                needs_root=walk_to_place,
            ),
            Waypoint(
                over,
                phase=RELEASE,
                label="release toilet paper roll into bin",
                needs_root=False,
            ),
        ],
        lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")),
        notes={"native_fixture": True, "walk_to_place": walk_to_place},
    )


# --------------------------------------------------------------------------
# The composites. Six tasks that are two or three of the families above in
# sequence, and the only ones left.
#
# ⚠️ READ THIS BEFORE READING A NUMBER OFF THEM. Every one needs at least one
# real grasp, and the measured grasp rate on this pipeline is 0 -- the hand
# spans 0.09-0.14 m, arrives with the palm unoriented, and h27 puts the wrist
# 0.317 m off its goal on rear bearings. These adapters exist so the ATTEMPT is
# legible and so a later fix has somewhere to land, not because the tasks are
# expected to pass today. A composite that fails at waypoint 1 of 7 tells you
# nothing about waypoints 2-7.
# --------------------------------------------------------------------------

#: How far above a surface the hand is asked to release a carried object.
STACK_CLEARANCE = 0.12


def _carry(src: str, here: np.ndarray, over: np.ndarray, root: np.ndarray,
           label: str) -> list[Waypoint]:
    """grasp -> carry -> release, the unit every composite is built from."""
    return [
        Waypoint(here, phase=CLOSE, label=f"grasp {label}", grasp_object=src),
        Waypoint(over, phase=HOLD, label=f"carry {label}", grasp_object=src,
                 needs_root=bool(np.linalg.norm(over[:2] - root[:2])
                                 > DELIVERABLE_REACH)),
        Waypoint(over, phase=RELEASE, label=f"release {label}", needs_root=False,
                 grasp_object=src),
    ]


def _root_xy(env) -> np.ndarray:
    import mujoco
    return np.asarray(env.sim.data.body_xpos[mujoco.mj_name2id(
        env.sim.model._model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_support")],
        dtype=np.float64)


@register("IThorBathroomPlungeToilet")
def _plunge_toilet(env) -> SceneGoals:
    """Pick the plunger up and put its HEAD in the bowl.

    ⚠️ The hand goal is not the bowl. Success is `_plunger_head_in_bowl`, the
    head is the plunger's LOWEST point, and the hand holds it somewhere above
    that -- so the hand has to end up as far above the bowl as the grasp is
    above the head. That offset is read off the plunger's live pose rather than
    assumed, because it depends on where along the handle the grasp lands.
    """
    src = "plunger"
    here = _object_pos(env, src)
    head = np.asarray(env._plunger_head_point(), dtype=np.float64)
    bowl = np.asarray(env._toilet_bowl_center(), dtype=np.float64)
    _, rim_z = env.BOWL_Z_RANGE
    # Where the HEAD has to go, then lifted by however far the grasp sits above it.
    head_goal_z = float(rim_z) - float(env.BOWL_INSERT_DEPTH) - 0.02
    grip_above_head = float(here[2] - head[2])
    over = np.array([bowl[0], bowl[1], head_goal_z + grip_above_head])
    return SceneGoals(
        waypoints=_carry(src, here, over, _root_xy(env), "plunger"),
        lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")),
        notes={"source": src, "bowl": [round(float(v), 3) for v in bowl],
               "grip_above_head": round(grip_above_head, 3),
               "carry_m": round(float(np.linalg.norm(over[:2] - here[:2])), 3)})


@register("IThorKitchenAppleBananaPlateSwap")
def _plate_swap(env) -> SceneGoals:
    """Two carries: the apple off the plate onto the table, the banana onto it.

    Order matters and is not interchangeable -- the banana's destination is the
    plate the apple is currently sitting on, so putting the banana down first
    stacks it on the apple and `_apple_on_table` never fires.
    """
    root = _root_xy(env)
    apple = _object_pos(env, "apple")
    banana = _object_pos(env, "banana")
    plate = _object_pos(env, "plate")

    # Somewhere on the table that is NOT the plate: step away from it along the
    # line from the plate to the robot, so the drop stays within reach.
    away = plate[:2] - root[:2]
    n = float(np.linalg.norm(away))
    away = away / n if n > 1e-6 else np.array([1.0, 0.0])
    side_xy = plate[:2] - away * 0.25
    apple_to = np.array([side_xy[0], side_xy[1], plate[2] + STACK_CLEARANCE])
    banana_to = plate + np.array([0.0, 0.0, STACK_CLEARANCE])

    return SceneGoals(
        waypoints=(_carry("apple", apple, apple_to, root, "apple off the plate")
                   + _carry("banana", banana, banana_to, root, "banana onto the plate")),
        lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")),
        notes={"order": "apple off first, then banana on",
               "plate": [round(float(v), 3) for v in plate]})


@register("IThorKitchenBowlStack")
def _bowl_stack(env) -> SceneGoals:
    """Stack three bowls: 2 onto 1, then 3 onto 2.

    ⚠️ The bottom bowl is chosen as the one nearest the robot rather than by
    name. `BOWL_NAMES` is placement order, not a stacking order, and carrying
    the far bowl first means the two later carries both start out of reach.
    """
    root = _root_xy(env)
    names = list(env.BOWL_NAMES)
    pos = {n: _object_pos(env, n) for n in names}
    names.sort(key=lambda n: float(np.linalg.norm(pos[n][:2] - root[:2])))
    base = names[0]

    wps, top_z = [], float(pos[base][2])
    for n in names[1:]:
        top_z += STACK_CLEARANCE
        over = np.array([pos[base][0], pos[base][1], top_z])
        wps += _carry(n, pos[n], over, root, f"{n} onto {base}")
    return SceneGoals(
        waypoints=wps,
        lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")),
        notes={"base": base, "order": names[1:],
               "reach_m": [round(float(np.linalg.norm(pos[n][:2] - root[:2])), 2)
                           for n in names]})


def _region_centre(env, *names) -> np.ndarray:
    """The world centre of the first region that matches, world frame.

    ⚠️ Looks at ATTRIBUTES before `region_refs`. These tasks score against
    `self.cabinet_receptacle_region` / `self.drawer_receptacle_region` /
    `self.table_region` directly, and a task is free to hold a region it never
    publishes -- resolving through `region_refs` alone would silently aim at a
    different region than the one `_check_success` reads.
    """
    for n in names:
        region = getattr(env, n if n.endswith("_region") else f"{n}_region", None)
        if region is not None:
            centre, _ = region.world_pose(env.sim)
            return np.asarray(centre, dtype=np.float64)
    regions = getattr(env, "region_refs", {}) or {}
    for n in names:
        for rname, region in regions.items():
            if str(n).lower().replace("_region", "") in str(rname).lower():
                centre, _ = region.world_pose(env.sim)
                return np.asarray(centre, dtype=np.float64)
    raise SystemExit(
        f"[adapters48] {type(env).__name__}: none of {names} resolved, either as "
        f"an attribute or in regions {sorted(regions)} -- the destination cannot "
        f"be guessed.")


def _as_if_open(env):
    """Plan against the fixture where the OPEN leg will leave it, not where it is.

    ⚠️ Without this the close leg cannot be planned at all, and the symptom is
    not subtle: `_articulation(behavior="close")` picks its contact point by
    moving the joint from where it IS to `closed_qpos` and taking the geom that
    travels furthest. On a drawer that is still shut those are the same value,
    every geom travels 0, and it raises "nothing on this body moves".

    So the joint is driven to `open_qpos` for the duration of the planning call
    and put back afterwards. `mj_forward` runs at both ends: geom positions are
    derived state, and a plan read off stale ones aims at where the drawer was.
    """
    import contextlib

    import mujoco

    @contextlib.contextmanager
    def ctx():
        j = env.target_semantic_joint
        d, m = env.sim.data, env.sim.model
        was = float(d.get_joint_qpos(j.name))
        try:
            d.set_joint_qpos(j.name, float(j.open_qpos))
            mujoco.mj_forward(m._model, d._data)
            yield
        finally:
            d.set_joint_qpos(j.name, was)
            mujoco.mj_forward(m._model, d._data)
    return ctx()


def _put_away(env, dest_names, *, open_first: bool, close_after: bool) -> SceneGoals:
    """(optionally open the fixture) -> carry the apple into it -> shut it.

    ⚠️ `open_first` is NOT derivable from the task's `BEHAVIOR`, which is "open"
    for both of these. `IThorKitchenAppleCabinetPnP` calls
    `_open_cabinet_initially()` in its own setup, so the cabinet is ALREADY open
    when the episode starts and an open leg has nothing to do -- `_articulation`
    refuses it, correctly, because moving the joint to `open_qpos` from
    `open_qpos` moves it zero metres. `IThorKitchenDrawerPnP` starts shut and
    latches `_drawer_opened` only once the fraction passes 0.50, so it needs
    both legs -- and its close leg has to be planned through `_as_if_open`.
    """
    wps, notes = [], {}
    if open_first:
        opening = _articulation(env, behavior="open")
        wps += list(opening.waypoints)
        notes.update(opening.notes or {})
    apple = _object_pos(env, "apple")
    into = _region_centre(env, *dest_names) + np.array([0.0, 0.0, 0.05])
    wps += _carry("apple", apple, into, _root_xy(env), "apple into it")
    if close_after:
        if open_first:
            with _as_if_open(env):
                shut = _articulation(env, behavior="close")
            notes["close_leg"] = "planned with the joint held at open_qpos"
        else:
            shut = _articulation(env, behavior="close")
            notes["close_leg"] = "the only articulation leg; it starts open"
        wps += list(shut.waypoints)
    notes["dest"] = [round(float(v), 3) for v in into]
    notes["open_first"] = open_first
    return SceneGoals(waypoints=wps, notes=notes,
                      lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")))


@register("IThorKitchenDrawerPnP")
def _drawer_pnp(env) -> SceneGoals:
    """Open the drawer, put the apple in, shut it. Starts shut; success wants <= 0.08."""
    return _put_away(env, ("drawer_receptacle", "drawer"),
                     open_first=True, close_after=True)


@register("IThorKitchenAppleCabinetPnP")
def _apple_cabinet_pnp(env) -> SceneGoals:
    """Put the apple in the already-open cabinet, then shut it.

    No open leg: the task opens the cabinet itself in `_setup_scene`.
    """
    return _put_away(env, ("cabinet_receptacle", "cabinet"),
                     open_first=False, close_after=True)


@register("IThorKitchenOpenDrawerTakeApple")
def _open_drawer_take_apple(env) -> SceneGoals:
    """Open the drawer, take the apple OUT, put it on the table.

    The mirror of `_drawer_pnp`: the drawer stays open (success wants fraction
    >= 0.50, so closing it after would UNDO the task) and the carry runs the
    other way, out of the receptacle and onto the table.
    """
    opening = _articulation(env, behavior="open")
    apple = _object_pos(env, "apple")
    root = _root_xy(env)
    table = _region_centre(env, "table", "surface", "counter")
    onto = np.array([table[0], table[1], float(table[2]) + STACK_CLEARANCE])
    notes = dict(opening.notes or {})
    notes.update({"dest": [round(float(v), 3) for v in onto],
                  "leaves_drawer_open": True})
    return SceneGoals(
        waypoints=list(opening.waypoints) + _carry("apple", apple, onto, root,
                                                   "apple out of the drawer"),
        notes=notes,
        lang=str(getattr(env, "get_ep_meta", lambda: {})().get("lang", "")))
