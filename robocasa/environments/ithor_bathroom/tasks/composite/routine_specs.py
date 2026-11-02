"""Task-local layouts for five routines; never mutate collected calibrations."""

from copy import deepcopy

from robocasa.models.scenes.ithor_bathroom_task_specs import get_ithor_bathroom_task_spec


def _atomic(scene, key):
    return get_ithor_bathroom_task_spec(f"FloorPlan{scene}", key)


def tidy_specs(*, shelf=False):
    items = dict(bottle=_atomic(401, "stand_soap_bottle_upright"),
                 sponge=_atomic(401, "put_sponge_in_bathtub"))
    if shelf:
        items["soap"] = _atomic(401, "soap_to_lower_shelf")
    navigation = None if shelf else _atomic(401, "navigate_to_bathing_area")
    # The atomic shelf stand pose intersects both index fingers in SONIC's
    # spawn stance. Back up only this new routine by 18 cm; leave the collected
    # atomic layout and every source / target placement untouched.
    start_pose = (dict(xy=[-2.57432, 3.38], yaw=1.5707963267948966) if shelf else
                  deepcopy(navigation["start_pose"]))
    return {"FloorPlan401": dict(items=items, navigation=navigation,
        start_pose=start_pose,
        lang=("put the bar of soap on the lowest shelf, stand the soap bottle upright, "
              "then put the sponge in the bathtub" if shelf else
              "walk to the bathtub, stand the soap bottle upright, then put the sponge in the bathtub"))}


_TOILET_ROUTINE_SCENES = (401, 405, 407, 415, 418)

# Each position is a collision-free Floor location calibrated beside the task's
# destination workspace. Keep 401 unchanged: it is the collected reference.
_SOAP_STOOL_LAYOUTS = {
    401: ([-2.70, 2.20], dict(xy=[-2.25, 2.15], yaw=3.030935)),
    405: ([-2.55, -0.25], dict(xy=[-2.05, -0.25], yaw=3.141593)),
    407: ([-0.90, -0.90], dict(xy=[-0.35, -0.90], yaw=3.141593)),
    415: ([-0.75, -2.55], dict(xy=[-1.35, -2.55], yaw=0.)),
    418: ([-1.80, -3.65], dict(xy=[-1.30, -3.65], yaw=3.141593)),
}
_BRUSH_STOOL_LAYOUTS = {
    401: ([-0.90, 2.80], dict(xy=[-1.40, 2.80], yaw=0.)),
    405: ([-2.50, 0.85], dict(xy=[-1.90, 0.85], yaw=0.)),
    407: ([-1.15, 0.20], dict(xy=[-0.55, 0.20], yaw=0.)),
    415: ([-1.65, -0.85], dict(xy=[-1.05, -0.85], yaw=0.)),
    418: ([-0.90, -3.55], dict(xy=[-1.50, -3.55], yaw=0.)),
}

# The collected close-lid poses place the SONIC collector stance against nearby
# geometry in these two scenes. Keep that calibration unchanged for the atomic
# task; only this routine starts from the adjacent collision-free floor pose.
_SOAP_CLOSE_ROUTINE_START_POSES = {
    405: dict(xy=[-2.393, 1.645], yaw=-2.487094),
    418: dict(xy=[-0.449, -2.495], yaw=-1.832596),
}


def _stool_item(scene, key, xy, pose, *, brush=False):
    spec = _atomic(scene, key)
    spec.update(
        source_xy=xy,
        robot_pose=pose,
        source_quat=([1., 0., 0., 0.] if brush else
                     [0.70710678, 0.70710678, 0., 0.]),
        surface_fixture=f"FloorPlan{scene}:routine_stool",
        surface_region=f"FloorPlan{scene}:routine_stool:region:support",
        # The routine relocates the source from its atomic support onto the
        # stool, so an inherited fixture-staging request is no longer valid.
        auxiliary_placements=[],
    )
    spec["source_stool"] = dict(
        model="/objects/lightwheel/stool/Stool005/model.xml",
        scale=[0.90, 1.90 if brush else 0.90, 0.60],
        xy=xy,
        yaw=0.,
        top_half_xy=[0.13, 0.28 if brush else 0.13],
        floor_probe_above=0.08,
        floor_probe_below=0.01,
        floor_probe_step=0.0005,
    )
    spec.pop("source_horizontal_distance", None)
    return spec


def toilet_specs(*, close):
    item_key = "lift_soap_bottle" if close else "lift_scrub_brush"
    layouts = _SOAP_STOOL_LAYOUTS if close else _BRUSH_STOOL_LAYOUTS
    specs = {}
    for scene in _TOILET_ROUTINE_SCENES:
        joint = _atomic(scene, "toilet_lid")
        joint.update(
            init_open_fraction=0.80 if close else 0.20,
            success_open_fraction=0.80,
            success_close_fraction=0.20,
        )
        nav = _atomic(scene, "approach_sink" if close else "approach_toilet")
        xy, pose = layouts[scene]
        item = _stool_item(scene, item_key, xy, pose, brush=not close)
        specs[f"FloorPlan{scene}"] = dict(
            items={"source": item},
            articulation=joint,
            navigation=nav,
            start_pose=deepcopy(
                _SOAP_CLOSE_ROUTINE_START_POSES.get(
                    scene, joint["close_robot_pose"]
                ) if close else nav["start_pose"]
            ),
            lang=(
                "close the toilet lid, walk to the sink, then pick up the soap bottle"
                if close else
                "walk to the toilet, open its lid, then pick up the scrub brush"
            ),
        )
    return specs


def drawer_specs():
    # Second drawer from the top, one level above the collected lower drawer.
    # All overrides are confined to this routine's defensive spec copies.
    joint = _atomic(415, "open_lower_drawer")
    pose = dict(xy=[-0.92, -3.5708], yaw=0.392699)
    joint.update(joint="chestofdrawers_665b5f9ef6118c922c09c75638260fde_1_4_0_DrawerMesh021_2",
                 robot_pose=deepcopy(pose), target="second drawer from the top",
                 lang="open the second bathroom drawer from the top")
    item = _atomic(415, "lift_toilet_paper_roll")
    support = joint["fixture"]
    item.update(surface_fixture=support, surface_region=support + ":routine_support",
                interior_region=support + ":region:receptacle:6",
                source_xy=[-0.245, -3.45], robot_pose=deepcopy(pose),
                source_quat=[0., 0., 0.70710678, 0.70710678],
                support_probe_above=0.02, support_probe_below=0.03,
                exit_direction=-1, exit_x=None)
    # 80% completes the opening stage; opening further gives retrieval clearance.
    item["operating_open_fraction"] = 1.0
    return {"FloorPlan415": dict(items={"source": item}, articulation=joint,
        navigation=None, start_pose=deepcopy(pose),
        lang="open the second drawer from the top, take out the toilet-paper roll, then close that drawer")}
