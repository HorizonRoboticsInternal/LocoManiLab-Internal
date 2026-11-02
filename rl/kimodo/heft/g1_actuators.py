"""HEFT's G1 actuator gains for the retained frozen tracker.

**This file exists to stop one specific mistake.** g1_bridge already ships a
per-joint kp/kd table at
``g1_bridge/vendored/wbt/utils/bydmimic_utils.py``
(``stiffness_dict`` / ``damping_dict``), it is already wired into
``WholeBodyG1Composite.run_controller``, and reusing it is the obvious move.
Those are **SONIC's** gains. Two groups disagree with HEFT's:

    joint group          bydmimic (SONIC)      G1_ARTICULATION_5 (HEFT)
    hip_pitch            kp 40.179 kd 2.558    kp 99.098 kd 6.309    2.47x
    wrist_pitch/yaw      kp 16.778 kd 1.068    kp  8.611 kd 0.548    1.95x
    (the other 10 groups agree)

hip_pitch is the joint that carries the most load during a reach or a step. Wrong
gains there do not topple the robot -- they make it track slightly worse and read
as "the tracker is a bit soft", which is the exact signature of all five silent
bugs heft_work documented.

The numbers below were transcribed from the original HEFT
``G1_ARTICULATION_5`` definitions in ``active_adaptation/assets/g1.py``. The
runtime uses this retained table and the bundled kinematic model; it does not
require the original training checkout.

Note ``G1_ARTICULATION_5`` is not ``G1_ARTICULATION``: the ``_5`` variants
override armature and effort_limit for hip_pitch and wrist_pitch/yaw while
inheriting stiffness and damping from their base group. Reading only the base
definitions gives hip_pitch armature 0.025102 instead of 0.010178.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.kimodo.heft.tracker_const import (
    ACTION_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    DATASET_JOINT_NAMES,
)


#: HEFT's own G1 model, and the reference `resolve_root_body` resolves the
#: anatomical pelvis against **on every reset** -- so this is a RUNTIME
#: dependency of the rollout, not a verification-only one.
#:
#: ⚠️ Vendored MESH-FREE. The original HEFT `g1_5.xml` requires 23 MB of STL;
#: this repository bundles only the kinematics needed by the tracker.
#: What `model_patch` reads from the model is the body tree, the joints and the
#: inertials -- never a geom -- so the 77 geoms and the whole `<asset>` block are
#: stripped. Verified against the original at the time of vendoring: FK over all
#: 36 bodies agrees to **0.0 m**, per-body mass to **0.0 kg** (33.3411 total),
#: and the 30 joint names are identical in order.
#:
#: ⚠️ `assets/`, NOT `_assets/`. `rl/kimodo/.gitignore` ignores `_assets/` at any
#: depth (local generated artifacts), so a file put there would be silently left
#: out of the commit and a fresh clone would die at the first reset with a
#: missing model -- which is precisely what happened on the first attempt.
#:
#: Point `HEFT_G1_5_XML` at the real file to check against the full model.
DEFAULT_G1_XML_5 = Path(
    os.environ.get("HEFT_G1_5_XML")
    or Path(__file__).resolve().parent / "assets" / "g1_5_kinematics.xml"
)


# The articulation the tracker was trained against. G1_ARTICULATION (no suffix)
# is a different robot variant and must not be used here.
ARTICULATION = "G1_ARTICULATION_5"


@dataclass(frozen=True)
class Gain:
    kp: float          # stiffness  [Nm/rad]
    kd: float          # damping    [Nms/rad]
    armature: float    # reflected rotor inertia [kg m^2]
    effort: float      # effort_limit [Nm]


# Transcribed from G1_ARTICULATION_5. Keys are regex patterns in the config's own
# declaration order; expand_gains() turns them into a per-joint table.
HEFT_ACTUATOR_GROUPS: tuple[tuple[str, tuple[str, ...], Gain], ...] = (
    ("G1_ACTUATOR_UPPER", (
        r".*_elbow_joint", r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint",
        r".*_shoulder_yaw_joint", r".*_wrist_roll_joint",
     ), Gain(kp=14.25062309787429, kd=0.907222843292423,
             armature=0.003609725, effort=25.0)),
    ("G1_ACTUATOR_HIP_PITCH_5", (
        r".*_hip_pitch_joint",
     ), Gain(kp=99.09842777666113, kd=6.3088018534966395,
             armature=0.010177520, effort=88.0)),
    ("G1_ACTUATOR_HIP_ROLL_KNEE_5", (
        r".*_hip_roll_joint", r".*_knee_joint",
     ), Gain(kp=99.09842777666113, kd=6.3088018534966395,
             armature=0.025101925, effort=139.0)),
    ("G1_ACTUATOR_HIP_YAW_WAIST_YAW", (
        r".*_hip_yaw_joint", r"waist_yaw_joint",
     ), Gain(kp=40.17923847137318, kd=2.5578897650279457,
             armature=0.010177520, effort=88.0)),
    ("G1_ACTUATOR_WRIST_PITCH_YAW_5", (
        r".*_wrist_pitch_joint", r".*_wrist_yaw_joint",
     ), Gain(kp=8.611032447370201, kd=0.548195351665136,
             armature=0.00425, effort=5.0)),
    ("G1_ACTUATOR_WAIST", (
        r"waist_pitch_joint", r"waist_roll_joint",
     ), Gain(kp=28.50124619574858, kd=1.814445686584846,
             armature=0.00721945, effort=35.0)),
    ("G1_ACTUATOR_ANKLE", (
        r".*_ankle_pitch_joint", r".*_ankle_roll_joint",
     ), Gain(kp=28.50124619574858, kd=1.814445686584846,
             armature=0.00721945, effort=35.0)),
)


def expand_gains(groups=HEFT_ACTUATOR_GROUPS) -> dict[str, Gain]:
    """Per-joint table over the 29 body joints.

    Each joint must match **exactly one** group. First-match-wins would also
    "work" and would hide an overlap introduced by an upstream edit; requiring
    uniqueness turns that into an error.
    """
    out: dict[str, Gain] = {}
    for name in DATASET_JOINT_NAMES:
        hits = [(gname, gain) for gname, pats, gain in groups
                if any(re.fullmatch(p, name) for p in pats)]
        if len(hits) != 1:
            raise SystemExit(
                f"[g1_actuators] {name!r} matches {len(hits)} actuator groups "
                f"({[h[0] for h in hits]}); expected exactly 1")
        out[name] = hits[0][1]
    return out


HEFT_GAINS: dict[str, Gain] = expand_gains()


def gain_arrays(order: tuple[str, ...] = CANONICAL_JOINT_NAMES):
    """(kp, kd, armature, effort) as float arrays in ``order``.

    Default is canonical order, which is what the policy's action and the PD
    target live in. Pass DATASET_JOINT_NAMES to get the order g1_5.xml and the
    motion files use, or a simulator's own joint-name tuple to get it in sim
    order -- always by name, never by position.
    """
    g = [HEFT_GAINS[n] for n in order]
    return (np.array([x.kp for x in g], dtype=np.float64),
            np.array([x.kd for x in g], dtype=np.float64),
            np.array([x.armature for x in g], dtype=np.float64),
            np.array([x.effort for x in g], dtype=np.float64))


# --------------------------------------------------------------------------- #
# The comparison that motivates this file
# --------------------------------------------------------------------------- #
def diff_against_sonic() -> list[tuple[str, str, float, float]]:
    """(joint, field, sonic_value, heft_value) for every disagreement.

    Imports the vendored SONIC tables lazily: this module must stay importable
    without g1_bridge on the path.
    """
    from g1_bridge.vendored.wbt.utils import bydmimic_utils as bu

    def _lookup(table: dict, joint: str) -> float:
        for pat, val in table.items():
            if re.fullmatch(pat, joint):
                return float(val)
        raise KeyError(joint)

    out = []
    for name in DATASET_JOINT_NAMES:
        heft = HEFT_GAINS[name]
        for field, table in (("kp", bu.stiffness_dict), ("kd", bu.damping_dict)):
            sonic = _lookup(table, name)
            if not np.isclose(sonic, getattr(heft, field), rtol=1e-9, atol=0):
                out.append((name, field, sonic, getattr(heft, field)))
    return out


def report_sonic_diff() -> None:
    rows = diff_against_sonic()
    if not rows:
        print("[g1_actuators] SONIC and HEFT gains agree on all 29 joints "
              "(unexpected -- verify this is really the HEFT table)")
        return
    print(f"[g1_actuators] SONIC's vendored gains differ from HEFT's on "
          f"{len({r[0] for r in rows})} joints -- do NOT reuse bydmimic_utils.kps/kds:")
    print(f"    {'joint':<28} {'field':<5} {'SONIC':>10} {'HEFT':>10} {'ratio':>7}")
    for joint, field, sonic, heft in rows:
        print(f"    {joint:<28} {field:<5} {sonic:>10.3f} {heft:>10.3f} "
              f"{sonic / heft:>7.2f}x")
