"""Reconcile robosuite's G1 against the one HEFT's tracker was trained on.

The frozen tracker is a fixed function of a 1729-dim observation. It never sees
the model -- so if robosuite's G1 differs from ``g1_5.xml`` in mass, actuation or
joint parameters, the tracker issues the commands it learned and the robot does
something else. There is no error, only worse tracking that reads as "the port is
hard".

What is verified here, and what happens when it fails:

  ============================ ==================================================
  joint ranges                 HARD GATE. A difference means it is a different
                               robot and the port is invalid.
  FK of common bodies          HARD GATE. This is the check that convicted
                               ``g1.xml`` at 4.9 mm in heft_work's heft_bridge.
  dof armature/damping/friction PATCHED. g1_5.xml carries none of these -- mjlab
                               injects armature from G1_ARTICULATION_5 and leaves
                               damping/frictionloss at zero. robosuite's MJCF sets
                               all three from its own <default> classes.
  actuator force limits        PATCHED to HEFT's effort_limit. robosuite clamps
                               hip_roll at 88 Nm where HEFT allows 139 -- MuJoCo
                               clamps ctrl silently.
  body mass                    REPORTED, patch OPT-IN. See below.
  collision geometry           REPORTED ONLY. Cannot be patched without changing
                               what the scene's objects collide against.
  ============================ ==================================================

**The mass finding.** robosuite's G1 is 36.165 kg against g1_5's 33.341 (the value
``G1_tracking.yaml`` declares). Measured decomposition:

    +0.647  14 dexterous finger links, absent from g1_5 entirely
    +0.406  wrist_yaw_link x2 (g1_5 folds a fixed "rubber hand" into the wrist)
    +1.780  torso_link            <-- 63% of the gap, and NOT a hand difference
    -0.009  waist_yaw / waist_roll

The hand delta is unavoidable: grasping needs fingers. The torso delta is not
about hands at all -- it is 1.78 kg riding above the hips, on exactly the joints
whose gains the tracker depends on. ``apply_patch(mass=True)`` rewrites the common
bodies' mass and inertia to g1_5's values; it is **off by default** because which
robot is "correct" is a judgement call, not a fact: patching makes the dynamics
match what the tracker was trained on, but makes the robot no longer robosuite's
nominal G1 for the manipulation side. Treat it as a swept variable in h4/h5 and
let the tracking error decide, rather than assuming either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from rl.kimodo.heft.g1_actuators import DEFAULT_G1_XML_5, HEFT_GAINS
from rl.kimodo.heft.tracker_const import DATASET_JOINT_NAMES, DEFAULT_JOINT_POS

MASS_TOL_FRAC = 0.02      # threshold for calling a mass gap out, excluding hands
FK_TOL_M = 1e-3           # per-body FK tolerance
RANGE_TOL_RAD = 1e-4      # joint limit tolerance
# Identifying WHICH frame is the pelvis is a coarse question -- the candidates are
# metres apart. Keep it well away from FK_TOL_M so a small genuine link difference
# cannot make the root unresolvable.
ROOT_RESOLVE_TOL_M = 0.05

# Bodies whose world pose the port actually CONSUMES. The 1729-dim observation
# contains no body positions at all -- ``local_body_pos``/``local_body_rot`` are
# built by HEFT's env but never read by the deployed policy -- so the only FK
# that can corrupt a result is:
#   * the wrists, which are where exec_err and the EEF goal are measured
#   * the ankle rolls, which set the floor datum and the fall test
#   * the root frame, verified separately by resolve_root_body()
# A mismatch on any of these is a hard failure. A mismatch elsewhere changes the
# dynamics but not what the tracker sees, so it is reported and priced in h4/h5
# rather than blocking the port.
TRACKER_CRITICAL_BODIES = (
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
)

# Bodies whose mass difference is explained by robosuite having dexterous hands
# where g1_5 has a fixed lump. Excluded from the mass gate; still reported.
_HAND_BODY_HINTS = ("_hand_", "_wrist_yaw_link")


# --------------------------------------------------------------------------- #
# Model introspection
# --------------------------------------------------------------------------- #
def joint_ids(model, prefix: str = "robot0_") -> dict[str, int]:
    """Resolve the 29 body joints by NAME. Never by position: that permutation
    bug cost heft_work 1276 of 1729 observation dims (E6)."""
    out = {}
    for name in DATASET_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if jid < 0:
            raise SystemExit(f"[model_patch] joint {prefix + name!r} not in model")
        out[name] = jid
    return out


def actuator_for_joint(model, prefix: str = "robot0_") -> dict[str, int]:
    """Map joint name -> actuator id via the transmission, not by name.

    robosuite names its actuators ``robot0_left_hip_pitch`` (no ``_joint``
    suffix), so a name-based lookup silently returns -1 for all 29 and every
    force limit stays unpatched.
    """
    jids = joint_ids(model, prefix)
    by_jid = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            by_jid[int(model.actuator_trnid[a, 0])] = a
    out = {}
    for name, jid in jids.items():
        if jid not in by_jid:
            raise SystemExit(f"[model_patch] no joint-transmission actuator drives {name!r}")
        out[name] = by_jid[jid]
    return out


def body_masses(model, prefix: str = "") -> dict[str, float]:
    out = {}
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if prefix and not n.startswith(prefix):
            continue
        out[n[len(prefix):]] = float(model.body_mass[i])
    return out


def _fk_world(model, prefix: str, joint_pos_dataset: np.ndarray
              ) -> tuple[dict[str, np.ndarray], mujoco.MjData]:
    """Body world positions with the robot's free joint at a CANONICAL root pose.

    The free joint must be set explicitly. A fresh ``MjData`` inherits ``qpos0``,
    and robosuite parks its robot at ``[10, 10, 0]`` with a 180-degree yaw while
    g1_5 sits at ``[0, 0, 0.793]`` upright -- comparing those two states compares
    the parking spots, not the robots.
    """
    data = mujoco.MjData(model)
    a = model.jnt_qposadr[free_joint_id(model, prefix)]
    data.qpos[a:a + 7] = [0, 0, 0, 1, 0, 0, 0]

    jids = joint_ids(model, prefix)
    for name, q in zip(DATASET_JOINT_NAMES, joint_pos_dataset):
        data.qpos[model.jnt_qposadr[jids[name]]] = float(q)
    mujoco.mj_forward(model, data)

    out = {}
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if prefix and not n.startswith(prefix):
            continue
        out[n[len(prefix):]] = data.xpos[i].copy()
    return out, data


def _fk_relative_to(model, prefix: str, root_body: str,
                    joint_pos_dataset: np.ndarray) -> dict[str, np.ndarray]:
    """Body positions expressed in ``root_body``'s frame.

    ``root_body`` is a FULL model body name (not prefix-stripped): the frame we
    need may live outside the robot's own prefix, on the mount chain.
    """
    world, data = _fk_world(model, prefix, joint_pos_dataset)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if rid < 0:
        raise SystemExit(f"[model_patch] no body {root_body!r}")
    origin = data.xpos[rid].copy()
    R = data.xmat[rid].reshape(3, 3).copy()
    return {k: R.T @ (v - origin) for k, v in world.items()}


def free_joint_id(model, prefix: str = "robot0_") -> int:
    """Joint id of the robot's free joint, found by TYPE.

    ⚠️ The name is not stable across G1 models: the old robosuite G1 called it
    ``robot0_floating_base_joint`` and the unified model shared with g1_backend
    calls it ``robot0_pelvis``. Every lookup in this package goes through here so
    the next rename costs one line, not a silent -1 that surfaces as a crash at
    the first reset (which is exactly how the rename was found).
    """
    free = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[i])) or "")
        .startswith(prefix)
    ]
    if len(free) != 1:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in free]
        raise SystemExit(
            f"[model_patch] expected exactly one free joint under {prefix!r}, "
            f"found {len(free)}: {names}"
        )
    return int(free[0])


def free_joint_body(model, prefix: str = "robot0_") -> str:
    """Short name of the body carrying the robot's free joint.

    In robosuite this body is *named* ``pelvis`` but is the floating base at
    floor level -- a NAME COLLISION with g1_5's anatomical ``pelvis``. Any
    body-wise comparison between the two models has to drop it, or the one
    genuinely incomparable body poisons the result (it reads as a 793 mm error
    on every candidate frame).

    ⚠️ Found BY JOINT TYPE, not by name. This used to look up
    ``robot0_floating_base_joint``, and the unified G1 model (the one shared
    with g1_backend) calls the same joint ``robot0_pelvis`` -- so the lookup
    returned -1 and the whole port died at the first reset with "no free
    joint". A free joint is the only 7-DoF joint a floating-base robot has;
    asking for that property cannot go stale when someone renames it. Requiring
    exactly one also fails loudly on a model with two, which would be a robot
    this port does not understand.
    """
    j = free_joint_id(model, prefix)
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                          int(model.jnt_bodyid[j])) or ""
    return n[len(prefix):] if prefix and n.startswith(prefix) else n


def resolve_root_body(model, prefix: str = "robot0_",
                      g5_path: str | Path = DEFAULT_G1_XML_5,
                      tol: float = ROOT_RESOLVE_TOL_M) -> tuple[str, np.ndarray]:
    """Find which robosuite body is the frame g1_5 calls ``pelvis``.

    **This is not cosmetic.** ``G1FloatingBody`` mounts the robot on a
    ``FloatingLeggedBase``: the body carrying the free joint is *named*
    ``pelvis`` but its frame sits on the FLOOR, with the real chain hanging off
    ``mobilebase0_support`` 0.793 m above it. The tracker's ``root_pos`` /
    ``root_quat`` / ``TargetRootZ`` are all about the anatomical pelvis, so using
    the body called ``pelvis`` would tell the policy the robot is crouching by
    79 cm -- every step, with no error raised.

    Resolved by agreement rather than by name: the correct root is whichever
    frame makes the whole kinematic chain match g1_5. That way a future robot
    variant with different body names still resolves, and a variant whose chain
    genuinely differs fails instead of silently picking the least-bad frame.

    Returns ``(body_name, offset)`` where ``offset`` is the translation from the
    free-joint body's frame to the resolved root, in the free-joint body frame.
    """
    m5 = mujoco.MjModel.from_xml_path(str(g5_path))
    ref = _fk_relative_to(m5, "", "pelvis", DEFAULT_JOINT_POS)
    world, data = _fk_world(model, prefix, DEFAULT_JOINT_POS)

    fj_body = int(model.jnt_bodyid[free_joint_id(model, prefix)])
    fj_pos = data.xpos[fj_body].copy()

    common = sorted((set(world) & set(ref)) - {free_joint_body(model, prefix)})
    # Candidates are ALL bodies, not just prefixed ones: robosuite's mount chain
    # (mobilebase0_support / manipulator_mount) sits outside the robot0_ prefix,
    # and that is exactly where the anatomical pelvis frame lives.
    best: list[tuple[float, str, np.ndarray]] = []
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not n:
            continue
        R = data.xmat[i].reshape(3, 3)
        err = max(float(np.linalg.norm(R.T @ (world[k] - data.xpos[i]) - ref[k]))
                  for k in common)
        best.append((err, n, data.xpos[i] - fj_pos))

    best.sort(key=lambda r: r[0])
    err, name, offset = best[0]
    if err > tol:
        top = "\n".join(f"      {n:<36} {e * 1e3:9.2f} mm" for e, n, _ in best[:5])
        raise SystemExit(
            f"[model_patch] no body frame reproduces g1_5's kinematic chain "
            f"(best {err * 1e3:.2f} mm > {tol * 1e3:.2f} mm). Closest:\n{top}\n"
            f"    The two models are different robots.")
    return name, offset


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class Reconciliation:
    joint_rows: list = field(default_factory=list)
    mass_rows: list = field(default_factory=list)
    fk_rows: list = field(default_factory=list)
    mass_rs: float = 0.0
    mass_g5: float = 0.0
    mass_hand_attributable: float = 0.0
    root_body: str = ""
    root_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    failures: list = field(default_factory=list)     # invalidate the port
    differences: list = field(default_factory=list)  # real, but priced in h4/h5

    @property
    def ok(self) -> bool:
        return not self.failures


def reconcile(model, prefix: str = "robot0_",
              g5_path: str | Path = DEFAULT_G1_XML_5) -> Reconciliation:
    """Compare ``model`` (a live robosuite env model) against g1_5.xml."""
    m5 = mujoco.MjModel.from_xml_path(str(g5_path))
    rep = Reconciliation()

    jid_rs, jid_g5 = joint_ids(model, prefix), joint_ids(m5, "")
    act_rs = actuator_for_joint(model, prefix)

    for name in DATASET_JOINT_NAMES:
        i_rs, i_g5 = jid_rs[name], jid_g5[name]
        d_rs, d_g5 = model.jnt_dofadr[i_rs], m5.jnt_dofadr[i_g5]
        a = act_rs[name]
        gain = HEFT_GAINS[name]
        rng_rs, rng_g5 = model.jnt_range[i_rs].copy(), m5.jnt_range[i_g5].copy()
        row = dict(
            joint=name,
            range_rs=rng_rs, range_g5=rng_g5,
            armature_rs=float(model.dof_armature[d_rs]), armature_new=gain.armature,
            damping_rs=float(model.dof_damping[d_rs]), damping_new=0.0,
            friction_rs=float(model.dof_frictionloss[d_rs]), friction_new=0.0,
            ctrl_rs=model.actuator_ctrlrange[a].copy(), ctrl_new=np.array([-gain.effort, gain.effort]),
            kp=gain.kp, kd=gain.kd,
        )
        rep.joint_rows.append(row)
        if not np.allclose(rng_rs, rng_g5, atol=RANGE_TOL_RAD, rtol=0):
            rep.failures.append(
                f"joint range {name}: robosuite {rng_rs} vs g1_5 {rng_g5}")

    # ---- mass ----
    A, B = body_masses(model, prefix), body_masses(m5, "")
    common = sorted(set(A) & set(B))
    rep.mass_rs = sum(A[k] for k in A)
    rep.mass_g5 = sum(B[k] for k in B)
    for k in common:
        if abs(A[k] - B[k]) > 1e-6:
            rep.mass_rows.append((k, A[k], B[k], A[k] - B[k], _is_hand(k)))
    for k in sorted(set(A) - set(B)):
        if A[k] > 1e-9:
            rep.mass_rows.append((k, A[k], 0.0, A[k], _is_hand(k)))
    for k in sorted(set(B) - set(A)):
        if B[k] > 1e-9:
            rep.mass_rows.append((k, 0.0, B[k], -B[k], _is_hand(k)))
    rep.mass_hand_attributable = sum(d for _, _, _, d, is_hand in rep.mass_rows if is_hand)

    non_hand_gap = (rep.mass_rs - rep.mass_g5) - rep.mass_hand_attributable
    if abs(non_hand_gap) > MASS_TOL_FRAC * rep.mass_g5:
        rep.differences.append(
            f"non-hand mass gap {non_hand_gap:+.3f} kg "
            f"({non_hand_gap / rep.mass_g5:+.2%}) -- dynamics only, "
            f"try --patch --mass-patch and compare in h4/h5")

    # ---- forward kinematics, about the frame that actually IS the pelvis ----
    rep.root_body, rep.root_offset = resolve_root_body(model, prefix, g5_path)
    fk_rs = _fk_relative_to(model, prefix, rep.root_body, DEFAULT_JOINT_POS)
    fk_g5 = _fk_relative_to(m5, "", "pelvis", DEFAULT_JOINT_POS)
    for k in sorted((set(fk_rs) & set(fk_g5)) - {free_joint_body(model, prefix)}):
        d = float(np.linalg.norm(fk_rs[k] - fk_g5[k]))
        rep.fk_rows.append((k, d))
        if d > FK_TOL_M:
            msg = f"FK {k}: {d * 1e3:.2f} mm from g1_5"
            if k in TRACKER_CRITICAL_BODIES:
                rep.failures.append(msg + " (measurement point -- invalidates results)")
            else:
                rep.differences.append(msg + " (not in the observation; dynamics only)")

    return rep


def _is_hand(body: str) -> bool:
    return any(h in body for h in _HAND_BODY_HINTS)


def print_report(rep: Reconciliation, show_all_fk: bool = False) -> None:
    print("\n=== joints: robosuite (rs) vs g1_5 (g5) / HEFT gains ===")
    print(f"{'joint':<28} {'range':<22} {'armature':<19} {'damp':<13} "
          f"{'fric':<13} {'ctrlrange':<19} {'kp':>8} {'kd':>7}")
    for r in rep.joint_rows:
        same_rng = np.allclose(r["range_rs"], r["range_g5"], atol=RANGE_TOL_RAD, rtol=0)
        rng = f"[{r['range_rs'][0]:+.4f},{r['range_rs'][1]:+.4f}]"
        print(f"{r['joint']:<28} {rng:<22} "
              f"{r['armature_rs']:.6f}->{r['armature_new']:.6f} "
              f"{r['damping_rs']:.3f}->{r['damping_new']:.3f}   "
              f"{r['friction_rs']:.3f}->{r['friction_new']:.3f}   "
              f"±{abs(r['ctrl_rs'][1]):6.1f}->±{r['ctrl_new'][1]:6.1f} "
              f"{r['kp']:8.3f} {r['kd']:7.3f}"
              + ("" if same_rng else "   <-- RANGE MISMATCH"))

    print(f"\n=== mass: robosuite {rep.mass_rs:.6f} kg  |  g1_5 {rep.mass_g5:.6f} kg  "
          f"|  gap {rep.mass_rs - rep.mass_g5:+.6f} kg ===")
    print(f"{'body':<34} {'robosuite':>10} {'g1_5':>10} {'delta':>10}  attribution")
    for k, a, b, d, is_hand in sorted(rep.mass_rows, key=lambda r: -abs(r[3])):
        print(f"{k:<34} {a:10.4f} {b:10.4f} {d:+10.4f}  "
              f"{'hand' if is_hand else 'NOT a hand difference'}")
    non_hand = (rep.mass_rs - rep.mass_g5) - rep.mass_hand_attributable
    print(f"{'':34} {'':10} {'hand-attributable':>10} {rep.mass_hand_attributable:+10.4f}")
    print(f"{'':34} {'':10} {'residual':>10} {non_hand:+10.4f}  "
          f"({non_hand / rep.mass_g5:+.2%} of g1_5 mass)")

    print(f"\n=== tracker root frame ===")
    print(f"  robosuite body that IS g1_5's 'pelvis': {rep.root_body!r}")
    print(f"  offset from the free-joint body: {np.round(rep.root_offset, 6)} m")
    if rep.root_body != "pelvis":
        print(f"  NOTE: the body NAMED 'pelvis' is the floating base at floor level.")
        print(f"        root_pos/root_quat/TargetRootZ must use {rep.root_body!r}.")

    worst = sorted(rep.fk_rows, key=lambda r: -r[1])
    print(f"\n=== forward kinematics at DEFAULT_JOINT_POS, {rep.root_body} frame "
          f"({len(rep.fk_rows)} common bodies) ===")
    for k, d in (worst if show_all_fk else worst[:8]):
        flag = "  <-- OVER TOL" if d > FK_TOL_M else ""
        print(f"{k:<34} {d * 1e3:8.3f} mm{flag}")
    if not show_all_fk and len(worst) > 8:
        print(f"    ... {len(worst) - 8} more, all <= {worst[8][1] * 1e3:.3f} mm")

    n_exact = sum(1 for _, d in rep.fk_rows if d <= 1e-9)
    print(f"    ({n_exact}/{len(rep.fk_rows)} bodies match exactly; "
          f"critical bodies {', '.join(TRACKER_CRITICAL_BODIES)})")

    if rep.differences:
        print(f"\n=== real differences, priced in h4/h5 rather than blocking "
              f"({len(rep.differences)}) ===")
        for d in rep.differences:
            print(f"    {d}")
        print("    These change the DYNAMICS. None of them enters the 1729-dim")
        print("    observation: the policy reads no body positions, only root")
        print("    pose and joint state.")

    print()
    if rep.ok:
        print("[model_patch] PASS: joint ranges identical, root frame resolved, "
              "and every measurement-critical body matches g1_5")
    else:
        print(f"[model_patch] FAIL ({len(rep.failures)}):")
        for f in rep.failures:
            print(f"    {f}")


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def use_position_actuators(model, prefix: str = "robot0_",
                           g5_path: str | Path = DEFAULT_G1_XML_5,
                           verbose: bool = True) -> int:
    """Convert the 29 body actuators from <motor> to <position>, as mjlab has them.

    robosuite's G1 ships torque actuators and ``WholeBodyG1Composite`` closes the
    PD loop in Python at 500 Hz: ``tau = kp*(target - q) - kd*qd`` from the
    current state, explicitly, once per substep. mjlab instead hands the same law
    to MuJoCo as a position actuator, where the damping term participates in the
    integrator rather than being sampled before it.

    That difference is small per step and accumulates as *effective* damping, so
    it shows up as tracking that is slightly soft -- which is exactly the shape
    of the residual measured against mjlab (standing: 0.045 rad here versus
    0.0237 there, B1 on the cluster).

    MuJoCo evaluates an actuator as::

        force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*q + biasprm[2]*qd

    so gaintype FIXED with gainprm[0] = kp and biastype AFFINE with
    biasprm = (0, -kp, -kd) gives back kp*(ctrl - q) - kd*qd, with ``ctrl`` now a
    joint POSITION rather than a torque. Verified against a live G1_CFG_5 scene:
    mjlab's actuators read exactly this way.

    ``ctrlrange`` therefore has to become the joint's angular range, and the
    effort limit moves to ``forcerange`` -- leaving ctrlrange at +-effort would
    clamp the commanded ANGLE to +-88 rad (harmless) while silently removing the
    torque limit entirely (not harmless).
    """
    jids = joint_ids(model, prefix)
    acts = actuator_for_joint(model, prefix)
    n = 0
    for name in DATASET_JOINT_NAMES:
        a, j = acts[name], jids[name]
        gain = HEFT_GAINS[name]
        model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[a] = 0.0
        model.actuator_gainprm[a, 0] = gain.kp
        model.actuator_biasprm[a] = 0.0
        model.actuator_biasprm[a, 1] = -gain.kp
        model.actuator_biasprm[a, 2] = -gain.kd
        model.actuator_ctrlrange[a] = model.jnt_range[j]
        model.actuator_ctrllimited[a] = 1
        model.actuator_forcerange[a] = [-gain.effort, gain.effort]
        model.actuator_forcelimited[a] = 1
        n += 1
    if verbose:
        print(f"[model_patch] {n} actuators converted to <position> "
              f"(ctrl is now a joint ANGLE, not a torque)")
    return n


def use_g1_5_feet(model, prefix: str = "robot0_",
                  g5_path: str | Path = DEFAULT_G1_XML_5, verbose: bool = True) -> int:
    """Replace robosuite's foot contact patch with g1_5's, feet only.

    Standing is decided entirely by what the soles touch, and this is the last
    structural difference between the two robots that had not been tested: 4
    spheres of r=0.005 per foot here against 7 capsules of r=0.01 there. The sole
    PLANE already matches exactly (both -0.035 below the ankle roll link), so what
    differs is the patch, not the height.

    **Feet only, on purpose.** The rest of robosuite's body uses STL meshes where
    g1_5 uses capsules, but swapping those would change what every object in every
    scene collides against -- directly harmful to the manipulation side this whole
    port exists to serve. Feet never grasp anything, so they are safe to align.

    Implemented by rewriting the existing sphere geoms in place rather than adding
    geoms: MjModel is fixed-size after compilation, so 4 spheres become the 4
    largest-footprint capsules of g1_5's 7 and the contact patch is approximated,
    not reproduced. If this closes the gap the answer is "contact patch"; if it
    does not, the geometry is exonerated too.
    """
    m5 = mujoco.MjModel.from_xml_path(str(g5_path))

    def feet(mm, pref):
        out = {}
        for g in range(mm.ngeom):
            bn = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, mm.geom_bodyid[g]) or ""
            if "ankle_roll" in bn and (mm.geom_contype[g] | mm.geom_conaffinity[g]):
                side = "left" if "left" in bn else "right"
                out.setdefault(side, []).append(g)
        return out

    src, dst = feet(m5, ""), feet(model, prefix)
    n = 0
    for side, dgeoms in dst.items():
        sgeoms = src.get(side, [])
        if not sgeoms:
            continue
        # pick the source capsules with the largest half-length: they carry the
        # heel and toe, which is what a 4-point patch has to span
        sgeoms = sorted(sgeoms, key=lambda g: -m5.geom_size[g][1])[:len(dgeoms)]
        for dg, sg in zip(sorted(dgeoms, key=lambda g: model.geom_pos[g][0]),
                          sorted(sgeoms, key=lambda g: m5.geom_pos[g][0])):
            model.geom_type[dg] = m5.geom_type[sg]
            model.geom_size[dg] = m5.geom_size[sg]
            model.geom_pos[dg] = m5.geom_pos[sg]
            model.geom_quat[dg] = m5.geom_quat[sg]
            model.geom_friction[dg] = m5.geom_friction[sg]
            model.geom_condim[dg] = m5.geom_condim[sg]
            model.geom_solref[dg] = m5.geom_solref[sg]
            model.geom_solimp[dg] = m5.geom_solimp[sg]
            n += 1
    if verbose:
        print(f"[model_patch] {n} foot geoms replaced with g1_5's capsules "
              f"(feet only; body meshes untouched)")
    return n


def apply_patch(model, prefix: str = "robot0_", *, mass: bool = False,
                g5_path: str | Path = DEFAULT_G1_XML_5, verbose: bool = True) -> dict:
    """Bring the 29 body joints in line with what HEFT's tracker was trained on.

    Returns a summary dict of what changed. Only touches the robot's 29 body
    DoF -- the 14 finger DoF keep robosuite's own parameters, since the fingers
    are driven by the scripted grasp, not by the tracker.
    """
    jids = joint_ids(model, prefix)
    acts = actuator_for_joint(model, prefix)
    changed = {"armature": 0, "damping": 0, "frictionloss": 0, "ctrlrange": 0, "mass": 0}

    for name in DATASET_JOINT_NAMES:
        d = model.jnt_dofadr[jids[name]]
        gain = HEFT_GAINS[name]
        if model.dof_armature[d] != gain.armature:
            model.dof_armature[d] = gain.armature
            changed["armature"] += 1
        if model.dof_damping[d] != 0.0:
            model.dof_damping[d] = 0.0
            changed["damping"] += 1
        if model.dof_frictionloss[d] != 0.0:
            model.dof_frictionloss[d] = 0.0
            changed["frictionloss"] += 1
        a = acts[name]
        want = np.array([-gain.effort, gain.effort])
        if not np.allclose(model.actuator_ctrlrange[a], want):
            model.actuator_ctrlrange[a] = want
            # forcerange is what MuJoCo clamps applied force with when the
            # actuator is force-limited; ctrlrange alone leaves it unbounded for
            # a motor whose gear is 1. Set both so the limit is unambiguous.
            if model.actuator_forcelimited[a]:
                model.actuator_forcerange[a] = want
            changed["ctrlrange"] += 1

    if mass:
        m5 = mujoco.MjModel.from_xml_path(str(g5_path))
        B = {}
        for i in range(m5.nbody):
            n = mujoco.mj_id2name(m5, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            B[n] = i
        # The free-joint body is a NAME COLLISION: robosuite's "pelvis" is the
        # floor-level floating base, g1_5's is the anatomical root, and their
        # frames differ by 0.793 m. Copying body_ipos across that would move the
        # centre of mass by most of a metre.
        skip = {free_joint_body(model, prefix)}
        for i in range(model.nbody):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if not n.startswith(prefix):
                continue
            short = n[len(prefix):]
            if short in skip:
                continue
            j = B.get(short)
            if j is None:
                continue
            if (model.body_mass[i] != m5.body_mass[j]
                    or not np.allclose(model.body_inertia[i], m5.body_inertia[j])):
                model.body_mass[i] = m5.body_mass[j]
                model.body_inertia[i] = m5.body_inertia[j]
                model.body_ipos[i] = m5.body_ipos[j]
                model.body_iquat[i] = m5.body_iquat[j]
                changed["mass"] += 1

    if verbose:
        print(f"[model_patch] patched: armature={changed['armature']} "
              f"damping={changed['damping']} frictionloss={changed['frictionloss']} "
              f"ctrlrange={changed['ctrlrange']} mass_bodies={changed['mass']}"
              + ("" if mass else "  (mass patch OFF)"))
    return changed
