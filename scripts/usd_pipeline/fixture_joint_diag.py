"""Diagnose a converted articulated fixture's hinge(s) in MuJoCo.

After converting a USD asset with ``usd_to_mjcf_fixture.py`` and authoring
collision geoms, run this to verify the hinge actually works before plugging
the fixture into a scene.

For every hinge joint in the fixture's ``model.xml`` this script reports:

  * axis, pivot, joint range
  * **axis-sign sanity**: sweeps the joint through its range, counts
    base<->child contacts and worst penetration at several angles. If contacts
    *grow* with positive angle, the joint is swinging *into* the body — almost
    always means the axis sign in the MJCF is flipped relative to the USD.
  * **gravity torque profile**: torque on the dof at several angles. This is
    what frictionloss has to dominate to make the joint "stay where placed".
  * **stays-where-placed check**: places the joint at sample angles, settles
    under gravity (with noslip_iterations bumped to 10), reports drift. Drift
    means frictionloss is too low *or* noslip_iterations did not get applied.
  * **drag handles**: minimum applied torque (both directions) to actually
    move the joint out of its placed angle — i.e. how stiff the drag feels.

Usage::

    conda activate tele_benchmark
    python -m scripts.fixture_joint_diag \\
        robocasa/models/assets/fixtures/articulated/ElectricCooker001/model.xml

Returns nonzero exit code on hard failures (model fails to load); diagnostics
are warnings printed to stdout.
"""

import argparse
import os
import sys
from collections import defaultdict

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def hinge_joints(model):
    out = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            out.append(jid)
    return out


def child_body_of_joint(model, jid):
    return int(model.jnt_bodyid[jid])


def parent_body_of(model, bid):
    return int(model.body_parentid[bid])


def descendants(model, root_bid):
    """All body ids in the subtree rooted at root_bid (inclusive)."""
    out = {root_bid}
    for bid in range(model.nbody):
        cur = bid
        while cur > 0 and cur != root_bid:
            cur = model.body_parentid[cur]
        if cur == root_bid:
            out.add(bid)
    return out


def cross_contacts(model, data, child_subtree, parent_subtree):
    """Contacts whose two geoms straddle the (child, parent) body split."""
    crosses = []
    for k in range(data.ncon):
        c = data.contact[k]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        in_child_1 = b1 in child_subtree
        in_child_2 = b2 in child_subtree
        in_par_1 = b1 in parent_subtree
        in_par_2 = b2 in parent_subtree
        if (in_child_1 and in_par_2) or (in_child_2 and in_par_1):
            crosses.append(c)
    return crosses


def jname(model, jid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    return n or f"<joint#{jid}>"


def bname(model, bid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return n or f"<body#{bid}>"


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def diag_joint(model, jid, n_settle_steps=4000, n_torque_steps=800):
    """Run all checks on a single hinge joint."""
    child_bid = child_body_of_joint(model, jid)
    parent_bid = parent_body_of(model, child_bid)
    child_sub = descendants(model, child_bid)
    parent_sub = descendants(model, parent_bid) - child_sub

    dofadr = int(model.jnt_dofadr[jid])
    qadr = int(model.jnt_qposadr[jid])
    lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
    axis = model.jnt_axis[jid].copy()
    pivot = model.jnt_pos[jid].copy()  # in child-body local frame

    print(f"\n=== joint {jname(model, jid)!r} ===")
    print(f"  child  = {bname(model, child_bid)}")
    print(f"  parent = {bname(model, parent_bid)}")
    print(f"  axis   = {axis.round(3)}   pivot(local) = {pivot.round(4)}")
    print(f"  range  = [{lo:.3f}, {hi:.3f}] rad")
    print(f"  damping={model.dof_damping[dofadr]:.3g}  "
          f"frictionloss={model.dof_frictionloss[dofadr]:.3g}  "
          f"armature={model.dof_armature[dofadr]:.3g}  "
          f"stiffness={model.jnt_stiffness[jid]:.3g}")

    sample_angles = np.linspace(lo, hi, 6)

    # 1) child<->parent contacts vs angle
    data = mujoco.MjData(model)
    contacts_log = []
    penetration_log = []
    for a in sample_angles:
        data.qpos[qadr] = a
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        cs = cross_contacts(model, data, child_sub, parent_sub)
        contacts_log.append(len(cs))
        worst = min((c.dist for c in cs), default=0.0)
        penetration_log.append(worst)
    print("  --- child<->parent contacts vs angle ---")
    for a, n, w in zip(sample_angles, contacts_log, penetration_log):
        print(f"    angle={a:+.3f}  contacts={n:3d}  worst_pen={w:+.4f} m")

    # heuristic warnings
    if contacts_log[-1] > contacts_log[0] + 3:
        print("    [WARN] contacts grow with positive angle — the joint may be "
              "swinging INTO the body. Consider flipping the joint axis sign "
              "in model.xml.")
    if min(penetration_log) < -0.005:
        print("    [WARN] >5mm penetration somewhere in range. Either the "
              "collision geometry interlocks (replace mesh hulls with non-"
              "overlapping primitive boxes) or the pivot/axis is wrong.")

    # 2) gravity torque profile
    print("  --- gravity torque on dof vs angle ---")
    grav = []
    for a in sample_angles:
        data = mujoco.MjData(model)
        data.qpos[qadr] = a
        mujoco.mj_forward(model, data)
        grav.append(float(data.qfrc_bias[dofadr]))
    for a, g in zip(sample_angles, grav):
        print(f"    angle={a:+.3f}  qfrc_bias={g:+.3f} N·m")
    grav_max = max(abs(g) for g in grav)
    fric = float(model.dof_frictionloss[dofadr])
    if fric > 0:
        if fric < grav_max:
            print(f"    [WARN] frictionloss ({fric:.2f}) < |gravity|_max "
                  f"({grav_max:.2f}). Joint will drift open/closed.")
        else:
            print(f"    [OK] frictionloss ({fric:.2f}) >= |gravity|_max "
                  f"({grav_max:.2f}); should hold static if noslip_iterations>0.")

    # 3) stays-where-placed (with noslip_iterations bumped)
    print("  --- stays-where-placed (bump noslip_iterations=10 for the test) ---")
    saved_noslip = int(model.opt.noslip_iterations)
    model.opt.noslip_iterations = max(saved_noslip, 10)
    try:
        for a in sample_angles[1:-1]:
            data = mujoco.MjData(model)
            data.qpos[qadr] = a
            for _ in range(n_settle_steps):
                mujoco.mj_step(model, data)
            drift = float(data.qpos[qadr]) - a
            tag = "OK" if abs(drift) < 0.01 else "DRIFT"
            print(f"    placed@{a:+.3f}  ->  rest={float(data.qpos[qadr]):+.3f}  "
                  f"(drift {drift:+.3f})  [{tag}]")
    finally:
        model.opt.noslip_iterations = saved_noslip
    if saved_noslip < 10:
        print(f"    [NOTE] model.opt.noslip_iterations={saved_noslip}. Set this "
              f"to >=10 at runtime (or via <option noslip_iterations=10/>) so "
              f"frictionloss can fully saturate. In robosuite-merged scenes the "
              f"fixture's <option> is dropped — set it from the env hook.")

    # 4) drag handles (min torque to move out of mid-range rest)
    print("  --- min applied torque to overcome static + gravity (from mid) ---")
    mid = (lo + hi) / 2
    model.opt.noslip_iterations = max(saved_noslip, 10)
    try:
        for sign, label in [(+1, "open "), (-1, "close")]:
            move_tq = None
            for tq in np.arange(0.1, 5.01, 0.1):
                data = mujoco.MjData(model)
                data.qpos[qadr] = mid
                for _ in range(n_torque_steps):
                    data.qfrc_applied[dofadr] = sign * float(tq)
                    mujoco.mj_step(model, data)
                if abs(float(data.qpos[qadr]) - mid) > 0.2:
                    move_tq = float(tq)
                    break
            if move_tq is None:
                print(f"    {label}: needs >5 N·m (very stiff)")
            else:
                print(f"    {label}: ~{move_tq:.1f} N·m moves it")
    finally:
        model.opt.noslip_iterations = saved_noslip


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xml", help="path to fixture model.xml")
    args = ap.parse_args()

    if not os.path.exists(args.xml):
        sys.exit(f"no such file: {args.xml}")

    try:
        model = mujoco.MjModel.from_xml_path(args.xml)
    except Exception as e:
        sys.exit(f"failed to compile MJCF: {e}")

    print(f"loaded {args.xml}")
    print(f"  nbody={model.nbody}  ngeom={model.ngeom}  "
          f"nmesh={model.nmesh}  njnt={model.njnt}")
    print(f"  opt.timestep={model.opt.timestep}  "
          f"opt.noslip_iterations={model.opt.noslip_iterations}")

    hinges = hinge_joints(model)
    if not hinges:
        print("no hinge joints found.")
        return
    for jid in hinges:
        diag_joint(model, jid)


if __name__ == "__main__":
    main()
