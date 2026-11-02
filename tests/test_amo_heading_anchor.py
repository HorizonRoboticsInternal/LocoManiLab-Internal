"""AMO's target heading vs. the heading the robot actually spawned in.

``amo_commands[1]`` is an ABSOLUTE world heading -- sim2real's AMO observation
feeds the policy ``dyaw = wrap(pelvis_world_yaw - commands[1])`` -- while the
operator only ever supplies a *rate* (right stick) that gets integrated. So the
integral needs a starting point, and zero is the wrong one anywhere except
hardware, where the IMU yaw starts at ~0 by construction. Robocasa spawns the
G1 at a per-scene calibrated heading (2.81 rad in FloorPlan329, measured), so a
zeroed command meant "turn that whole angle out".

The failure was invisible until the operator moved, because the observation
suppresses ``dyaw`` entirely while standing in place: the robot stood correctly,
then swung ~160 deg mid-walk on the first stick push and carried on in the new
direction. An in-sim A/B (6 s of full-forward vx in FloorPlan329) measured
-160.4 deg of heading change with the anchor at 0 versus +0.9 deg with it
anchored.

These checks are the sim-free half of that: where the integral starts, that the
stick still steers relative to it, and that the recorded ``target_yaw`` column
stays the scene-independent relative heading a policy can actually learn.

Run: ``python tests/test_amo_heading_anchor.py``  (no sim, no PICO, no GPU)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_bridge.backends import make_backend  # noqa: E402
from g1_bridge.backends.amo import _POLICY_DT, _amo_world_yaw  # noqa: E402

#: commands layout: [vx, target_yaw, vy, height_delta, yaw, pitch, roll, toggle]
_TARGET_YAW = 1

#: FloorPlan329's calibrated spawn heading, as measured at the end of the
#: settling window in the scene the operator reported this from.
_SPAWN_YAW = 2.8129

_F32 = 1e-6


class ProcessPicoVRSource:
    """The name is the point -- _tick_amo_commands gates every stick read on
    ``type(ctx.pico_source).__name__`` being a real source."""


def _quat_at_yaw(yaw):
    """``[w, x, y, z]`` for a pure-yaw rotation."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


class Ctx:
    """Just the two surfaces the heading path touches."""

    pico_source = ProcessPicoVRSource()
    yaw = _SPAWN_YAW

    def read_pelvis_state(self):
        return _quat_at_yaw(self.yaw), np.zeros(3)


class FakeSim:
    class data:
        qpos = np.zeros(64)
        qvel = np.zeros(64)


class ResetCtx(Ctx):
    sim = FakeSim()
    qpos_addrs = [None] * 29
    qvel_addrs = [None] * 29


def _backend():
    b = make_backend("amo")
    b._amo_torso_control_mode = "joystick"
    b._build_amo_joint_mapping()
    b._amo_ik = None
    return b


def _push(b, ctx, n=1, rx=0.0, ry=0.0, lx=0.0, ly=0.0):
    """Hold the sticks for n ticks; return the absolute target heading."""
    buttons = {"left_axis": (lx, ly), "right_axis": (rx, ry)}
    for _ in range(n):
        b._tick_amo_commands(ctx, buttons)
    return float(b._amo_commands[_TARGET_YAW])


# --------------------------------------------------------------------------- #
def test_yaw_helper_matches_the_observation():
    """``_amo_world_yaw`` is subtracted from sim2real's own ``_quat_to_euler``
    yaw, so it has to be the same expression -- not merely the same on the pure
    yaw quaternions a test would think to use."""
    print("H1 yaw convention")
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q
        want = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        assert abs(_amo_world_yaw(q) - want) < 1e-12
    for yaw in (-3.0, -1.5, 0.0, 0.7, _SPAWN_YAW, 3.1):
        got = _amo_world_yaw(_quat_at_yaw(yaw))
        assert abs(got - yaw) < 1e-9, f"{yaw} -> {got}"
    print("  ok  200 random quats + pure-yaw round trips agree to 1e-12")
    print("PASS: the yaw helper is the observation's own convention")


def test_heading_starts_where_the_robot_is():
    """The reported bug. Before the fix ``commands[1]`` was 0 out of reset, so
    the first walking tick handed the policy the whole spawn yaw as error."""
    print("H2 anchor")
    b = _backend()
    ctx = Ctx()
    assert b._amo_yaw_anchor is None, "the anchor must not exist before a tick"

    anchor = b._ensure_yaw_anchor(ctx)
    assert abs(anchor - _SPAWN_YAW) < 1e-9, anchor
    target = float(b._amo_commands[_TARGET_YAW])
    assert abs(target - _SPAWN_YAW) < _F32, f"target_yaw={target}, want the spawn yaw"

    dyaw = np.remainder(_SPAWN_YAW - target + np.pi, 2 * np.pi) - np.pi
    assert abs(dyaw) < _F32, f"policy would see {np.degrees(dyaw):.1f} deg of error"
    print(f"  ok  spawn yaw {np.degrees(_SPAWN_YAW):.1f} deg -> dyaw {dyaw:+.2e} rad")

    # Idle ticks must not move it: a robot standing where it spawned is already
    # on target, and the operator has asked for nothing.
    held = _push(b, ctx, n=100)
    assert abs(held - _SPAWN_YAW) < _F32, f"heading drifted to {held} while idle"
    print(f"  ok  2 s of centred sticks: still {np.degrees(held):.1f} deg")
    print("PASS: the heading integral starts at the robot's own heading")


def test_anchor_is_taken_once_and_survives_the_robot_turning():
    """Absolute is the point: a heading that re-anchored to the live pelvis
    would make the error the policy sees independent of how far it has actually
    turned, so a yaw disturbance would never be opposed."""
    print("H3 one-shot")
    b = _backend()
    ctx = Ctx()
    b._ensure_yaw_anchor(ctx)
    ctx.yaw = _SPAWN_YAW - 1.0  # the robot got pushed a radian off
    again = b._ensure_yaw_anchor(ctx)
    assert abs(again - _SPAWN_YAW) < 1e-9, "the anchor was re-taken mid-episode"
    assert abs(float(b._amo_commands[_TARGET_YAW]) - _SPAWN_YAW) < _F32
    print("  ok  a 1 rad disturbance leaves target_yaw where the operator put it")
    print("PASS: the anchor is one-shot, so heading error stays correctable")


def test_stick_steers_relative_to_the_anchor():
    """The right stick still integrates a rate; it just no longer defines the
    absolute heading on its own."""
    print("H4 steering")
    b = _backend()
    ctx = Ctx()
    b._ensure_yaw_anchor(ctx)
    target = _push(b, ctx, n=50, rx=-0.9)  # -rx -> +vyaw, i.e. turn left
    rel = float(b._amo_heading_rel)
    assert rel > 0.0, f"left turn integrated {rel}"
    assert abs(rel - b._amo_vyaw * _POLICY_DT * 50) < _F32
    assert abs(target - (_SPAWN_YAW + rel)) < _F32, (
        f"absolute {target} != anchor {_SPAWN_YAW} + relative {rel}"
    )
    print(f"  ok  1 s left turn -> +{np.degrees(rel):.1f} deg from the spawn heading")

    back = _push(b, ctx, n=50, rx=0.9)
    assert abs(float(b._amo_heading_rel)) < _F32, "the return turn did not cancel"
    assert abs(back - _SPAWN_YAW) < _F32, back
    print("  ok  turning back cancels exactly, to the spawn heading")
    print("PASS: the stick steers a heading relative to the episode start")


def test_recorded_heading_is_scene_independent():
    """``target_yaw`` is a recorded action column, and the images it is paired
    with carry no world frame -- so the same walk must produce the same label in
    every scene. Recording ``commands[1]`` would make it the spawn yaw."""
    print("H5 recording")
    per_scene = {}
    for spawn in (0.0, _SPAWN_YAW, -3.0, 1.2):
        b = _backend()
        ctx = Ctx()
        ctx.yaw = spawn
        b._ensure_yaw_anchor(ctx)
        _push(b, ctx, n=50, rx=-0.9)
        sample_yaw = float(b._amo_heading_rel)  # what _amo_teleop_data_sample records
        per_scene[spawn] = sample_yaw
        assert abs(float(b._amo_commands[_TARGET_YAW]) - (spawn + sample_yaw)) < _F32
    values = list(per_scene.values())
    assert max(values) - min(values) < _F32, f"same walk, different labels: {per_scene}"
    print(f"  ok  4 spawn headings -> one label {np.degrees(values[0]):.1f} deg")

    # ...and the VLA path puts it back into the world frame it needs.
    b = _backend()
    ctx = Ctx()
    b._ensure_yaw_anchor(ctx)
    b.set_velocity_command(
        {
            "torso_vx": np.array([0.6]),
            "torso_vy": np.array([0.0]),
            "torso_vyaw": np.array([0.0]),
            "target_yaw": np.array([values[0]]),
        }
    )
    replayed = float(b._amo_commands[_TARGET_YAW])
    assert abs(replayed - (_SPAWN_YAW + values[0])) < _F32, replayed
    print(f"  ok  replayed as {np.degrees(replayed):.1f} deg absolute in this scene")
    print("PASS: recorded relative, replayed absolute -- round trip closes")


def test_reset_clears_the_anchor_and_the_heading():
    """robosuite runs reset() BEFORE the scene teleports the robot, so the
    anchor must be dropped there and re-taken on the first post-settling tick --
    keeping it would pin episode 2 to episode 1's heading."""
    print("H6 reset")
    b = _backend()
    ctx = ResetCtx()
    b._ensure_yaw_anchor(ctx)
    _push(b, ctx, n=50, rx=-0.9)
    assert b._amo_yaw_anchor is not None and b._amo_heading_rel != 0.0

    b._reset_amo(ctx)
    assert b._amo_yaw_anchor is None, "the anchor survived reset"
    assert b._amo_heading_rel == 0.0, "the operator's turning survived reset"
    assert float(b._amo_commands[_TARGET_YAW]) == 0.0

    ctx.yaw = -1.0  # next episode, different scene heading
    assert abs(b._ensure_yaw_anchor(ctx) + 1.0) < 1e-9
    assert abs(float(b._amo_commands[_TARGET_YAW]) + 1.0) < _F32
    print("  ok  re-anchors to the next episode's own spawn heading (-57.3 deg)")
    print("PASS: reset drops the anchor; the next episode takes its own")


if __name__ == "__main__":
    tests = [
        test_yaw_helper_matches_the_observation,
        test_heading_starts_where_the_robot_is,
        test_anchor_is_taken_once_and_survives_the_robot_turning,
        test_stick_steers_relative_to_the_anchor,
        test_recorded_heading_is_scene_independent,
        test_reset_clears_the_anchor_and_the_heading,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
        print()
    if failed:
        print(f"{failed}/{len(tests)} checks FAILED")
        sys.exit(1)
    print("PASS: all AMO heading-anchor checks")
