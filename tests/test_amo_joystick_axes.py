"""Right-stick axis arbitration for the AMO backend.

The bug these guard against is not a crash and not a wrong number in isolation
-- every individual scaler was fine. It was that both channels live on the same
stick and the *unintended* one (torso height, deadzone 0.15) was more sensitive
than the intended one (turning, deadzone 0.2), so a thumb pushing sideways to
turn crossed the height threshold first and the torso lurched on essentially
every turn.

So the assertions are about the pair, not the parts: what happens to height
while turning, what happens to turning while adjusting height, and whether the
height the operator dialled in survives the things that should not disturb it.

Run: ``python tests/test_amo_joystick_axes.py``  (no sim, no PICO, no GPU)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_bridge.backends import make_backend  # noqa: E402
from g1_bridge.backends.amo import (  # noqa: E402
    _AMO_TORSO_HEIGHT_DEADZONE,
    _AMO_TORSO_HEIGHT_RANGE,
    _AMO_TORSO_HEIGHT_RATE,
    _AMO_YAW_DEADZONE,
    _AMO_YAW_RELEASE,
    _POLICY_DT,
    _amo_height_rate,
)

#: commands layout: [vx, target_yaw, vy, height_delta, yaw, pitch, roll, toggle]
_HEIGHT = 3

#: ``_amo_commands`` is a float32 array, so every value read back through it
#: carries float32 resolution (~6e-8 at 0.3). Comparisons against float64
#: arithmetic need this; comparisons of a value against ITSELF (held, frozen)
#: are still exact and stay at ==.
_F32 = 1e-6


class ProcessPicoVRSource:
    """The name is the point -- _tick_amo_commands gates every stick read on
    ``type(ctx.pico_source).__name__`` being a real source."""


class Ctx:
    pico_source = ProcessPicoVRSource()


def _backend():
    b = make_backend("amo")
    b._amo_torso_control_mode = "joystick"
    return b


def _push(b, n=1, rx=0.0, ry=0.0, lx=0.0, ly=0.0):
    """Hold the sticks at a fixed position for n ticks; return the height."""
    buttons = {"left_axis": (lx, ly), "right_axis": (rx, ry)}
    for _ in range(n):
        b._tick_amo_commands(Ctx(), buttons)
    return float(b._amo_commands[_HEIGHT])


# --------------------------------------------------------------------------- #
def test_turning_does_not_move_height():
    """The reported bug, and its diagonal form. Before the fix the diagonal
    case reached 0.15 m on the very first tick."""
    print("J1 turning suppresses height")
    for name, ry in (("pure turn", 0.0), ("diagonal turn", 0.5), ("sloppy", 0.35)):
        b = _backend()
        h = _push(b, n=50, rx=0.8, ry=ry)
        assert h == 0.0, f"{name} (rx=0.8, ry={ry}) moved height to {h:.4f} m"
        assert b._amo_vyaw != 0.0, f"{name}: the robot should still be turning"
        print(
            f"  ok  {name:<14} rx=0.8 ry={ry:<4} -> height {h:.3f}, vyaw {b._amo_vyaw}"
        )
    print("PASS: no stick position that turns can also drive height")


def test_vertical_push_drives_height():
    """...and the axis still works when the operator actually means it."""
    print("J2 vertical push")
    b = _backend()
    h = _push(b, n=50, rx=0.0, ry=0.8)
    expect = _AMO_TORSO_HEIGHT_RATE * _amo_height_rate(0.8) * _POLICY_DT * 50
    assert abs(h - expect) < _F32, f"height {h:.6f}, expected {expect:.6f}"
    assert b._amo_vyaw == 0.0, "a vertical push must not turn the robot"
    assert h > 0.0, "positive ry must RAISE the torso (sign convention)"
    print(f"  ok  50 ticks at ry=0.8 -> {h:.4f} m, vyaw 0")

    down = _push(_backend(), n=50, rx=0.0, ry=-0.8)
    assert down < 0.0 and abs(down + h) < _F32, f"asymmetric down/up: {down} vs {h}"
    print(f"  ok  ry=-0.8 -> {down:.4f} m (symmetric)")
    print("PASS: vertical pushes drive height, both directions, no turn bleed")


def test_height_holds_on_release():
    """Hold-style: the operator dials a height in and it stays. Before, the
    channel was proportional and snapped back to the 0.75 baseline."""
    print("J3 hold on release")
    b = _backend()
    h = _push(b, n=40, ry=0.8)
    assert h > 0.01
    held = _push(b, n=100, ry=0.0)
    assert abs(held - h) < 1e-12, f"height drifted {h:.4f} -> {held:.4f} after release"
    print(f"  ok  {h:.4f} m held for 2 s of centred stick")

    # ...and it survives a turn, which is the whole point of suppressing rather
    # than zeroing.
    during = _push(b, n=50, rx=0.9)
    assert abs(during - h) < 1e-12, f"turning changed a held height: {during:.4f}"
    after = _push(b, n=10, rx=0.0)
    assert abs(after - h) < 1e-12
    print(f"  ok  {h:.4f} m survived a 1 s turn and is still adjustable after")
    print("PASS: height holds through release and through turning")


def test_no_step_at_the_deadzone():
    """The old scaler used the raw ry, so crossing the deadzone jumped
    0.15 * 0.30 = 0.045 m in one tick."""
    print("J4 deadzone continuity")
    just_inside = _push(_backend(), n=1, ry=_AMO_TORSO_HEIGHT_DEADZONE - 1e-6)
    assert just_inside == 0.0, f"deadzone leaked: {just_inside}"
    just_outside = _push(_backend(), n=1, ry=_AMO_TORSO_HEIGHT_DEADZONE + 1e-3)
    assert 0.0 < just_outside < 1e-4, (
        f"one tick just past the deadzone moved {just_outside*1000:.2f} mm "
        "-- the renormalisation is missing"
    )
    assert _amo_height_rate(1.0) == 1.0, "full deflection must map to full rate"
    assert _amo_height_rate(-1.0) == -1.0
    print(f"  ok  first tick past the deadzone: {just_outside*1000:.4f} mm (was 45 mm)")
    print("PASS: height is continuous across the deadzone and proportional")


def test_hysteresis_has_no_dead_band_gap():
    """The Schmitt trigger must not leave a window where the stick neither
    turns nor adjusts height -- that would feel like a dead spot."""
    print("J5 hysteresis")
    assert _AMO_YAW_RELEASE < _AMO_YAW_DEADZONE, "release must be below the trigger"
    # Between release and deadzone, coming from centred: not turning yet, so
    # height must still respond.
    b = _backend()
    mid = 0.5 * (_AMO_YAW_RELEASE + _AMO_YAW_DEADZONE)
    h = _push(b, n=20, rx=mid, ry=0.8)
    assert h > 0.0, "a below-threshold rx must not suppress height"
    assert b._amo_vyaw == 0.0, "a below-threshold rx must not turn"
    # Latch on, then back down into the band: stays latched (no chatter).
    _push(b, n=1, rx=0.9, ry=0.8)
    frozen = float(b._amo_commands[_HEIGHT])
    still = _push(b, n=20, rx=mid, ry=0.8)
    assert abs(still - frozen) < 1e-12, "height resumed inside the hysteresis band"
    # Fully released -> height responds again.
    resumed = _push(b, n=20, rx=0.0, ry=0.8)
    assert resumed > still, "height did not resume after the stick recentred"
    print(
        f"  ok  band [{_AMO_YAW_RELEASE}, {_AMO_YAW_DEADZONE}] latches without chatter"
    )
    print("PASS: hysteresis latches cleanly with no dead band")


def test_travel_is_clamped():
    print("J6 clamp")
    b = _backend()
    h = _push(b, n=2000, ry=1.0)
    assert abs(h - _AMO_TORSO_HEIGHT_RANGE) < _F32, f"clamped to {h}, want 0.30"
    lo = _push(b, n=4000, ry=-1.0)
    assert abs(lo + _AMO_TORSO_HEIGHT_RANGE) < _F32, f"clamped to {lo}, want -0.30"
    print(f"  ok  travel stays within +/-{_AMO_TORSO_HEIGHT_RANGE} m (0.45-1.05 m)")
    print("PASS: height travel is clamped both ways")


def test_reset_clears_the_held_height():
    """A held offset that survives env.reset() would silently bias every
    subsequent episode -- and the recorded `height` column with it."""
    print("J7 reset")

    class FakeSim:
        class data:
            qpos = np.zeros(64)
            qvel = np.zeros(64)

    class ResetCtx(Ctx):
        sim = FakeSim()
        qpos_addrs = [None] * 29
        qvel_addrs = [None] * 29

    b = _backend()
    b._build_amo_joint_mapping()
    b._amo_ik = None
    h = _push(b, n=40, ry=0.9)
    assert h > 0.01
    b._reset_amo(ResetCtx())
    assert b._amo_torso_h_cmd == 0.0, "held height survived reset"
    assert b._amo_turning is False, "turning latch survived reset"
    assert float(b._amo_commands[_HEIGHT]) == 0.0
    print(f"  ok  {h:.4f} m cleared to 0.0 by _reset_amo")
    print("PASS: reset clears the held height and the latch")


def test_recorded_height_still_reconstructs():
    """`height_state` is 0.75 + commands[3] (amo.py) and the policy adapter does
    the same reconstruction -- changing how commands[3] evolves must not change
    what it means."""
    print("J8 recorded height")
    from g1_bridge.backends.amo import _AMO_TORSO_HEIGHT_DEFAULT

    b = _backend()
    _push(b, n=40, ry=0.8)
    recorded = _AMO_TORSO_HEIGHT_DEFAULT + float(b._amo_commands[_HEIGHT])
    assert 0.45 - 1e-9 <= recorded <= 1.05 + 1e-9, recorded
    assert recorded > _AMO_TORSO_HEIGHT_DEFAULT, "raising the torso must raise it"
    print(f"  ok  recorded height {recorded:.4f} m, in [0.45, 1.05]")
    print("PASS: the recorded height still reconstructs to a sane absolute value")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    tests = [
        test_turning_does_not_move_height,
        test_vertical_push_drives_height,
        test_height_holds_on_release,
        test_no_step_at_the_deadzone,
        test_hysteresis_has_no_dead_band_gap,
        test_travel_is_clamped,
        test_reset_clears_the_held_height,
        test_recorded_height_still_reconstructs,
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
    print("PASS: all AMO joystick-axis checks")
