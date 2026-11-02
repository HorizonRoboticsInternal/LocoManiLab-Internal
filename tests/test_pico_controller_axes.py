"""Joystick axes survive the GMR relay path, end to end.

While a GMR-driven backend runs, the retargeting publisher owns the only PICO
SDK connection, so every other consumer reads controller input relayed over ZMQ
(``GMRButtonRelay``). That channel used to carry buttons and hand triggers but
not the thumbsticks, which is invisible for a pose-tracking backend and fatal
for a locomotion one: ALMI's mode chords worked, the operator's body tracked,
and the sticks read centred forever, so the robot never walked.

These pin the three links in that chain: the wire format carries the axes, a
pre-axis publisher still decodes, and the relay surfaces them under the same
key and convention the direct SDK sources use.
"""

from __future__ import annotations

import numpy as np
import pytest

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from g1_bridge.pico.vr_source import empty_buttons
from sim2real.utils.common import (
    _PICO_CONTROLLER_STATE_STRUCT_V1,
    PicoControllerStateMessage,
)


def test_axes_round_trip():
    msg = PicoControllerStateMessage(
        timestamp_ns=123,
        A=True,
        Y=True,
        left_axis=(-0.5, 0.75),
        right_axis=(0.25, -1.0),
    )
    back = PicoControllerStateMessage.from_bytes(msg.to_bytes())
    assert (back.A, back.B, back.X, back.Y) == (True, False, False, True)
    assert back.timestamp_ns == 123
    assert back.left_axis == pytest.approx((-0.5, 0.75))
    assert back.right_axis == pytest.approx((0.25, -1.0))


def test_pre_axis_publisher_still_decodes():
    """A stale publisher (its own venv, its own process) must degrade to
    centred sticks, not make every consumer throw."""
    legacy = _PICO_CONTROLLER_STATE_STRUCT_V1.pack(7, 1, 0, 1, 0)
    msg = PicoControllerStateMessage.from_bytes(legacy)
    assert (msg.A, msg.B, msg.X, msg.Y) == (True, False, True, False)
    assert msg.left_axis == (0.0, 0.0) and msg.right_axis == (0.0, 0.0)


def test_a_truncated_message_is_still_rejected():
    with pytest.raises(ValueError, match="invalid size"):
        PicoControllerStateMessage.from_bytes(b"\x00" * 5)


def test_axis_key_and_shape_match_the_direct_sources():
    """The relay must not invent its own key or convention."""
    blank = empty_buttons()
    assert blank["left_axis"] == (0.0, 0.0)
    assert blank["right_axis"] == (0.0, 0.0)


def test_relay_surfaces_axes_from_the_wire():
    """Drive GMRButtonRelay's own decode path, over the real wire format."""
    import threading

    from g1_bridge.gmr_utils.button_relay import GMRButtonRelay

    relay = GMRButtonRelay.__new__(GMRButtonRelay)  # no ZMQ connect
    relay._buttons = empty_buttons()
    relay._lock = threading.Lock()

    wire = PicoControllerStateMessage(
        timestamp_ns=1, A=True, left_axis=(0.3, -0.9), right_axis=(-0.2, 0.4)
    ).to_bytes()
    relay._apply_controller_message(PicoControllerStateMessage.from_bytes(wire))

    assert relay._buttons["a"] is True
    assert relay._buttons["left_axis"] == pytest.approx((0.3, -0.9))
    assert relay._buttons["right_axis"] == pytest.approx((-0.2, 0.4))


def test_relay_axes_reach_get_latest():
    """...and come out of the PicoVRSource surface a backend actually reads."""
    import threading

    from g1_bridge.gmr_utils.button_relay import GMRButtonRelay

    relay = GMRButtonRelay.__new__(GMRButtonRelay)
    relay._buttons = empty_buttons()
    relay._lock = threading.Lock()
    relay._apply_controller_message(
        PicoControllerStateMessage(left_axis=(1.0, -1.0), right_axis=(0.5, 0.5))
    )
    buttons = relay.get_latest()["buttons"]
    assert buttons["left_axis"] == pytest.approx((1.0, -1.0))
    assert buttons["right_axis"] == pytest.approx((0.5, 0.5))


def test_almi_steers_from_relayed_axes():
    """The payoff: a relayed stick has to move ALMI's twist command.

    Full-deflection left stick forward is the trained maximum, and this is the
    exact `buttons` shape GMRButtonRelay hands the backend.
    """
    import yaml

    from g1_bridge.backends.almi import AlmiBackend
    from sim2real.rl_policy.families import policy_config_path
    from test_almi_backend import _FakeCtx, _Resources

    policy_config = yaml.safe_load(policy_config_path("almi").read_text())
    backend = AlmiBackend()
    ctx = _FakeCtx(policy_config)
    backend.on_load(ctx, cfg={}, res=_Resources())
    backend.reset(ctx)

    buttons = empty_buttons()
    buttons["left_axis"] = (0.0, 1.0)  # full forward
    buttons["right_axis"] = (-1.0, 0.0)  # full left turn

    class _Tick:
        pass

    _Tick.buttons = buttons
    twist = backend._twist_from_sticks(_Tick(), driving=True)
    assert twist[0] == pytest.approx(0.8), "vx did not follow the relayed stick"
    assert twist[2] == pytest.approx(1.8), "wz did not follow the relayed stick"

    # ...and a centred stick really is a stop, not a stale command.
    buttons["left_axis"] = (0.0, 0.0)
    buttons["right_axis"] = (0.0, 0.0)
    np.testing.assert_allclose(
        backend._twist_from_sticks(_Tick(), driving=True)[:3], 0.0
    )
