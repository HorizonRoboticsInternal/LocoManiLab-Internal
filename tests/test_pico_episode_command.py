from __future__ import annotations

from sim2real.rl_policy.control_mode import (
    PicoButtonState,
    resolve_keyboard_episode_command,
    resolve_pico_control_mode,
    resolve_pico_episode_command,
)


def test_keyboard_episode_command_mapping() -> None:
    assert resolve_keyboard_episode_command("c") == "toggle"
    assert resolve_keyboard_episode_command("x") == "discard"
    assert resolve_keyboard_episode_command("q") is None


def test_pico_y_tap_toggles_recording() -> None:
    previous = PicoButtonState()
    current = PicoButtonState(Y=True)

    assert resolve_pico_episode_command(current, previous) == "toggle"


def test_pico_y_held_does_not_retrigger_toggle() -> None:
    previous = PicoButtonState(Y=True)
    current = PicoButtonState(Y=True)

    assert resolve_pico_episode_command(current, previous) is None


def test_pico_y_plus_a_discards_episode() -> None:
    previous = PicoButtonState()
    current = PicoButtonState(Y=True, A=True)

    assert resolve_pico_episode_command(current, previous) == "discard"


def test_pico_y_plus_a_takes_precedence_over_plain_y_edge() -> None:
    # Both Y and A newly pressed in the same tick: this must resolve to "discard",
    # not "toggle" (which would otherwise fire since Y also has a fresh rising edge).
    previous = PicoButtonState()
    current = PicoButtonState(Y=True, A=True)

    assert resolve_pico_episode_command(current, previous) == "discard"


def test_pico_x_is_not_consumed_for_episode_commands() -> None:
    # X already has a live meaning inside teleop/pico_retarget_pub.py (its own
    # live/pause toggle) -- rl_policy's episode-command resolver must ignore it.
    previous = PicoButtonState()
    current = PicoButtonState(X=True)

    assert resolve_pico_episode_command(current, previous) is None


def test_bridge_engage_chord_does_not_discard_the_episode() -> None:
    # A+B+X+Y is g1_bridge's ModeManager engage chord, and it satisfies "Y and A".
    # Without the B/X guard every engage would silently discard the episode the
    # operator had just started recording -- the demos you cared about, gone.
    previous = PicoButtonState()
    current = PicoButtonState(A=True, B=True, X=True, Y=True)

    assert resolve_pico_episode_command(current, previous) is None
    # ... and releasing the chord must not leave a stray toggle behind either.
    assert resolve_pico_episode_command(PicoButtonState(), current) is None


def test_ax_pose_chord_does_not_toggle_recording() -> None:
    # A+X is the bridge's PLANNER<->POSE toggle; Y is not involved, but a held Y
    # plus a fresh X must still not fire an episode command.
    previous = PicoButtonState(Y=True)
    current = PicoButtonState(Y=True, A=True, X=True)

    assert resolve_pico_episode_command(current, previous) is None


def test_episode_command_does_not_interfere_with_control_mode() -> None:
    # A+B still resolves to "policy" control mode, unaffected by the new Y/X fields.
    previous = PicoButtonState()
    current = PicoButtonState(A=True, B=True)

    assert resolve_pico_control_mode(current, previous) == "policy"
    assert resolve_pico_episode_command(current, previous) is None
