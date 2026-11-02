from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ControlMode = Literal["init", "zero", "policy"]
EpisodeCommand = Literal["toggle", "discard"]


@dataclass(frozen=True)
class PicoButtonState:
    A: bool = False
    B: bool = False
    X: bool = False
    Y: bool = False


_KEYBOARD_CONTROL_MODES: dict[str, ControlMode] = {
    "i": "init",
    "o": "zero",
    "]": "policy",
}


def resolve_keyboard_control_mode(keycode: str) -> ControlMode | None:
    return _KEYBOARD_CONTROL_MODES.get(keycode)


_KEYBOARD_EPISODE_COMMANDS: dict[str, EpisodeCommand] = {
    "c": "toggle",
    "x": "discard",
}


def resolve_keyboard_episode_command(keycode: str) -> EpisodeCommand | None:
    """'c' toggles record start/stop, 'x' discards the in-progress episode.

    Matches the keyboard convention already used by downstream data-collection forks
    (e.g. LocoManiLab's heft/AMO/sonic controllers) so operators don't have to learn a new one.
    """
    return _KEYBOARD_EPISODE_COMMANDS.get(keycode)


_UNITREE_JOYSTICK_CONTROL_MODES: dict[str, ControlMode] = {
    "A": "init",
    "R2": "zero",
    "R1": "policy",
}


def resolve_unitree_joystick_control_mode(cur_key: str) -> ControlMode | None:
    return _UNITREE_JOYSTICK_CONTROL_MODES.get(cur_key)


def resolve_pico_control_mode(
    current: PicoButtonState,
    previous: PicoButtonState,
) -> ControlMode | None:
    current_combo = current.A and current.B
    previous_combo = previous.A and previous.B
    if current_combo and not previous_combo:
        return "policy"
    if current.A and not previous.A and not current.B:
        return "init"
    if current.B and not previous.B and not current.A:
        return "zero"
    return None


def resolve_pico_episode_command(
    current: PicoButtonState,
    previous: PicoButtonState,
) -> EpisodeCommand | None:
    """'Y' tap toggles record start/stop; 'Y+A' discards the in-progress episode.

    Deliberately does not use 'X': it already has a live, in-process meaning inside
    ``teleop/pico_retarget_pub.py`` (its own live/pause toggle), so reusing it here
    would fire two unrelated actions in two different processes from one button press.
    'Y' is otherwise unused end-to-end, making it the only fully free bit on this
    message (see ``PicoControllerStateMessage`` in ``sim2real.utils.common``).

    Any chord involving B or X is ignored outright. The teleop bridge engages
    with A+B+X+Y (``g1_bridge.pico.mode_manager``), which satisfies "Y and A":
    without this guard, every single engage would silently discard the episode
    the operator had just started recording -- and the operator would see only a
    dataset that is short by exactly the demos they cared about.
    """
    if current.B or current.X:
        return None
    current_discard = current.Y and current.A
    previous_discard = previous.Y and previous.A
    if current_discard and not previous_discard:
        return "discard"
    if current.Y and not previous.Y and not current.A:
        return "toggle"
    return None
