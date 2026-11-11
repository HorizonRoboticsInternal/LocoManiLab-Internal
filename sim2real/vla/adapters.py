"""Turning a VLA's action into something a whole-body policy consumes.

A policy family does not consume "a VLA action" -- it consumes one specific
command space (see ``sim2real.rl_policy.families``). And which command space a
VLA emits was fixed at data-collection time, because the recorded action IS
that command space.

So the cross-product never materialises. There are as many adapters as there
are command spaces, and any VLA emitting ``ref_qpos`` drives every
reference-tracking family without another line of code.

Each adapter declares the action keys it needs. Check them against the
server's actual response once, at startup (:func:`check_action_keys`), rather
than discovering the mismatch as a robot that stands still: a missing key
usually means the checkpoint was trained for a different family.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

import numpy as np

from sim2real.config.robots.base import (
    JOINT_NAMES_KEY,
    MOTION_FIRST_FRAME_KEY,
    PUBLISH_T_NS_KEY,
)
from sim2real.vla.client import action_field, as_chunk


class ActionSink(Protocol):
    """What an adapter needs from the runtime it is driving.

    A ``Tracking`` satisfies this. So does a test double, which is the point.
    """

    motion_buffer: Any


def _buffer_now_ns(sink: "ActionSink") -> int:
    """The timestamp to stamp a pushed reference frame with.

    The buffer samples at ITS clock (sim time under a sim host, wall time on
    hardware -- see motion_buffer._default_clock_ns), so a frame the adapter
    stamps must be on that same clock, not wall time. Reads the buffer's clock
    when it exposes one; falls back to wall time (the buffer's own default) so a
    plain test double still works.
    """
    buf = getattr(sink, "motion_buffer", None)
    clock = getattr(buf, "_clock_ns", None)
    return int(clock()) if callable(clock) else time.time_ns()


_REGISTRY: dict[str, type["ActionAdapter"]] = {}


def register(cls: type["ActionAdapter"]) -> type["ActionAdapter"]:
    _REGISTRY[cls.command_space] = cls
    return cls


def get_adapter(command_space: str, **kwargs) -> "ActionAdapter":
    try:
        cls = _REGISTRY[command_space]
    except KeyError:
        raise ValueError(
            f"no adapter for command space {command_space!r}; "
            f"registered: {sorted(_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def command_spaces() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def _action_keys(action) -> list:
    """Every key an action dict exposes, for diagnostics (flat keys plus the
    group names under a nested 'action' mapping, if any)."""
    keys = set()
    try:
        keys.update(str(k) for k in action.keys())
        inner = action.get("action")
        if isinstance(inner, dict):
            keys.update(str(k) for k in inner.keys())
    except Exception:
        pass
    return sorted(keys)


def check_action_keys(command_space: str, action: dict) -> None:
    """Raise unless ``action`` carries everything the adapter needs.

    Call once on the first response. The error names both sides because the
    fix is almost always "this checkpoint belongs to a different family", and
    seeing the two key sets side by side is what makes that obvious.
    """
    cls = _REGISTRY[command_space]
    present = {k.replace("action.", "", 1) for k in action}
    missing = [k for k in cls.required_keys if k not in present]
    if missing:
        raise KeyError(
            f"policy action is missing {missing} required by command space "
            f"{command_space!r}. Policy returned: {sorted(present)}. "
            "A checkpoint trained on a different controller's data will look "
            "exactly like this."
        )


class ActionAdapter:
    """Applies one tick's action to the runtime."""

    #: The command space this adapter implements.
    command_space: str = ""
    #: Action keys that must be present, prefixed or not.
    required_keys: tuple[str, ...] = ()

    def reset(self) -> None:
        """Forget per-episode state."""

    def apply(self, sink: ActionSink, action: dict) -> None:
        """Apply ONE timestep's action (not a chunk -- the driver indexes)."""
        raise NotImplementedError


@register
class RefQposAdapter(ActionAdapter):
    """50 Hz reference pose -> the reference-motion buffer.

    The portable one: every reference-tracking family in
    ``families_for_command_space("ref_qpos")`` accepts it unchanged, so a
    policy trained here is not married to the controller it was collected on.

    ``motion_first_frame`` is stamped on the opening push of each episode --
    and ONLY there. The recorded ``ref_root_pos``/``ref_root_quat`` (schema v8)
    are the segment-transformed, episode-relative root the policy tracked, so a
    predicted trajectory needs exactly one re-attachment per episode to land on
    the robot; a per-chunk stamp would re-anchor every chunk onto the held
    frame and turn the model's progress along the trajectory into a teleport.
    Without it the buffer treats the new episode's first reference as a
    continuation of the last one and carries a stale segment transform into a
    scene the robot was just teleported into.
    """

    command_space = "ref_qpos"
    required_keys = ("ref_root_pos", "ref_root_quat", "ref_joint_pos")

    def __init__(self, frame_builder: Callable[[np.ndarray], dict] | None = None,
                 joint_names: list[str] | None = None) -> None:
        #: ``qpos -> payload`` forward kinematics. Required, and it must be the
        #: SAME FK the host's live reference stream uses: the buffer consumes
        #: body poses, not joint angles, so a second FK here would put the
        #: policy's reference in a subtly different frame from the one the
        #: training data was recorded against.
        self.frame_builder = frame_builder
        self.joint_names = joint_names
        self._first = True

    def reset(self) -> None:
        self._first = True

    def apply(self, sink: ActionSink, action: dict) -> None:
        if self.frame_builder is None:
            raise RuntimeError(
                "RefQposAdapter needs a frame_builder: the motion buffer "
                "consumes body poses, so the predicted qpos has to be run "
                "through the host's own forward kinematics."
            )
        root_pos = np.asarray(action_field(action, "ref_root_pos"), np.float32).reshape(-1)
        root_quat = np.asarray(action_field(action, "ref_root_quat"), np.float32).reshape(-1)
        joint_pos = np.asarray(action_field(action, "ref_joint_pos"), np.float32).reshape(-1)
        payload = dict(self.frame_builder(np.concatenate([root_pos, root_quat, joint_pos])))
        payload.setdefault(PUBLISH_T_NS_KEY, _buffer_now_ns(sink))
        names = self.joint_names or list(
            getattr(sink.motion_buffer, "joint_names", []) or []
        )
        if names:
            payload.setdefault(JOINT_NAMES_KEY, list(names))
        if self._first:
            payload[MOTION_FIRST_FRAME_KEY] = True
            self._first = False
        sink.motion_buffer.push_payload(payload)


@register
class SmplAdapter(ActionAdapter):
    """SMPL joint positions -> the SMPL reference buffer (SONIC's encoder).

    The policy predicts the encoder's *input*, so the encoder still runs and
    produces the latent itself. Contrast :class:`MotionTokenAdapter`, which
    predicts the latent and skips the encoder.
    """

    command_space = "smpl"
    required_keys = ("smpl_joints",)

    def __init__(self, payload_builder: Callable[[np.ndarray], dict] | None = None) -> None:
        #: How to turn predicted SMPL joints into a buffer payload. The full
        #: conversion needs a retargeting config, so the host supplies it;
        #: without one this adapter only forwards an already-built payload.
        self.payload_builder = payload_builder
        self._first = True

    def reset(self) -> None:
        self._first = True

    def apply(self, sink: ActionSink, action: dict) -> None:
        smpl = np.asarray(action_field(action, "smpl_joints"), dtype=np.float32)
        if self.payload_builder is None:
            raise RuntimeError(
                "SmplAdapter needs a payload_builder: turning predicted SMPL "
                "joints into a motion payload requires the host's retargeting "
                "config (see sim2real/teleop/smpl_stream.py)."
            )
        payload = dict(self.payload_builder(smpl))
        payload.setdefault(PUBLISH_T_NS_KEY, _buffer_now_ns(sink))
        if self._first:
            payload[MOTION_FIRST_FRAME_KEY] = True
            self._first = False
        sink.motion_buffer.push_payload(payload)


@register
class MotionTokenAdapter(ActionAdapter):
    """SONIC's 64-D latent -> the decoder, skipping the encoder.

    SONIC-only by construction: the token is that checkpoint's own latent
    space, and nothing else can read it. Requires a runtime carrying
    ``decode_tokens`` -- i.e. ``SplitOnnxPolicyMixin``.
    """

    command_space = "motion_token"
    required_keys = ("motion_token",)

    #: Recorded tokens sit well inside this; anything beyond it is the model
    #: extrapolating, and the decoder's output stops being trustworthy.
    TOKEN_ABS_MAX = 1.25

    def __init__(self, dim: int = 64, warn_out_of_range: bool = True) -> None:
        self.dim = dim
        self.warn_out_of_range = warn_out_of_range
        self.last_token: np.ndarray | None = None

    def reset(self) -> None:
        self.last_token = None

    def apply(self, sink: ActionSink, action: dict) -> None:
        token = as_chunk(action_field(action, "motion_token"), self.dim)[0]
        if self.warn_out_of_range:
            peak = float(np.abs(token).max())
            if peak > self.TOKEN_ABS_MAX:
                from loguru import logger

                logger.warning(
                    "motion_token |max| {:.3f} > {:.2f}: out of the training "
                    "range, decoder output is unreliable",
                    peak,
                    self.TOKEN_ABS_MAX,
                )
        self.last_token = token
        setter = getattr(sink, "set_motion_token", None)
        if setter is None:
            raise RuntimeError(
                "motion_token needs a runtime that can decode it "
                "(SplitOnnxPolicyMixin, or a host exposing set_motion_token); "
                f"{type(sink).__name__} cannot."
            )
        setter(token)


@register
class VelocityCmdAdapter(ActionAdapter):
    """Torso velocity + limb targets (AMO).

    Command-driven, not motion-tracking: nothing goes to the motion buffer.
    The host owns how these reach the policy's observation, so this only
    normalises and hands them over.
    """

    command_space = "velocity_cmd"
    required_keys = ("torso_vx", "torso_vy", "torso_vyaw")

    def __init__(self, optional_keys: tuple[str, ...] = (
        "target_yaw", "rpy", "height",
        "left_arm", "right_arm",
        # hands: schema v5 renamed the groups to *_hand_joints; both spellings
        # are accepted (the older one for pre-v5 checkpoints).
        "left_hand_joints", "right_hand_joints", "left_hand", "right_hand",
    )) -> None:
        self.optional_keys = optional_keys
        self._reported_missing: set = set()

    def apply(self, sink: ActionSink, action: dict) -> None:
        cmd = {
            k: np.asarray(action_field(action, k), dtype=np.float32).reshape(-1)
            for k in self.required_keys
        }
        missing = []
        for k in self.optional_keys:
            try:
                cmd[k] = np.asarray(action_field(action, k), dtype=np.float32).reshape(-1)
            except KeyError:
                missing.append(k)
        # Optional means "a rate-only / arm-less checkpoint may not predict it";
        # it does not mean silently. A key the checkpoint DOES predict under a
        # name this list does not carry would be dropped here without a trace
        # -- and for target_yaw/height/rpy that is a robot that walks but never
        # turns to a heading, leans or crouches. Say so, once per key.
        new = [k for k in missing if k not in self._reported_missing]
        if new:
            self._reported_missing.update(new)
            print(
                f"[vla] VelocityCmdAdapter: action has no {', '.join(new)} "
                f"(got: {', '.join(_action_keys(action))}); those "
                "channels fall back to the controller's own integrators/defaults."
            )
        setter = getattr(sink, "set_velocity_command", None)
        if setter is None:
            raise RuntimeError(
                "velocity_cmd needs a host exposing set_velocity_command; "
                f"{type(sink).__name__} does not."
            )
        setter(cmd)


@register
class TwistEeAdapter(ActionAdapter):
    """ALMI's command, split by body half: a twist for the legs, a pose
    reference for the waist and arms.

    Command-driven like :class:`VelocityCmdAdapter` -- nothing reaches the
    motion buffer, because ALMI consumes commands rather than a reference
    trajectory. ALMI-only by construction: the split is what that checkpoint
    was trained on, and no other family here takes a joint reference for half
    its body and a velocity command for the other half.

    The keys are the recorded action groups (``g1_bridge.modality``'s
    ``ALMI_MODALITY_BLOBS``) one for one, which is not a convention but the
    definition: what teleop recorded IS what a policy replacing the operator
    has to emit.
    """

    command_space = "twist_ee"
    required_keys = (
        "lin_vel",
        "ang_vel_z",
        "base_height",
        "left_ee_pos",
        "right_ee_pos",
        "left_ee_quat",
        "right_ee_quat",
        "ref_joint_pos",
    )

    #: Recorded, deliberately NOT consumed. ``AlmiCommands.validated()``
    #: re-derives the waist command out of ``ref_joint_pos``'s waist entries,
    #: exactly as mjlab's ``WaistRpyCommand`` syncs off the same motion frame at
    #: training time -- so the two columns always agreed in the data, and taking
    #: the predicted copy here would introduce a disagreement that never existed.
    #: It is not in ``required_keys`` either: a model that omits a column it
    #: could only have learned as a function of another one is not broken.
    derived_keys = ("waist_rpy",)

    def apply(self, sink: ActionSink, action: dict) -> None:
        cmd = {
            k: np.asarray(action_field(action, k), dtype=np.float32).reshape(-1)
            for k in self.required_keys
        }
        setter = getattr(sink, "set_twist_ee_command", None)
        if setter is None:
            raise RuntimeError(
                "twist_ee needs a host exposing set_twist_ee_command; "
                f"{type(sink).__name__} does not."
            )
        setter(cmd)


@register
class HolomotionReferenceAdapter(ActionAdapter):
    """A controller reference window, assembled by VlaDriver before handoff."""
    command_space = "holomotion_reference_v1"
    required_keys = ("reference",)

    def apply(self, sink, action):
        raise ValueError("HoloMotion windows must be delivered through the host backend")
