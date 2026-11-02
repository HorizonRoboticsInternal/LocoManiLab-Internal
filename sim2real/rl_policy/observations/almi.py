"""ALMI observations -- mjlab-loco ``Mjlab-Velocity-Flat-Unitree-G1-Dual``.

Port of mjlab's two actor observation groups (``dual_env_cfgs.py``'s
``lower_actor`` / ``lower_actor_history`` / ``upper_actor``) into sim2real's
Observation framework, following the same "shared stateful core, one
Observation subclass per named output" pattern ``amo.py`` and ``scalebfm.py``
already use. The reference implementation this was checked against tick-for-tick
is mjlab-loco's ``deploy_mujoco/sim2sim.py:ObservationBuilder.build``.

Two layouts have shipped and a checkpoint only runs under its own, so each
observation term takes a ``layout`` argument naming which one its policy.yaml
was trained against.

``whole_body`` (the default; ``checkpoints/almi/flat_dual``, mjlab-loco before
5bbaf67 of 2026-08-12 -- what upstream's deploy code calls "legacy")::

    lower_obs (100) = ang_vel(3) | proj_gravity(3) | q-q_default(29) | qdot(29)
                    | prev_action(29) | twist(4) | waist_rpy(3)
    lower_history   = lower_obs over 5 ticks, oldest -> newest, flattened (500)
    upper_obs (144) = ang_vel(3) | proj_gravity(3) | q-q_default(29) | qdot(29)
                    | q_ref(17) | q_ref-q(17) | prev_action(29)
                    | waist_rpy(3) | ee_pos(6) | ee_quat(8)

``half_body`` (mjlab-loco 5bbaf67 onwards -- upstream's "current")::

    lower_obs (46)  = ang_vel(3) | proj_gravity(3) | q-q_default(12) | qdot(12)
                    | prev_action(12) | twist(4)
    lower_history   = lower_obs over 5 ticks, oldest -> newest, flattened (230)
    upper_obs (105) = ang_vel(3) | proj_gravity(3) | q-q_default(17) | qdot(17)
                    | q_ref(17) | q_ref-q(17) | prev_action(17)
                    | ee_pos(6) | ee_quat(8)

The difference is only what each branch OBSERVES; what it controls is 12 legs
and 17 waist/arm joints either way. ``whole_body`` shows both branches all 29
joints and the whole 29-D concatenated action, plus the waist command;
``half_body`` narrows every proprioception block to the branch's own half and
drops ``waist_rpy`` from both, so the lower branch sees nothing about the upper
body except the disturbance it causes, through ang_vel/projected_gravity.

Four things that are easy to get wrong and produce a policy that loads, runs,
and falls over:

* the layout has to match the checkpoint. That much is not silent -- onnxruntime
  rejects the input width -- but it fails at the first policy tick, which on
  hardware is with the operator already in the headset.
* ``half_body`` slices by NAME, through ``lower_joint_names`` /
  ``upper_joint_names``. Take the wrong 12 or the wrong 17 and the widths are
  still 46/105, every graph loads, and nothing raises.
* ``q_ref`` and ``q_ref_error`` are **absolute** upper joint angles, not
  relative to the default pose like the ``q`` block is.
* the history is ordered **oldest -> newest**; its last frame is the current
  ``lower_obs``. Reversed, the robot destabilises immediately.

What this module cannot get from sim2real's own state is the command stream:
ALMI takes commands, not a reference trajectory. ``env.almi_commands`` is an
:class:`AlmiCommands` that ``g1_bridge/backends/almi.py`` refreshes before each
``Tracking.step()`` -- PICO sticks for the twist, GMR whole-body retargeting for
waist/arms. If it is unset (e.g. a self-check that never touches the composite
layer) the neutral command applies: stand still at ``NEUTRAL_BASE_HEIGHT``,
waist level, and the
training-time manual-mode fallback ``q_ref = q_current`` / ``q_ref_error = 0``
(mjlab ``velocity/mdp/observations.py:156-176``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.common import _get_simulation_joint_selection
from sim2real.utils.math import quat_rotate_inverse_numpy

NUM_JOINTS = 29
NUM_UPPER_REF_JOINTS = 17
NUM_LOWER_JOINTS = NUM_JOINTS - NUM_UPPER_REF_JOINTS
HISTORY_STEPS = 5

#: ``layout`` -> (lower_obs width, upper_obs width). See the module docstring.
#: Named for what actually differs -- how much of the robot each branch sees --
#: rather than upstream's "legacy"/"current", which only says which came first
#: and stops being true the next time the layout changes.
LAYOUT_DIMS = {
    "whole_body": (100, 144),
    "half_body": (46, 105),
}
#: What a policy.yaml that does not say gets. Every ALMI checkpoint this repo
#: shipped before 2026-08-27 is whole_body, so an existing manifest keeps
#: working with no edit; a half_body checkpoint has to ask for it.
DEFAULT_LAYOUT = "whole_body"

# Neutral command used until the backend publishes a real one: the robot's own
# standing height, clipped into the trained band [0.35, 0.79] (RobotCfg's
# default_qpos root is 0.80, hence 0.79). Zero would command a crouch into the
# floor on the very first tick, and the old 0.65 was a 15 cm squat that left the
# knee at 1.4 rad -- stable enough to look fine, marginal enough that the
# residual drift varied run to run. g1_bridge/backends/almi.py derives the same
# number from the robot cfg rather than reading this constant.
NEUTRAL_BASE_HEIGHT = 0.79
# The G1's standing pose, hands in the torso frame -- forward kinematics of the
# default joint angles, which is the pose the upper branch can actually hold.
#
# Do NOT substitute deploy_mujoco/reference_sources.py's _DEFAULT_EE_POS /
# _DEFAULT_EE_QUAT here. Those are the slope rig's, and they are wrong for this
# robot by 27 cm in z (they say +0.20, the hands are at -0.068) with an identity
# wrist orientation instead of w=0.917. Feeding them cost 1.5 m of backward
# walking per 20 s in the benchmark sim, because the upper branch chases an EE
# target its joint reference disagrees with.
#
# These are the fallback for a host that never sets `env.almi_commands` at all
# (a self-check). g1_bridge/backends/almi.py does not use them: it FKs the
# running policy's own default_dof_angles, so the neutral tracks the checkpoint
# rather than this constant.
NEUTRAL_EE_POS = np.array(
    [0.101, 0.214, -0.068, 0.101, -0.214, -0.068], dtype=np.float32
)
NEUTRAL_EE_QUAT = np.array(
    [0.917, 0.090, 0.386, 0.046, 0.917, -0.090, 0.386, -0.046], dtype=np.float32
)

# Index of (waist_roll, waist_pitch, waist_yaw) inside the 17-D upper reference,
# whose own order is (waist_yaw, waist_roll, waist_pitch, <14 arm joints>).
WAIST_RPY_IN_UPPER_REF = (1, 2, 0)


@dataclass
class AlmiCommands:
    """One control tick's teleoperation command, already policy-ready.

    ``upper_ref_joint_pos`` is deliberately optional and separate from the EE
    pose: a source that only produces end-effector targets leaves it ``None``
    and the observation falls back to ``q_ref = q_current`` with zero error,
    which is exactly what mjlab does in manual/random EE mode. This checkpoint
    *was* trained with a live joint reference (its ``q_ref_error`` normalizer
    statistics have std 0.05-0.34, far above the 0.0058 a constantly-zero term
    plus its uniform noise would leave), so ``None`` here means degraded
    tracking, not equivalent behaviour.
    """

    twist: np.ndarray  # (4,) vx, vy, wz, base_height
    # (3,) roll, pitch, yaw. Observed by both branches under the `whole_body`
    # layout and by NEITHER under `half_body`, where it survives only because it
    # is still clipped, still recorded into `action.upper_ref`, and still the
    # thing a replay of an older dataset drives. Derived from `q_ref` either way
    # (see validated()), so it cannot disagree with what the arms are tracking.
    waist_rpy: np.ndarray
    ee_pos: np.ndarray  # (6,) left xyz, right xyz -- torso frame
    ee_quat: np.ndarray  # (8,) left wxyz, right wxyz -- torso frame
    upper_ref_joint_pos: np.ndarray | None = None  # (17,) absolute joint angles

    @classmethod
    def neutral(cls) -> "AlmiCommands":
        return cls(
            twist=np.array([0.0, 0.0, 0.0, NEUTRAL_BASE_HEIGHT], dtype=np.float32),
            waist_rpy=np.zeros(3, dtype=np.float32),
            ee_pos=NEUTRAL_EE_POS.copy(),
            ee_quat=NEUTRAL_EE_QUAT.copy(),
            upper_ref_joint_pos=None,
        )

    def validated(self) -> "AlmiCommands":
        """Shape-check, normalize quaternions, and re-derive the waist target.

        A valid 17-D reference owns the waist command: mjlab's
        ``WaistRpyCommand`` syncs from the same motion frame the arm reference
        came from, so publishing the two independently would let the lower
        branch's coordination cue disagree with what the upper branch is
        tracking. Verified in the checkpoint: the ``waist_rpy`` normalizer
        statistics are element-for-element equal to ``q_ref``'s waist entries.
        """
        twist = _vector(self.twist, 4, "twist")
        waist = _vector(self.waist_rpy, 3, "waist_rpy")
        ee_pos = _vector(self.ee_pos, 6, "ee_pos")
        ee_quat = _vector(self.ee_quat, 8, "ee_quat").reshape(2, 4)
        norms = np.linalg.norm(ee_quat, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("ee_quat contains a zero quaternion")
        ee_quat = (ee_quat / norms).reshape(8).astype(np.float32)

        q_ref = None
        if self.upper_ref_joint_pos is not None:
            q_ref = _vector(
                self.upper_ref_joint_pos, NUM_UPPER_REF_JOINTS, "upper_ref_joint_pos"
            )
            waist = q_ref[list(WAIST_RPY_IN_UPPER_REF)].copy()

        return AlmiCommands(
            twist=twist,
            waist_rpy=waist,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            upper_ref_joint_pos=q_ref,
        )


def _vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


class _AlmiCore:
    """Shared per-``env`` state: joint selection, default pose, proprio history.

    Cached once per env regardless of how many named Observation instances
    (``almi_lower_obs``, ``almi_lower_history``, ``almi_upper_obs``) reference
    it, so the 5-deep history advances exactly once per control tick.
    """

    def __init__(
        self,
        env,
        *,
        joint_names: Sequence[str],
        upper_joint_names: Sequence[str],
        lower_joint_names: Sequence[str],
        history_steps: int = HISTORY_STEPS,
        layout: str = DEFAULT_LAYOUT,
    ):
        self.env = env
        self.layout = str(layout)
        if self.layout not in LAYOUT_DIMS:
            raise ValueError(
                f"unknown ALMI observation layout {self.layout!r}; expected one "
                f"of {sorted(LAYOUT_DIMS)}"
            )
        self.lower_dim, self.upper_dim = LAYOUT_DIMS[self.layout]
        self.joint_names = tuple(str(name) for name in joint_names)
        self.upper_joint_names = tuple(str(name) for name in upper_joint_names)
        self.lower_joint_names = tuple(str(name) for name in lower_joint_names)
        if len(self.joint_names) != NUM_JOINTS:
            raise ValueError(
                f"ALMI expects exactly {NUM_JOINTS} joint_names (mjlab's compiled "
                f"order), got {len(self.joint_names)}"
            )
        if len(self.upper_joint_names) != NUM_UPPER_REF_JOINTS:
            raise ValueError(
                f"ALMI expects exactly {NUM_UPPER_REF_JOINTS} upper_joint_names, "
                f"got {len(self.upper_joint_names)}"
            )
        if len(self.lower_joint_names) != NUM_LOWER_JOINTS:
            raise ValueError(
                f"ALMI expects exactly {NUM_LOWER_JOINTS} lower_joint_names, "
                f"got {len(self.lower_joint_names)}"
            )
        missing = [
            n
            for n in self.upper_joint_names + self.lower_joint_names
            if n not in self.joint_names
        ]
        if missing:
            raise ValueError(f"branch joint names not present in joint_names: {missing}")
        # The two branches are joint-DISJOINT -- lower actuates 12, upper 17,
        # together exactly the 29. An overlap means the split was mis-declared,
        # which under `half_body` produces obs of the right width built from the
        # wrong joints: the one failure mode in this module that nothing
        # downstream can see.
        overlap = sorted(set(self.lower_joint_names) & set(self.upper_joint_names))
        if overlap:
            raise ValueError(
                f"lower_joint_names and upper_joint_names overlap on {overlap}; "
                "the two ALMI branches must partition the 29 joints"
            )

        self.history_steps = int(history_steps)
        self._joint_ids: list[int] | None = None
        self._upper_ids_in_selection: list[int] | None = None
        self._lower_ids_in_selection: list[int] | None = None
        self._default_joint_pos: np.ndarray | None = None
        self._gravity = np.array([0.0, 0.0, -1.0])

        self._history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self._lower = np.zeros(self.lower_dim, dtype=np.float32)
        self._upper = np.zeros(self.upper_dim, dtype=np.float32)
        self._last_update_token: tuple[int, int] | None = None
        self.reset()

    def reset(self) -> None:
        # Cleared, not zero-filled: the next update() backfills all five slots
        # with the first real frame, so a fresh episode never feeds the policy a
        # history of zeros that never occurred.
        self._history.clear()
        self._last_update_token = None

    def _refresh_indices(self) -> None:
        if self._joint_ids is not None:
            return
        self._joint_ids, ordered = _get_simulation_joint_selection(
            self.env, list(self.joint_names)
        )
        if tuple(ordered) != self.joint_names:
            raise ValueError(
                "ALMI joint order does not survive resolution against "
                "joint_names_simulation.\n"
                f"  expected={self.joint_names}\n  actual={tuple(ordered)}"
            )
        self._upper_ids_in_selection = [
            self.joint_names.index(name) for name in self.upper_joint_names
        ]
        self._lower_ids_in_selection = [
            self.joint_names.index(name) for name in self.lower_joint_names
        ]
        # default_dof_angles is laid out in action_manager.joint_names order,
        # which is the robot cfg's, not necessarily this observation's. Map by
        # name rather than assuming the two coincide -- they do today, and a
        # positional index would keep "working" on the day they stop.
        by_name = dict(
            zip(self.env.action_manager.joint_names, self.env.default_dof_angles)
        )
        self._default_joint_pos = np.array(
            [by_name[name] for name in self.joint_names], dtype=np.float32
        )

    def _commands(self) -> AlmiCommands:
        commands = getattr(self.env, "almi_commands", None)
        if commands is None:
            commands = AlmiCommands.neutral()
        return commands.validated()

    def update_once(self, data: dict[str, Any]) -> None:
        token = (id(data), int(getattr(self.env, "total_inference_cnt", -1)))
        if self._last_update_token == token:
            return
        self._last_update_token = token

        self._refresh_indices()
        assert self._joint_ids is not None
        assert self._upper_ids_in_selection is not None
        assert self._lower_ids_in_selection is not None
        assert self._default_joint_pos is not None

        state = self.env.state_processor
        ang_vel = np.asarray(state.root_ang_vel_b, dtype=np.float32).reshape(3)
        projected_gravity = quat_rotate_inverse_numpy(
            np.asarray(state.root_quat_w).reshape(1, 4),
            self._gravity.reshape(1, 3),
        ).reshape(3).astype(np.float32)

        joint_pos = np.asarray(state.joint_pos, dtype=np.float32)[self._joint_ids]
        joint_vel = np.asarray(state.joint_vel, dtype=np.float32)[self._joint_ids]
        joint_pos_rel = joint_pos - self._default_joint_pos
        # data["action"] is written after inference, so at update() time this is
        # the previous tick's raw 29-D network output -- which is what mjlab's
        # last_action term holds.
        prev_action = np.asarray(data["action"], dtype=np.float32).reshape(-1)
        if prev_action.shape != (NUM_JOINTS,):
            raise ValueError(
                f"ALMI expects a {NUM_JOINTS}-D previous action, got {prev_action.shape}"
            )

        commands = self._commands()

        half = self.layout == "half_body"
        lower_ids = self._lower_ids_in_selection
        upper_ids = self._upper_ids_in_selection

        # half_body: the branch's own 12 joints and its own 12 action entries,
        # and no waist command at all. whole_body: all 29 of each, plus the
        # waist command the lower branch used as its coordination cue.
        if half:
            lower_proprio = (
                joint_pos_rel[lower_ids],
                joint_vel[lower_ids],
                prev_action[lower_ids],
            )
            lower_command = (commands.twist,)
        else:
            lower_proprio = (joint_pos_rel, joint_vel, prev_action)
            lower_command = (commands.twist, commands.waist_rpy)
        self._lower = np.concatenate(
            (ang_vel, projected_gravity) + lower_proprio + lower_command
        ).astype(np.float32)
        if self._lower.shape != (self.lower_dim,):
            raise RuntimeError(
                f"Internal lower obs shape is {self._lower.shape}, expected "
                f"({self.lower_dim},) for layout {self.layout!r}"
            )
        _append_with_backfill(self._history, self._lower)

        current_upper = joint_pos[self._upper_ids_in_selection]
        if commands.upper_ref_joint_pos is not None:
            q_ref = commands.upper_ref_joint_pos
            q_ref_error = q_ref - current_upper
        else:
            q_ref = current_upper
            q_ref_error = np.zeros_like(current_upper)

        if half:
            upper_proprio = (joint_pos_rel[upper_ids], joint_vel[upper_ids])
            upper_prev_action = prev_action[upper_ids]
            upper_waist = ()
        else:
            upper_proprio = (joint_pos_rel, joint_vel)
            upper_prev_action = prev_action
            upper_waist = (commands.waist_rpy,)
        self._upper = np.concatenate(
            (ang_vel, projected_gravity)
            + upper_proprio
            + (q_ref, q_ref_error, upper_prev_action)
            + upper_waist
            + (commands.ee_pos, commands.ee_quat)
        ).astype(np.float32)
        if self._upper.shape != (self.upper_dim,):
            raise RuntimeError(
                f"Internal upper obs shape is {self._upper.shape}, expected "
                f"({self.upper_dim},) for layout {self.layout!r}"
            )

    def output(self, name: str) -> np.ndarray:
        if name == "lower_obs":
            return self._lower
        if name == "upper_obs":
            return self._upper
        if name == "lower_history":
            # oldest -> newest, flattened. The ONNX graph takes it flat because
            # ONNXModule reshapes every input to (1, -1) anyway.
            return np.stack(self._history, axis=0).reshape(-1).astype(np.float32)
        raise KeyError(f"Unknown ALMI observation output {name!r}")


def _append_with_backfill(history: deque, value: np.ndarray) -> None:
    """Append, filling an empty buffer with copies of the first frame."""
    value = np.asarray(value, dtype=np.float32).copy()
    if not history:
        history.extend(value.copy() for _ in range(history.maxlen or 1))
    else:
        history.append(value)


def _get_core(env, **kwargs) -> _AlmiCore:
    cache = getattr(env, "_almi_core_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_almi_core_cache", cache)
    key = (
        tuple(kwargs["joint_names"]),
        tuple(kwargs["upper_joint_names"]),
        tuple(kwargs["lower_joint_names"]),
        int(kwargs["history_steps"]),
        str(kwargs["layout"]),
    )
    # One ALMI policy per env, so one core: the three observation terms are
    # three named outputs OF THE SAME vector-building step, not three
    # independent observations. A second key means the terms were configured
    # inconsistently -- `layout` set on upper_obs but not on lower_history is
    # the easy one -- and the result would be a history built to a different
    # layout than the observation it is supposed to be a window onto. Both
    # widths would still be individually correct, and nothing further down could
    # notice.
    if cache and key not in cache:
        (existing,) = cache
        differing = [
            name
            for name, a, b in zip(
                ("joint_names", "upper_joint_names", "lower_joint_names",
                 "history_steps", "layout"),
                existing,
                key,
            )
            if a != b
        ]
        raise ValueError(
            "ALMI observation terms disagree on "
            f"{', '.join(differing)} -- every almi_* term in one policy.yaml "
            "must be given the same arguments (they share one observation "
            "core). Existing: "
            + repr(dict(zip(("joint_names", "upper_joint_names",
                             "lower_joint_names", "history_steps", "layout"),
                            existing)))
        )
    if key not in cache:
        cache[key] = _AlmiCore(env, **kwargs)
    return cache[key]


class _AlmiObservation(Observation, namespace="almi"):
    output_name: str

    def __init__(
        self,
        joint_names: Sequence[str],
        upper_joint_names: Sequence[str] | None = None,
        lower_joint_names: Sequence[str] | None = None,
        history_steps: int = HISTORY_STEPS,
        layout: str = DEFAULT_LAYOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Only the upper branch names its 17 reference joints; the other two
        # groups share the same core, so derive the default from the full list
        # rather than making every group repeat it in the yaml. mjlab's compiled
        # order is the 12 legs followed by the 17 waist/arm joints, so each
        # branch's half is one end of that list -- but either can still be given
        # explicitly, and then _AlmiCore checks it against the resolved order
        # instead of trusting the convention.
        if upper_joint_names is None:
            upper_joint_names = list(joint_names)[NUM_JOINTS - NUM_UPPER_REF_JOINTS :]
        if lower_joint_names is None:
            lower_joint_names = list(joint_names)[:NUM_LOWER_JOINTS]
        self.core = _get_core(
            self.env,
            joint_names=list(joint_names),
            upper_joint_names=list(upper_joint_names),
            lower_joint_names=list(lower_joint_names),
            history_steps=history_steps,
            layout=layout,
        )

    def reset(self) -> None:
        self.core.reset()

    def update(self, data: dict[str, Any]) -> None:
        self.core.update_once(data)

    def compute(self) -> np.ndarray:
        return self.core.output(self.output_name)


class almi_lower_obs(_AlmiObservation, namespace="almi"):
    output_name = "lower_obs"


class almi_lower_history(_AlmiObservation, namespace="almi"):
    output_name = "lower_history"


class almi_upper_obs(_AlmiObservation, namespace="almi"):
    output_name = "upper_obs"
