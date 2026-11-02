"""In-process bridge between tele_humanoid_benchmark's MuJoCo sim tick and
sim2real's ``Tracking``/``BasePolicy`` runtime.

All three seams this needs are now public API in sim2real, so this module is
down to the parts that are genuinely about *this* host:

* ``robot_io="push"`` → ``PushRobotIO``: the host pushes a ``RobotState`` in
  and reads the command back off ``last_command``.
* ``RealtimeMotionBuffer.push_payload()`` / ``reset_stream()``: reference
  motion delivered by direct call. Leave ``motion_zmq_connect`` unset and no
  subscriber thread starts.
* ``controller="passive"``: no operator input, no stdin listener thread; the
  host drives the control mode.

Those replaced, respectively, a locally-defined ``RobotIO``, two subclasses
reaching into ``__append_payload`` through name mangling, and a
``_build_controller`` override — see git history if you need the old shape.

What remains host-specific: ``motion_mode`` (this host can feed the buffer
in-process, or subscribe to the GMR subprocess over ZMQ), and the two
checkpoint-family runtimes below.

Still coupled to one piece of sim2real's internal control flow: ``step()``
only uses policy output while ``state_dict["control_mode"] == "policy"``,
which ``BridgeTracking.__init__`` arranges once via ``set_policy_mode()``.
"""

from __future__ import annotations

import numpy as np
import yaml

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.rl_policy.robot_io import PushRobotIO
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.rl_policy.tracking_factory import (
    DualBranchPolicyMixin,
    SplitOnnxPolicyMixin,
    TorchJitAdapterPolicyMixin,
    policy_family_mixin,
)
from sim2real.rl_policy.utils.motion_buffer import (
    RealtimeMotionBuffer,
    RealtimeSmplMotionBuffer,
)

# Import side effect: registers "sonic.sonic_g1_low_latency_encoder_input"
# into sim2real's Observation registry. Must happen before any policy.yaml
# referencing it is loaded (i.e. before BridgeTracking/SplitOnnxBridgeTracking
# construction below) -- see that module's docstring.
import g1_bridge.sonic_utils  # noqa: F401


#: What ``G1BridgeComposite.run_controller`` drives the runtime through: push a
#: RobotState read from ``sim.data.qpos``/``qvel``, call ``tracking.step()``,
#: read the command back off ``last_command``. Kept as a name of its own because
#: it is this package's half of the contract, and every type hint here says so.
BridgeRobotIO = PushRobotIO


class BridgeTracking(Tracking):
    """``Tracking`` driven by an externally-pushed RobotState (always -- the
    robot's body state always comes from tele's own MuJoCo sim, in-process),
    plus reference-motion fed one of two ways selected by ``motion_mode``:

    * ``"push"`` -- reference motion is pushed synchronously via
      ``motion_buffer.push_payload()`` (Flavor A / SONIC-smpl: tele's own
      PICO sample needs no GMR, so there is no separate process to connect
      to). ``_init_motion_backend`` is overridden to force
      ``motion_zmq_connect=None``, so the buffers start no subscriber thread
      and only ever see what this host pushes.

    * ``"zmq"`` -- reference motion is real full-body GMR-retargeted output,
      which sim2real can only produce from its isolated ``venv/pico/.venv``
      (Python 3.10; ``general_motion_retargeting`` doesn't import under the
      3.11 env this bridge otherwise runs in). That retargeting runs as a
      separate OS process (``sim2real.teleop.pico_retarget_pub`` under that
      venv's interpreter -- see ``gmr_subprocess.py``), publishing over its
      normal ZMQ interface exactly as designed. Here ``_init_motion_backend``
      is left as sim2real's own unmodified implementation, which builds a
      standard ZMQ-subscribing ``RealtimeMotionBuffer``/
      ``RealtimeSmplMotionBuffer`` connecting to that subprocess.
    """

    def __init__(
        self,
        args: "TrackingArgs",
        external_io: BridgeRobotIO,
        motion_mode: str = "push",
    ):
        if motion_mode not in ("push", "zmq"):
            raise ValueError(
                f"motion_mode must be 'push' or 'zmq', got {motion_mode!r}"
            )
        self._motion_mode = motion_mode
        # Latest SONIC encoder latent, set per policy tick by the split-onnx
        # runtime and recorded as `action.motion_token`. Stays None on
        # runtimes that have no separate encoder (nothing to record).
        self.token_state = None
        # robot_io="push"/controller="passive" make BasePolicy.__init__ build
        # exactly what this bridge wants: no socket, no DDS, no stdin listener.
        # "inline" would try to open real G1 DDS hardware and must never be
        # used here. `external_io` is the io we then adopt, so make_bridge_tracking
        # can hand the same object to the composite.
        if args.robot_io != "push":
            raise ValueError(
                "BridgeTracking requires args.robot_io == 'push' — got "
                f"{args.robot_io!r}."
            )
        super().__init__(args)
        try:
            self.robot_io.close()
        except Exception:
            pass
        self.robot_io = external_io
        self.state_processor.robot_io = external_io
        self.action_manager.robot_io = external_io
        # Enter "policy" control mode once, up front. process_controllers()
        # (called every step()) reads self.controller.get_control_mode(),
        # which PassiveController always answers with None, so this is never
        # overwritten from underneath us afterwards.
        self.set_policy_mode(source="g1_bridge")

    def _init_motion_backend(self) -> None:
        if self._motion_mode == "zmq":
            # Real ZMQ-subscribing buffers, unmodified -- connects to the
            # sim2real.teleop.pico_retarget_pub subprocess (gmr_subprocess.py).
            super()._init_motion_backend()
            return
        self._init_motion_backend_push()

    def _init_motion_backend_push(self) -> None:
        self.motion_config = dict(self.policy_config.get("motion", {}))
        self.motion_data = None
        self.motion_future_steps = np.asarray(
            self.motion_config.get("future_steps", []), dtype=int
        )
        if self.motion_future_steps.ndim != 1:
            raise ValueError(
                f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}"
            )
        self.motion_backend = (
            str(self.motion_config.get("motion_backend", "npz")).lower().strip()
        )
        self.motion_config["motion_backend"] = self.motion_backend

        common_kwargs = dict(
            robot_cfg=self.robot_cfg,
            future_steps=self.motion_future_steps,
            motion_zmq_connect=None,  # never start a ZMQ subscriber thread
            motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
            dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
            tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
        )

        if self.motion_backend == "smpl_zmq":
            self.motion_buffer = RealtimeSmplMotionBuffer(**common_kwargs)
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = []
            return

        if self.motion_backend == "zmq":
            self.motion_buffer = RealtimeMotionBuffer(
                **common_kwargs,
                root_body_name=str(self.motion_config.get("root_body_name", "pelvis")),
            )
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = list(self.motion_buffer.body_names)
            return

        raise ValueError(
            f"BridgeTracking(motion_mode='push') only supports motion_backend "
            f"in {{'smpl_zmq', 'zmq'}}; got {self.motion_backend!r} -- 'npz' "
            "offline playback is not meaningful for live PICO teleop."
        )


class SplitOnnxBridgeTracking(SplitOnnxPolicyMixin, BridgeTracking):
    """``BridgeTracking`` for a split encoder/decoder ONNX checkpoint (SONIC).

    The runtime itself is ``SplitOnnxPolicyMixin`` in sim2real; this exists so
    the composite can ask ``isinstance(tracking, SplitOnnxBridgeTracking)``
    when it needs the g1 split-onnx flavor specifically (see
    ``backends/sonic.py``). Selected by ``make_bridge_tracking`` from the
    yaml's ``encoder_model_path``.
    """


class AmoBridgeTracking(TorchJitAdapterPolicyMixin, BridgeTracking):
    """``BridgeTracking`` for AMO's TorchScript checkpoint.

    Runtime lives in sim2real's ``TorchJitAdapterPolicyMixin``. Note what it
    leaves alone: the 8 shoulder/elbow and 6 wrist joints, which
    ``backends/amo.py`` overwrites in ``q_target`` with its PICO-3-point
    IK-driven targets after ``step()`` returns -- mirroring how the original
    ``AMOG1Composite`` built ``pd_target`` in one place. Selected by
    ``make_bridge_tracking`` from the yaml's ``amo_adapter_path``.
    """


class DualBranchBridgeTracking(DualBranchPolicyMixin, BridgeTracking):
    """``BridgeTracking`` for ALMI's two-actor checkpoint.

    Runtime lives in sim2real's ``DualBranchPolicyMixin``. Unlike AMO's split,
    both branches here are neural and together cover all 29 joints, so nothing
    downstream has to patch ``q_target`` afterwards. What ``backends/almi.py``
    does supply is the command stream the observations read
    (``tracking.almi_commands``). Selected by ``make_bridge_tracking`` from the
    yaml's ``upper_model_path``.
    """

    #: The current teleop command, refreshed by the backend before each step.
    #: ``observations/almi.py`` reads it off the policy (which is ``env`` from
    #: an Observation's point of view) and falls back to a neutral stand-still
    #: command while it is None -- so a self-check that never touches the
    #: composite layer still runs.
    almi_commands = None


#: Which BridgeTracking subclass carries each sim2real policy-family runtime.
_BRIDGE_TRACKING_BY_MIXIN = {
    SplitOnnxPolicyMixin: SplitOnnxBridgeTracking,
    TorchJitAdapterPolicyMixin: AmoBridgeTracking,
    DualBranchPolicyMixin: DualBranchBridgeTracking,
}


def make_bridge_tracking(
    policy_config_path: str,
    motion_mode: str = "push",
    **arg_overrides,
) -> tuple[BridgeTracking, BridgeRobotIO]:
    """Construct a (BridgeTracking, BridgeRobotIO) pair for one sim2real
    policy checkpoint. ``arg_overrides`` may override any TrackingArgs field
    (e.g. ``inference_backend="onnx-gpu"``, or ``motion_zmq_connect=...`` when
    ``motion_mode="zmq"`` to point at a non-default GMR subprocess address).

    The checkpoint family comes from sim2real's own ``policy_family_mixin``,
    read off the yaml here because ``BasePolicy.__init__`` (which parses
    policy_config for real) only runs once the class is already chosen."""
    arg_overrides.setdefault("robot_io", "push")
    arg_overrides.setdefault("controller", "passive")
    args = TrackingArgs(policy_config=policy_config_path, **arg_overrides)
    external_io = BridgeRobotIO()
    with open(policy_config_path) as f:
        raw_policy_config = yaml.safe_load(f)
    tracking_cls = _BRIDGE_TRACKING_BY_MIXIN.get(
        policy_family_mixin(raw_policy_config), BridgeTracking
    )
    tracking = tracking_cls(args, external_io, motion_mode=motion_mode)
    return tracking, external_io
