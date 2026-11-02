"""Checkpoint families whose runtime is not one fused ``policy.onnx``.

``BasePolicy.setup_policy`` builds exactly one inference module from
``model_path``, which covers most of ``checkpoints/`` -- but not all of it.
Three families shipped in this repo need more:

* **split encoder/decoder ONNX** (``checkpoints/sonic/**``): two graphs, with
  the encoder's output concatenated onto a second observation group to form
  the decoder's input. Declared by ``encoder_model_path`` in policy.yaml.
* **TorchScript + adapter** (``checkpoints/amo``): a ``(obs, extra) -> action``
  TorchScript module rather than ONNX, plus its own action post-processing.
  Declared by ``amo_adapter_path``.
* **dual branch ONNX** (``checkpoints/almi/**``): two independent, joint-disjoint
  actors run in parallel and their outputs concatenate into one action.
  Declared by ``upper_model_path``.

All three are expressed as *mixins* overriding only ``setup_policy``, so they
compose with ``Tracking`` (the normal entrypoint) and equally with a host's
own ``Tracking`` subclass -- pass it as ``base=``. ``make_tracking`` picks the
right mixin by reading the yaml, since the class must be chosen before
``BasePolicy.__init__`` parses the config for real.
"""

from __future__ import annotations

import pathlib
from typing import Type

import numpy as np
import yaml
from loguru import logger

from sim2real.rl_policy.inference import build_inference_module
from sim2real.rl_policy.tracking import Tracking, TrackingArgs


class SplitOnnxPolicyMixin:
    """Encoder + decoder ONNX pair instead of one fused ``policy.onnx``.

    Wiring, matching gear_sonic_deploy's own C++ encoder/decoder split
    (``src/g1/g1_deploy_onnx_ref/include/encoder.hpp`` + its
    ``observation_config.yaml``):

    * encoder: obs_dict=<``g1_input`` obs group> -> ``encoded_tokens``
    * decoder: obs_dict=concat([encoded_tokens, <``proprioception`` group>]) -> action

    So ``g1_input`` must already BE the encoder's full flat input, and
    ``proprioception`` the decoder's input minus that token prefix. The
    ``encoder_model_path`` in policy.yaml is resolved relative to the yaml's
    own directory (absolute paths are used as-is), same as ``model_path``.
    """

    #: Token supplied from outside for the next tick, bypassing the encoder.
    #: Set by :meth:`set_motion_token`; consumed and cleared by the policy.
    _external_token = None

    #: The latent that actually drove the decoder on the last tick, whichever
    #: way it got there -- encoded here, or handed in from outside. Exposed for
    #: recording (it is the ``motion_token`` action field a dataset stores) and
    #: for diagnostics. None before the first tick.
    token_state = None

    def set_motion_token(self, token) -> None:
        """Use ``token`` instead of running the encoder, for the next tick.

        The encoder exists to turn a reference into a latent. A policy that
        predicts the latent directly has already done that job, so running the
        encoder anyway would overwrite its answer with one derived from
        whatever stale reference the buffer happens to hold.

        Set it every tick you want it honoured: it is consumed on use, so a
        producer that stops publishing falls back to the encoder rather than
        freezing on its last token.
        """
        self._external_token = None if token is None else np.asarray(
            token, dtype=np.float32
        ).reshape(-1)

    #: None until the first encoded tick; then the width we already validated,
    #: so the check below costs one int comparison per tick.
    _checked_encoder_width = None

    def _check_encoder_width(self, width: int, encoder_path: str) -> None:
        """Fail with both numbers when the ``g1_input`` group and the encoder
        graph disagree on the input width.

        Each sonic variant has its own encoder width (release 1762,
        low_latency 1247, v1_1 1751) while sharing every other name and shape,
        so a manifest that names the wrong observation class for its weights is
        an easy mistake and the native symptom is an ORT shape error naming
        neither the manifest nor the class.
        """
        if width == self._checked_encoder_width:
            return
        # ONNXModule/TensorRTModule expose input_shapes positionally; these
        # encoders declare exactly one input, so anything else means a graph
        # this check was not written for -- skip rather than guess.
        shapes = getattr(self._encoder_module, "input_shapes", None) or []
        expected = shapes[0][-1] if len(shapes) == 1 and len(shapes[0]) else None
        if isinstance(expected, (int, np.integer)) and int(expected) != width:
            raise ValueError(
                f"g1_input is {width}D but {encoder_path} expects {expected}D. "
                f"The 'observation.g1_input' entry in {self.args.policy_config} "
                "names an observation class built for a different SONIC "
                "encoder variant."
            )
        self._checked_encoder_width = width

    def setup_policy(self, model_path: str) -> None:
        encoder_path = self.policy_config.get("encoder_model_path")
        if not encoder_path:
            raise ValueError(
                f"{type(self).__name__} requires policy.yaml to set "
                "'encoder_model_path' (absolute, or relative to the yaml's "
                "own directory)."
            )
        encoder_path = str(
            pathlib.Path(self.args.policy_config).parent / str(encoder_path)
        )

        self._encoder_module = build_inference_module(
            encoder_path, self.inference_backend
        )
        self._decoder_module = build_inference_module(
            model_path, self.inference_backend
        )
        logger.info(
            "{} loaded encoder={} decoder={}",
            type(self).__name__,
            encoder_path,
            model_path,
        )

        def policy(input_dict):
            proprioception = np.asarray(
                input_dict["proprioception"], dtype=np.float32
            ).reshape(-1)

            if self._external_token is not None:
                token_state, self._external_token = self._external_token, None
            else:
                g1_input = np.asarray(
                    input_dict["g1_input"], dtype=np.float32
                ).reshape(-1)
                self._check_encoder_width(g1_input.shape[0], encoder_path)
                # ONNXModule requires the literal (batch=1, dim) shape each
                # graph declares -- both of these have a fixed, not symbolic,
                # batch dim -- so add it back on the way in; ORT outputs get
                # theirs auto-squeezed on the way out.
                encoder_out = self._encoder_module({"obs_dict": g1_input[None, :]})
                token_state = np.asarray(
                    encoder_out["encoded_tokens"], dtype=np.float32
                ).reshape(-1)
            # Record whichever latent is about to drive the decoder -- an
            # externally supplied one is no less "what the robot did" than an
            # encoded one, and a recording that only captured the encoder would
            # be blank for exactly the runs driven by a policy.
            self.token_state = token_state

            decoder_in = np.concatenate([token_state, proprioception])
            decoder_out = self._decoder_module({"obs_dict": decoder_in[None, :]})
            action = np.asarray(decoder_out["action"], dtype=np.float32).reshape(-1)

            q_target = self.default_dof_angles.copy()
            q_target[self.controlled_joint_indices] += action * self.action_scale
            return action, q_target, input_dict

        self.policy = policy

    def encode_tokens(self, g1_input: np.ndarray) -> np.ndarray:
        """Run the encoder alone: full encoder input -> latent token.

        Exposed because the token is a boundary worth standing on -- anything
        that produces tokens by another route (a policy predicting them
        directly) needs the decoder without this, and anything comparing
        against recorded tokens needs this without the decoder.
        """
        g1_input = np.asarray(g1_input, dtype=np.float32).reshape(-1)
        out = self._encoder_module({"obs_dict": g1_input[None, :]})
        return np.asarray(out["encoded_tokens"], dtype=np.float32).reshape(-1)

    def decode_tokens(self, token_state: np.ndarray, proprioception: np.ndarray):
        """Run the decoder alone: (token, proprioception) -> (action, q_target).

        The entry point for a latent-token policy: whoever supplies the token
        replaces the encoder, and everything downstream stays identical.
        """
        token_state = np.asarray(token_state, dtype=np.float32).reshape(-1)
        proprioception = np.asarray(proprioception, dtype=np.float32).reshape(-1)
        decoder_in = np.concatenate([token_state, proprioception])
        out = self._decoder_module({"obs_dict": decoder_in[None, :]})
        action = np.asarray(out["action"], dtype=np.float32).reshape(-1)
        q_target = self.default_dof_angles.copy()
        q_target[self.controlled_joint_indices] += action * self.action_scale
        return action, q_target


class TorchJitAdapterPolicyMixin:
    """AMO (Li/Cheng/Huang/Wang, RSS 2025): TorchScript main policy.

    The obs side -- history buffers, gait phase, the adapter sub-model -- is
    ordinary configuration, in ``observations/amo.py``. This only wires the
    *main* policy (``amo_jit.pt``, a 2-input ``(obs, extra) -> action``
    TorchScript module) and its post-inference math: action scaling, the
    default-pose offset, and the 23-dim ``last_action`` fed back into the next
    tick's obs.

    Note what it does NOT set: the 8 shoulder/elbow and 6 wrist joints. AMO's
    network controls 15 joints (legs + waist); the arms are driven externally
    (IK from an operator, or a policy predicting arm targets) and whoever does
    that overwrites those slots in ``q_target`` after ``step()`` returns. The
    pseudo-action fed back for them keeps the obs history consistent meanwhile.
    """

    _AMO_JOINT_NAMES = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    )
    #: Not read from policy.yaml's action_scale: this family computes q_target
    #: itself and the yaml value does not apply to it.
    _ACTION_SCALE = 0.25
    #: The 15 leading entries of _AMO_JOINT_NAMES the network actually drives.
    _NUM_POLICY_JOINTS = 15

    def setup_policy(self, model_path: str) -> None:
        # TorchScript, not ONNX -- always torch-jit regardless of
        # args.inference_backend (which defaults to "onnx-cpu" and has no
        # meaning for this checkpoint family).
        self._amo_module = build_inference_module(
            model_path,
            "torch-jit",
            input_keys=("obs", "extra"),
            output_keys=("action",),
        )
        sim_names = list(self.state_processor.joint_names)
        missing = [n for n in self._AMO_JOINT_NAMES if n not in sim_names]
        if missing:
            raise ValueError(f"{type(self).__name__}: simulation joints missing {missing}")
        self._amo_sim_indices = np.asarray(
            [sim_names.index(n) for n in self._AMO_JOINT_NAMES], dtype=int
        )
        logger.info("{} loaded main policy={}", type(self).__name__, model_path)

        n = self._NUM_POLICY_JOINTS

        def policy(input_dict):
            obs = np.asarray(input_dict["obs"], dtype=np.float32).reshape(-1)
            extra = np.asarray(input_dict["extra"], dtype=np.float32).reshape(-1)
            raw = np.asarray(
                self._amo_module({"obs": obs, "extra": extra})["action"],
                dtype=np.float32,
            ).reshape(-1)
            raw = np.clip(raw, -40.0, 40.0)

            dof_pos_amo = np.asarray(self.state_processor.joint_pos, dtype=np.float32)[
                self._amo_sim_indices
            ]
            default_amo = self.default_dof_angles[self._amo_sim_indices]
            # 23-dim: real NN output for legs+waist(15), a pseudo-action for the
            # (externally driven) shoulder/elbow(8) so the adapter/obs history
            # still sees a consistent signal there.
            arm_pseudo_action = (dof_pos_amo - default_amo)[n:] / self._ACTION_SCALE
            action = np.concatenate([raw, arm_pseudo_action]).astype(np.float32)

            q_target = self.default_dof_angles.copy()
            # legs+waist: default + scaled NN action. arms/wrists stay at their
            # yaml default here -- see the class docstring.
            q_target[self._amo_sim_indices[:n]] += raw * self._ACTION_SCALE
            return action, q_target, input_dict

        self.policy = policy


class DualBranchPolicyMixin:
    """ALMI (mjlab-loco dual policy): two actors running in parallel.

    Unlike ``SplitOnnxPolicyMixin``'s encoder -> decoder chain, these two graphs
    are independent and joint-disjoint: ``lower`` walks on the 12 leg joints and
    ``upper`` drives the 17 waist/arm joints, off separate observations, and
    their raw outputs concatenate into the single 29-D action the rest of the
    runtime expects. Declared by ``upper_model_path`` in policy.yaml
    (``model_path`` is the lower branch).

    What each branch OBSERVES depends on the checkpoint's layout (see
    ``sim2real.rl_policy.observations.almi``: ``whole_body`` shows both branches
    all 29 joints, ``half_body`` narrows each to its own half). This mixin is
    indifferent to that -- it hands the observation terms the full 29-D
    concatenated action as ``state_dict["action"]`` and lets them take whatever
    slice their layout calls for, so ``prev_action`` is correct either way with
    nothing to configure here. A partial-DOF family like AMO has to synthesize a
    pseudo-action for the joints its network does not emit; this one does not,
    because the two branches together emit all 29.
    """

    #: Observation group -> ONNX input name, per branch. The group names are the
    #: ALMI runtime's own (see checkpoints/almi/*/policy.yaml); the input names
    #: are what scripts/export_almi_onnx.py wrote into the graphs.
    _LOWER_INPUTS = {"lower_obs": "obs", "lower_history": "lower_actor_history"}
    _UPPER_INPUTS = {"upper_obs": "obs"}

    def setup_policy(self, model_path: str) -> None:
        upper_path = self.policy_config.get("upper_model_path")
        if not upper_path:
            raise ValueError(
                f"{type(self).__name__} requires policy.yaml to set "
                "'upper_model_path' (absolute, or relative to the yaml's own "
                "directory)."
            )
        upper_path = str(
            pathlib.Path(self.args.policy_config).parent / str(upper_path)
        )

        self._lower_module = build_inference_module(model_path, self.inference_backend)
        self._upper_module = build_inference_module(upper_path, self.inference_backend)
        logger.info(
            "{} loaded lower={} upper={}",
            type(self).__name__,
            model_path,
            upper_path,
        )

        action_clip_cfg = self.policy_config.get("action_clip")
        action_clip = (
            None if action_clip_cfg is None else float(action_clip_cfg)
        )
        if action_clip is not None and action_clip <= 0.0:
            raise ValueError("ALMI action_clip must be greater than zero")

        soft_limit_cfg = self.policy_config.get("soft_joint_pos_limit_factor")
        soft_limit_factor = (
            None if soft_limit_cfg is None else float(soft_limit_cfg)
        )
        if soft_limit_factor is not None and not 0.0 < soft_limit_factor <= 1.0:
            raise ValueError(
                "ALMI soft_joint_pos_limit_factor must be in (0, 1]"
            )

        controlled_indices = np.asarray(
            self.controlled_joint_indices, dtype=np.int64
        )
        soft_lower = soft_upper = None
        if soft_limit_factor is not None:
            hard_lower = np.asarray(
                self.joint_pos_lower_limit[controlled_indices], dtype=np.float64
            )
            hard_upper = np.asarray(
                self.joint_pos_upper_limit[controlled_indices], dtype=np.float64
            )
            if (
                not np.all(np.isfinite(hard_lower))
                or not np.all(np.isfinite(hard_upper))
                or np.any(hard_upper <= hard_lower)
            ):
                raise ValueError(
                    "ALMI cannot build soft joint limits from invalid robot limits"
                )
            center = 0.5 * (hard_lower + hard_upper)
            half_range = 0.5 * soft_limit_factor * (hard_upper - hard_lower)
            soft_lower = center - half_range
            soft_upper = center + half_range

        action_clip_warning_emitted = False

        def branch(module, mapping, input_dict):
            # ONNXModule wants the literal (batch=1, dim) shape each graph
            # declares, and every ALMI input has a fixed batch dim.
            feed = {
                onnx_name: np.asarray(
                    input_dict[group], dtype=np.float32
                ).reshape(1, -1)
                for group, onnx_name in mapping.items()
            }
            out = module(feed)
            return np.asarray(out["action"], dtype=np.float32).reshape(-1)

        def policy(input_dict):
            nonlocal action_clip_warning_emitted
            lower_action = branch(self._lower_module, self._LOWER_INPUTS, input_dict)
            upper_action = branch(self._upper_module, self._UPPER_INPUTS, input_dict)
            raw_action = np.concatenate([lower_action, upper_action]).astype(np.float32)
            if raw_action.shape[0] != self.num_actions:
                raise ValueError(
                    f"ALMI branches produced {raw_action.shape[0]} actions "
                    f"(lower={lower_action.shape[0]}, upper={upper_action.shape[0]}), "
                    f"but policy_joint_names declares {self.num_actions}"
                )
            if not np.all(np.isfinite(raw_action)):
                raise FloatingPointError("ALMI policy produced a non-finite action")

            action = raw_action
            if action_clip is not None:
                action = np.clip(
                    raw_action, -action_clip, action_clip
                ).astype(np.float32, copy=False)
                if (
                    not action_clip_warning_emitted
                    and np.any(action != raw_action)
                ):
                    logger.warning(
                        "ALMI raw action |a|max={:.3f} clipped to +/-{:.3f} "
                        "before q_target and prev_action",
                        float(np.max(np.abs(raw_action))),
                        action_clip,
                    )
                    action_clip_warning_emitted = True

            q_target = self.default_dof_angles.copy()
            q_target[controlled_indices] += action * self.action_scale
            if soft_lower is not None and soft_upper is not None:
                q_target[controlled_indices] = np.clip(
                    q_target[controlled_indices], soft_lower, soft_upper
                )
            return action, q_target, input_dict

        self.policy = policy


def policy_family_mixin(policy_config: dict) -> type | None:
    """Which runtime mixin a parsed policy.yaml calls for, or None for the
    plain fused-ONNX case."""
    if policy_config.get("encoder_model_path"):
        return SplitOnnxPolicyMixin
    if policy_config.get("amo_adapter_path"):
        return TorchJitAdapterPolicyMixin
    if policy_config.get("upper_model_path"):
        return DualBranchPolicyMixin
    return None


def tracking_class_for(
    policy_config: dict, base: Type[Tracking] = Tracking
) -> Type[Tracking]:
    """Compose ``base`` with whichever family mixin the config calls for.

    ``base`` lets a host keep its own ``Tracking`` subclass (a simulator
    adapter, a test double) and still get the right family runtime on top.
    Returns ``base`` unchanged when no mixin applies.
    """
    mixin = policy_family_mixin(policy_config)
    if mixin is None:
        return base
    return type(f"{mixin.__name__.removesuffix('PolicyMixin')}{base.__name__}",
                (mixin, base), {})


def make_tracking(
    policy_config_path: str,
    *,
    base: Type[Tracking] = Tracking,
    args_cls: type = TrackingArgs,
    **arg_overrides,
) -> Tracking:
    """Build the right ``Tracking`` for one checkpoint, from its yaml alone.

    ``arg_overrides`` sets any ``TrackingArgs`` field (``inference_backend``,
    ``robot_io``, ``motion_backend``, ...). The yaml is read once here to pick
    the class, because ``BasePolicy.__init__`` cannot choose its own type.
    """
    with open(policy_config_path) as f:
        raw_policy_config = yaml.safe_load(f)
    cls = tracking_class_for(raw_policy_config, base=base)
    return cls(args_cls(policy_config=policy_config_path, **arg_overrides))
