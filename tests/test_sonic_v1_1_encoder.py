"""sonic_v1_1's 1751D g1 encoder input.

Shape being right is the cheap half. What these tests pin is the part that
fails silently: block offsets, which blocks stay zero, and that the anchor is
normalized by the robot's HEADING rather than its body frame -- a difference
that is invisible whenever the robot happens to be upright, which is most of
the time in a smoke test.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from g1_bridge.sonic_utils.encoder_obs import (
    ENCODER_DIM,
    ENCODER_DIM_V1_1,
    sonic_g1_low_latency_encoder_input,
    sonic_g1_v1_1_encoder_input,
)
from sim2real.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_1_DIR = REPO_ROOT / "checkpoints" / "sonic" / "v1_1" / "g1"

# The block spans from sonic_v1_1/observation_config.yaml's
# encoder_observations order x model_config.yaml's group_obs_dims.tokenizer.
# Written out longhand, NOT derived from the constants in encoder_obs.py --
# a test that recomputes the layout from the same expressions the code uses
# would pass no matter what either of them said.
V1_1_LAYOUT = [
    ("encoder_mode_4", 0, 4),
    ("motion_joint_positions_10frame_step5", 4, 294),
    ("motion_joint_velocities_10frame_step5", 294, 584),
    ("motion_anchor_orientation_heading_10frame_step5", 584, 644),
    ("motion_anchor_orientation_heading", 644, 650),
    ("motion_joint_positions_lowerbody_10frame_step5", 650, 770),
    ("motion_joint_velocities_lowerbody_10frame_step5", 770, 890),
    ("vr_3point_local_target", 890, 899),
    ("vr_3point_local_orn_target", 899, 911),
    ("smpl_joints_10frame_step1", 911, 1631),
    ("smpl_anchor_orientation_heading_10frame_step1", 1631, 1691),
    ("motion_joint_positions_wrists_10frame_step1", 1691, 1751),
]
#: Everything from here up is teleop/smpl and is not implemented.
G1_BLOCK_END = 650

ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
N_JOINTS = len(ISAAC_JOINT_NAMES)
N_FRAMES = 10


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w, x, y, z) from intrinsic xyz Euler angles."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def make_env(robot_root_quat_w: np.ndarray, seed: int = 20260812) -> SimpleNamespace:
    """A Tracking-shaped stand-in carrying one deterministic motion window.

    The motion arrays are indexed by *position* in ``motion_future_steps``, not
    by step value, so the same window serves a step1 and a step5 observation --
    which is what makes the low_latency-vs-v1_1 comparison below meaningful.
    """
    rng = np.random.default_rng(seed)
    body_names = ["pelvis"]
    quats = rng.standard_normal((1, N_FRAMES, 1, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    motion_data = SimpleNamespace(
        joint_pos=rng.standard_normal((1, N_FRAMES, N_JOINTS)).astype(np.float32),
        joint_vel=rng.standard_normal((1, N_FRAMES, N_JOINTS)).astype(np.float32),
        body_pos_w=rng.standard_normal((1, N_FRAMES, 1, 3)).astype(np.float32),
        body_lin_vel_w=rng.standard_normal((1, N_FRAMES, 1, 3)).astype(np.float32),
        body_quat_w=quats,
        body_ang_vel_w=rng.standard_normal((1, N_FRAMES, 1, 3)).astype(np.float32),
    )
    return SimpleNamespace(
        state_processor=SimpleNamespace(
            root_quat_w=np.asarray(robot_root_quat_w, dtype=np.float32)
        ),
        motion_data=motion_data,
        motion_joint_names=list(ISAAC_JOINT_NAMES),
        motion_body_names=list(body_names),
        policy_joint_names=list(ISAAC_JOINT_NAMES),
        joint_names_simulation=list(ISAAC_JOINT_NAMES),
        body_names_simulation=list(body_names),
        motion_future_steps=list(range(N_FRAMES)),
        motion_config={},
    )


def build(cls, env, future_steps=None):
    obs = cls(
        env=env,
        future_steps=list(range(N_FRAMES)) if future_steps is None else future_steps,
        joint_names=list(ISAAC_JOINT_NAMES),
        root_body_name="pelvis",
    )
    obs.reset()
    obs.update({})
    return obs


class V11LayoutTest(unittest.TestCase):
    def test_block_offsets_sum_to_the_encoder_width(self) -> None:
        self.assertEqual(V1_1_LAYOUT[0][1], 0)
        for (_, _, prev_end), (name, start, _) in zip(V1_1_LAYOUT, V1_1_LAYOUT[1:]):
            self.assertEqual(prev_end, start, f"gap or overlap before {name}")
        self.assertEqual(V1_1_LAYOUT[-1][2], ENCODER_DIM_V1_1)
        self.assertEqual(ENCODER_DIM_V1_1, 1751)

    def test_only_the_g1_blocks_are_populated(self) -> None:
        env = make_env(quat_from_euler(0.11, -0.07, 0.9))
        out = build(sonic_g1_v1_1_encoder_input, env).compute()

        self.assertEqual(out.shape, (ENCODER_DIM_V1_1,))
        self.assertEqual(out.dtype, np.float32)
        # mode_id 0 = g1, and only the first of the four slots carries it
        np.testing.assert_array_equal(out[0:4], [0.0, 0.0, 0.0, 0.0])
        # every g1 block has some signal in it (a zero block here would mean a
        # wiring slip that the total-width check above cannot see)
        for name, start, end in V1_1_LAYOUT[1:4]:
            self.assertTrue(np.any(out[start:end] != 0.0), f"{name} is all zero")
        # teleop + smpl: not implemented, must stay exactly zero
        np.testing.assert_array_equal(
            out[G1_BLOCK_END:], np.zeros(ENCODER_DIM_V1_1 - G1_BLOCK_END, np.float32)
        )

    def test_g1_reference_blocks_match_low_latency(self) -> None:
        """Same window, same step selection -> the mode/pos/vel blocks must be
        bit-identical to the shipping 1247D builder. Only the anchor and the
        total width are allowed to differ."""
        env_a = make_env(quat_from_euler(0.11, -0.07, 0.9))
        env_b = make_env(quat_from_euler(0.11, -0.07, 0.9))
        old = build(sonic_g1_low_latency_encoder_input, env_a).compute()
        new = build(sonic_g1_v1_1_encoder_input, env_b).compute()

        self.assertEqual(old.shape, (ENCODER_DIM,))
        np.testing.assert_array_equal(new[0:584], old[0:584])
        # ...and the anchor genuinely is different (this robot is not upright)
        self.assertFalse(np.allclose(new[584:650], old[584:650]))


class V11HeadingAnchorTest(unittest.TestCase):
    """The heading normalization, isolated.

    conj(yaw_only(robot)) * ref  ==  conj(robot) * ref  exactly when the robot
    has no roll or pitch. So an upright robot cannot tell the two apart, and a
    leaning one must.
    """

    def _anchor_blocks(self, robot_quat):
        env_a = make_env(robot_quat)
        env_b = make_env(robot_quat)
        body = build(sonic_g1_low_latency_encoder_input, env_a).compute()[584:650]
        heading = build(sonic_g1_v1_1_encoder_input, env_b).compute()[584:650]
        return body, heading

    def test_identical_when_robot_is_upright(self) -> None:
        for yaw in (0.0, 0.7, -2.4, np.pi):
            with self.subTest(yaw=yaw):
                body, heading = self._anchor_blocks(quat_from_euler(0.0, 0.0, yaw))
                np.testing.assert_allclose(heading, body, atol=1e-6)

    def test_differs_when_robot_leans(self) -> None:
        for roll, pitch in ((0.25, 0.0), (0.0, -0.3), (0.15, 0.2)):
            with self.subTest(roll=roll, pitch=pitch):
                body, heading = self._anchor_blocks(quat_from_euler(roll, pitch, 0.6))
                self.assertFalse(np.allclose(heading, body, atol=1e-4))

    def test_matches_a_hand_rolled_heading_computation(self) -> None:
        """Recompute the multi-frame anchor from the raw motion window with an
        independent expression, and require the observation to agree."""
        robot_quat = quat_from_euler(0.18, -0.12, 1.3)
        env = make_env(robot_quat)
        obs = build(sonic_g1_v1_1_encoder_input, env)
        got = obs.compute()[584:644].reshape(N_FRAMES, 6)

        ref = env.motion_data.body_quat_w[0, :, 0, :]
        # the parent's episode-start yaw alignment, reused verbatim -- this
        # test is about the heading frame, not about that offset
        offset = np.broadcast_to(obs._anchor._heading_offset.reshape(1, 4), ref.shape)
        ref = quat_mul(offset, ref)
        robot_yaw = np.broadcast_to(
            yaw_quat(np.asarray(robot_quat, np.float32).reshape(1, 4)), ref.shape
        )
        want = matrix_from_quat(quat_mul(quat_conjugate(robot_yaw), ref))[..., :, :2]
        np.testing.assert_allclose(got, want.reshape(N_FRAMES, 6), atol=1e-6)


class V11EncoderModesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = build(sonic_g1_v1_1_encoder_input, make_env(quat_from_euler(0, 0, 0)))

    def test_every_sub_encoder_is_accepted(self) -> None:
        for mode in ("g1", "smpl", "teleop"):
            with self.subTest(mode=mode):
                self.obs.set_encoder_mode(mode)

    def test_unknown_mode_is_rejected(self) -> None:
        # The guard still has to bite: encoding a zeroed block for a mode this
        # class cannot populate would look like it worked.
        with self.assertRaisesRegex(ValueError, "encoder mode must be one of"):
            self.obs.set_encoder_mode("nope")

    def test_declared_capabilities_match_what_it_accepts(self) -> None:
        self.assertEqual(
            sonic_g1_v1_1_encoder_input.SUPPORTED_ENCODER_MODES,
            ("g1", "smpl", "teleop"),
        )
        # The two shipping classes must agree, or POSE silently becomes
        # variant-dependent for reasons nobody wrote down.
        self.assertEqual(
            sonic_g1_low_latency_encoder_input.SUPPORTED_ENCODER_MODES,
            sonic_g1_v1_1_encoder_input.SUPPORTED_ENCODER_MODES,
        )

    def test_anchor_frame_is_declared_heading(self) -> None:
        # _push_pose_reference reads this to pick relative_anchor6's frame.
        self.assertEqual(sonic_g1_v1_1_encoder_input.POSE_ANCHOR_FRAME, "heading")
        self.assertEqual(sonic_g1_low_latency_encoder_input.POSE_ANCHOR_FRAME, "body")


class V11SmplBlockTest(unittest.TestCase):
    """The smpl (POSE) block: 10-frame, three slots, everything else zero."""

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.joints = rng.standard_normal((24, 3)).astype(np.float32)
        self.anchor = rng.standard_normal(6).astype(np.float32)
        self.wrists = rng.standard_normal(6).astype(np.float32)
        self.obs = build(sonic_g1_v1_1_encoder_input, make_env(quat_from_euler(0, 0, 0.4)))
        self.obs.set_pose_reference(self.joints, self.anchor, self.wrists)
        self.obs.set_encoder_mode("smpl")
        self.out = self.obs.compute()

    def test_mode_id_and_width(self) -> None:
        self.assertEqual(self.out.shape, (ENCODER_DIM_V1_1,))
        np.testing.assert_array_equal(self.out[0:4], [2.0, 0.0, 0.0, 0.0])

    def test_every_frame_carries_the_current_pose(self) -> None:
        """Live operator stream: there is no future to look ahead at, so all 10
        frames hold the current one (the C++ deploy's play==false behaviour)."""
        joints = self.out[911:1631].reshape(10, 24, 3)
        anchor = self.out[1631:1691].reshape(10, 6)
        wrists = self.out[1691:1751].reshape(10, 6)
        for frame in range(10):
            np.testing.assert_array_equal(joints[frame], self.joints)
            np.testing.assert_array_equal(anchor[frame], self.anchor)
            np.testing.assert_array_equal(wrists[frame], self.wrists)

    def test_ten_frames_not_four(self) -> None:
        """The 1247 -> 1751 growth is entirely this. A 4-frame port would leave
        the tail zero and still be the right total width."""
        self.assertTrue(np.any(self.out[911 + 4 * 24 * 3 : 1631] != 0.0))

    def test_g1_and_teleop_blocks_stay_zero(self) -> None:
        np.testing.assert_array_equal(self.out[4:911], np.zeros(907, np.float32))

    def test_switching_back_to_g1_restores_the_g1_block(self) -> None:
        self.obs.set_encoder_mode("g1")
        g1 = self.obs.compute()
        np.testing.assert_array_equal(g1[911:], np.zeros(1751 - 911, np.float32))
        self.assertTrue(np.any(g1[4:650] != 0.0))


class HeadingAnchorFrameTest(unittest.TestCase):
    """``relative_anchor6(frame=...)`` -- the POSE-side half of the heading
    normalization, which lives in the backend rather than the observation."""

    def test_yaw_only_quat_matches_the_training_convention(self) -> None:
        """g1_runtime cannot import sim2real (torch), so it reimplements
        yaw_quat. Bit-compatibility with the original is the whole point."""
        from sim2real.utils.math import yaw_quat

        from g1_bridge.g1_runtime import _yaw_only_quat

        rng = np.random.default_rng(11)
        for _ in range(50):
            q = rng.standard_normal(4).astype(np.float32)
            q /= np.linalg.norm(q)
            np.testing.assert_allclose(
                _yaw_only_quat(q), yaw_quat(q.reshape(1, 4))[0], atol=1e-6
            )

    def test_frames_agree_when_upright_and_differ_when_leaning(self) -> None:
        from g1_bridge.g1_runtime import relative_anchor6

        ref = quat_from_euler(0.3, -0.2, 0.5)
        upright = quat_from_euler(0.0, 0.0, 0.9)
        np.testing.assert_allclose(
            relative_anchor6(upright, ref, frame="heading"),
            relative_anchor6(upright, ref, frame="body"),
            atol=1e-6,
        )
        leaning = quat_from_euler(0.2, 0.15, 0.9)
        self.assertFalse(
            np.allclose(
                relative_anchor6(leaning, ref, frame="heading"),
                relative_anchor6(leaning, ref, frame="body"),
                atol=1e-4,
            )
        )

    def test_unknown_frame_is_refused(self) -> None:
        from g1_bridge.g1_runtime import relative_anchor6

        with self.assertRaisesRegex(ValueError, "body.*heading"):
            relative_anchor6(
                quat_from_euler(0, 0, 0), quat_from_euler(0, 0, 0), frame="world"
            )


class BackendPoseGuardTest(unittest.TestCase):
    """SonicBackend asks the observation what it supports instead of
    type-checking, so POSE availability follows the manifest."""

    def _backend(self):
        from g1_bridge.backends.sonic import SonicBackend

        return SonicBackend

    def test_both_shipping_variants_allow_pose(self) -> None:
        backend = self._backend()
        for cls in (sonic_g1_low_latency_encoder_input, sonic_g1_v1_1_encoder_input):
            with self.subTest(cls=cls.__name__):
                obs = build(cls, make_env(quat_from_euler(0, 0, 0)))
                self.assertTrue(backend._encoder_supports(obs, "smpl"))
                # no raise
                backend._reject_pose_without_smpl_encoder(backend, None, obs)

    def test_a_g1_only_encoder_would_still_be_refused(self) -> None:
        """The guard is not dead code just because nothing trips it today."""
        backend = self._backend()

        class G1Only:
            SUPPORTED_ENCODER_MODES = ("g1",)

        obs = G1Only()
        self.assertFalse(backend._encoder_supports(obs, "smpl"))
        with self.assertRaisesRegex(RuntimeError, "does not populate"):
            backend._reject_pose_without_smpl_encoder(backend, None, obs)

    def test_an_encoder_that_declares_nothing_is_treated_as_g1_only(self) -> None:
        backend = self._backend()
        self.assertFalse(backend._encoder_supports(object(), "smpl"))


class V11ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        path = V1_1_DIR / "policy.yaml"
        if not path.exists():
            self.skipTest(f"{path} not present")
        with path.open() as handle:
            self.config = yaml.safe_load(handle)

    def test_points_at_the_v1_1_observation(self) -> None:
        entry = self.config["observation"]["g1_input"]["g1_encoder_input"]
        self.assertEqual(entry["_target_"], "sonic.sonic_g1_v1_1_encoder_input")

    def test_reference_horizon_is_step5(self) -> None:
        entry = self.config["observation"]["g1_input"]["g1_encoder_input"]
        self.assertEqual(entry["future_steps"], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        # the buffer must offer every step the observation selects
        self.assertEqual(self.config["motion"]["future_steps"], entry["future_steps"])

    def test_proprioception_history_stays_step1(self) -> None:
        """Only the *reference* went to step5; the decoder is unchanged, so its
        10-frame proprioception history must not move with it."""
        for name, term in self.config["observation"]["proprioception"].items():
            self.assertEqual(term["history_steps"], list(range(10)), name)

    def test_action_contract_is_unchanged_from_low_latency(self) -> None:
        with (REPO_ROOT / "checkpoints" / "sonic" / "low_latency" / "g1"
              / "policy.yaml").open() as handle:
            base = yaml.safe_load(handle)
        for key in ("policy_joint_names", "joint_names_simulation",
                    "body_names_simulation", "action_scale", "default_joint_pos",
                    "joint_kp", "joint_kd"):
            self.assertEqual(self.config[key], base[key], key)


class V11OnnxContractTest(unittest.TestCase):
    """The one check that cannot be inferred from any config file."""

    def setUp(self) -> None:
        self.encoder = V1_1_DIR / "model_encoder.onnx"
        self.decoder = V1_1_DIR / "model_decoder.onnx"
        if not (self.encoder.exists() and self.decoder.exists()):
            self.skipTest("v1_1 weights not downloaded; see docs/artifacts.md")
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("onnx not installed")

    def _io(self, path):
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        shape = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
        return (
            {v.name: shape(v) for v in model.graph.input},
            {v.name: shape(v) for v in model.graph.output},
        )

    def test_encoder_width_matches_the_observation(self) -> None:
        inputs, outputs = self._io(self.encoder)
        self.assertEqual(inputs, {"obs_dict": [1, ENCODER_DIM_V1_1]})
        self.assertEqual(outputs, {"encoded_tokens": [1, 64]})

    def test_decoder_is_the_shared_994_to_29(self) -> None:
        inputs, outputs = self._io(self.decoder)
        self.assertEqual(inputs, {"obs_dict": [1, 994]})
        self.assertEqual(outputs, {"action": [1, 29]})


if __name__ == "__main__":
    unittest.main()
