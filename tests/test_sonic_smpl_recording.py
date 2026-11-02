"""SonicBackend: the recorded ``teleop.smpl_*`` columns.

The regression these cover: the PICO source only pre-computes ``pose_smpl_*``
when its background pose pipeline is enabled (``PICO_ENABLE_BACKGROUND_POSE_DATA``,
default OFF), and nothing in the collection path sets it. POSE-mode teleop still
worked -- the encoder path re-derived the pose from the raw sample and kept the
result to itself -- so every real-PICO dataset recorded ``teleop.smpl_joints`` /
``smpl_root_quat`` / ``wrist_joints`` as all-zero while the robot was visibly
tracking the operator. The recorder now runs the same derivation.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from g1_bridge.backends.sonic import SonicBackend
from g1_bridge.pico.mode_manager import ManagerMode


def _raw_smpl(seed: int = 0) -> np.ndarray:
    """A plausible (24, 7) xrobotoolkit body-joints sample: positions plus
    normalized scalar-last quaternions."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.3, size=(24, 3))
    quat = rng.normal(size=(24, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return np.hstack([pos, quat]).astype(np.float32)


class _FakePicoSource:
    def __init__(self, sample):
        self.sample = sample

    def get_latest(self):
        return self.sample


class _FakeModeManager:
    def __init__(self, mode):
        self.current = mode


class _FakeCtx:
    """The slice of Sim2RealContext teleop_sample touches."""

    def __init__(self, sample, mode=ManagerMode.POSE):
        self.pico_source = _FakePicoSource(sample)
        self.mode_manager = _FakeModeManager(mode)
        self.tracking = None
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.cached_target_q = np.zeros(29, dtype=np.float32)


def _sample(*, raw=True, pose_precomputed=False, age_s=0.0, seed=0,
            sample_clock_age_s=None):
    """A PICO sample as the REAL source builds it: raw SMPL always present,
    ``pose_smpl_*`` only when the background pose pipeline is on.

    ``age_s`` ages the BODY clock. ``sample_clock_age_s`` ages the sample clock
    independently, to reproduce the isolated worker's stale-body path, where
    ``timestamp`` is re-stamped every loop while ``body_timestamp`` stands
    still.
    """
    now = time.monotonic()
    sample = {
        # Deliberately non-zero: an all-zero fixture cannot tell "gated out"
        # from "passed through", which is the whole point of these tests.
        "vr_3point_local_target": np.arange(1, 10, dtype=np.float32) * 0.1,
        "vr_3point_local_orn_target": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 3
        ),
        "calibrated": True,
        "timestamp": now - (age_s if sample_clock_age_s is None else sample_clock_age_s),
        "body_timestamp": now - age_s,
        "buttons": {},
        "raw_smpl": _raw_smpl(seed) if raw else None,
        "pose_smpl_joints_local": None,
        "pose_smpl_root_quat": None,
        "pose_anchor_6d": None,
        "pose_wrist_joints": None,
    }
    if pose_precomputed:
        from g1_bridge.pico.vr_source import process_smpl_pose_full

        data = process_smpl_pose_full(sample["raw_smpl"])
        sample["pose_smpl_joints_local"] = data["smpl_joints_local"]
        sample["pose_smpl_root_quat"] = data["smpl_root_quat"]
        sample["pose_anchor_6d"] = data["anchor_6d"]
        sample["pose_wrist_joints"] = data["wrist_joints"]
    return sample


#: (recorded key, flattened width) for every column fed by the SMPL derivation.
SMPL_COLUMNS = (
    ("smpl_joints_local", 72),
    ("smpl_root_quat", 4),
    ("wrist_joints", 6),
    ("smpl_pose", 63),
)

#: The 3-point columns come from the SAME body frame (vr_source computes them
#: with three_point.process_smpl_pose(body_poses_np)), so they share its
#: freshness verdict -- otherwise one half of a frame would carry a stale
#: operator while the other half zeroed.
BODY_DERIVED_COLUMNS = SMPL_COLUMNS + (
    ("vr_3point_pos", 9),
    ("vr_3point_orn", 12),
)


def test_smpl_recorded_without_background_pose_data():
    """The default real-PICO shape: raw SMPL only. Every smpl column must be
    populated -- this is the all-zeros bug."""
    out = SonicBackend().teleop_sample(_FakeCtx(_sample()))
    for key, width in SMPL_COLUMNS:
        value = out[key]
        assert value is not None, f"{key} is None -> recorded as zeros"
        assert value.shape == (width,)
        assert np.any(value != 0.0), f"{key} is all-zero"


def test_smpl_matches_the_source_precomputed_values():
    """Deriving here and having the source derive it must agree exactly --
    otherwise the columns would mean different things depending on an env var."""
    derived = SonicBackend().teleop_sample(_FakeCtx(_sample()))
    precomputed = SonicBackend().teleop_sample(
        _FakeCtx(_sample(pose_precomputed=True))
    )
    for key, _ in SMPL_COLUMNS:
        np.testing.assert_allclose(derived[key], precomputed[key], rtol=0, atol=0)


def test_stale_body_is_caught_even_while_the_sample_clock_ticks():
    """The isolated PICO worker (what --pico-source real actually uses) keeps
    re-stamping ``timestamp`` through a body-tracking outage so buttons stay
    live. Gating on that clock would never fire, and the recorder would write
    the last known pose forever."""
    sample = _sample(age_s=1.0, sample_clock_age_s=0.0)
    out = SonicBackend().teleop_sample(_FakeCtx(sample))
    for key, _ in BODY_DERIVED_COLUMNS:
        assert out[key] is None, f"{key} followed the re-stamped sample clock"


def test_stale_smpl_records_zeros_not_a_frozen_operator():
    """A dead SMPL stream must be visible in the data, not look like someone
    standing very still. Mirrors the official collector's 100 ms gate."""
    out = SonicBackend().teleop_sample(_FakeCtx(_sample(age_s=1.0)))
    for key, _ in BODY_DERIVED_COLUMNS:
        assert out[key] is None, f"{key} recorded a stale pose"


def test_a_frame_never_mixes_a_fresh_and_a_stale_operator():
    """The failure this rules out: 3-point columns carrying where the operator
    was 800 ms ago while the smpl columns beside them read zero. Nothing
    downstream could tell the two apart -- both are valid-looking numbers for
    the same tick."""
    fresh = SonicBackend().teleop_sample(_FakeCtx(_sample()))
    stale = SonicBackend().teleop_sample(_FakeCtx(_sample(age_s=1.0)))
    assert all(fresh[k] is not None for k, _ in BODY_DERIVED_COLUMNS)
    assert all(stale[k] is None for k, _ in BODY_DERIVED_COLUMNS)


def test_operator_is_told_when_the_body_stream_dies_and_recovers(capsys):
    """Zeros in a column are evidence you only see after the session. The
    official collector says it out loud while it happens; so does this."""
    backend = SonicBackend()
    backend.teleop_sample(_FakeCtx(_sample()))  # healthy: silent
    capsys.readouterr()

    backend.teleop_sample(_FakeCtx(_sample(age_s=0.8)))
    warned = capsys.readouterr().out
    assert "stale" in warned and "ZEROS" in warned, warned

    # Still stale on the very next tick -> no second line (50 Hz would flood).
    backend.teleop_sample(_FakeCtx(_sample(age_s=0.8)))
    assert capsys.readouterr().out == ""

    backend.teleop_sample(_FakeCtx(_sample()))
    assert "recovered" in capsys.readouterr().out


def test_no_smpl_stream_records_zeros():
    """Mock/keyboard sources, and a real headset before body tracking is up."""
    out = SonicBackend().teleop_sample(_FakeCtx(_sample(raw=False)))
    for key, _ in SMPL_COLUMNS:
        assert out[key] is None


@pytest.mark.parametrize(
    "mode",
    [ManagerMode.POSE, ManagerMode.VR_3PT, ManagerMode.PLANNER, ManagerMode.OFF],
)
def test_smpl_recorded_in_every_mode(mode):
    """Deliberate superset of the official collector, which zeroes these
    outside stream_mode 1/4: the mode is recorded alongside in
    teleop.stream_mode, so a consumer can reproduce upstream's gate offline.
    Dropping the data at collection time is the one thing that cannot be undone."""
    out = SonicBackend().teleop_sample(_FakeCtx(_sample(), mode=mode))
    assert out["stream_mode"][0] == int(mode)
    for key, _ in SMPL_COLUMNS:
        assert out[key] is not None


def test_derivation_is_memoized_per_sample():
    """teleop_sample and the encoder path both ask for the same pose; the
    retarget must run once per PICO sample, not once per caller."""
    backend = SonicBackend()
    sample = _sample()
    calls = []
    import g1_bridge.backends.sonic as sonic_mod

    real = sonic_mod.process_smpl_pose_full

    def counting(raw):
        calls.append(raw)
        return real(raw)

    sonic_mod.process_smpl_pose_full = counting
    try:
        ctx = _FakeCtx(sample)
        backend.teleop_sample(ctx)
        backend._read_operator_pose(ctx)
        backend.teleop_sample(ctx)
        assert len(calls) == 1
        # A new sample (new timestamp) must invalidate the memo.
        ctx.pico_source.sample = _sample(seed=1)
        backend.teleop_sample(ctx)
        assert len(calls) == 2
    finally:
        sonic_mod.process_smpl_pose_full = real
