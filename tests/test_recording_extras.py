from __future__ import annotations

import pytest

from sim2real.rl_policy.recording.extras.base import RecordingExtra


def test_resolve_finds_the_builtin_example_extra() -> None:
    # Importing sim2real.rl_policy.recording.extras (done at module load below via the
    # conftest-free direct import chain) registers the built-in example extra.
    import sim2real.rl_policy.recording.extras  # noqa: F401

    resolved = RecordingExtra.resolve("example")
    assert resolved.extra_name == "example"


def test_resolve_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="not found"):
        RecordingExtra.resolve("does-not-exist")


def test_duplicate_extra_name_registration_raises() -> None:
    class _First(RecordingExtra, extra_name="dup-test"):
        pass

    with pytest.raises(ValueError, match="already registered"):

        class _Second(RecordingExtra, extra_name="dup-test"):
            pass


def test_example_extra_features_and_compute_round_trip() -> None:
    import sim2real.rl_policy.recording.extras  # noqa: F401

    extra_cls = RecordingExtra.resolve("example")
    extra = extra_cls(policy=None)

    features = extra.features()
    assert "action.example_step_parity" in features
    assert features["action.example_step_parity"]["shape"] == (1,)

    even = extra.compute({"step_index": 4})
    odd = extra.compute({"step_index": 5})
    assert even["action.example_step_parity"][0] == 1.0
    assert odd["action.example_step_parity"][0] == 0.0


def test_default_features_and_compute_are_empty() -> None:
    class _Bare(RecordingExtra, extra_name="bare-test"):
        pass

    extra = _Bare(policy=None)
    assert extra.features() == {}
    assert extra.compute({"step_index": 0}) == {}
    extra.close()  # should not raise
