"""Ego-stream configuration without sockets, SDKs, policies or scene assets."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def stream(monkeypatch):
    # Load this leaf module directly: configuration does not require importing
    # the simulator task catalog or any optional camera/robot runtime.
    path = Path(__file__).resolve().parents[1] / "robocasa/utils/ego_stream.py"
    spec = importlib.util.spec_from_file_location("_test_ego_stream_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        module.ENV_ENDPOINT, module.ENV_KEY, module.ENV_STRIDE, module.ENV_QUALITY,
        *module._LEGACY_ENV_NAMES.values(),
    ):
        monkeypatch.delenv(name, raising=False)
    # The real publisher must remain lazy when disabled. Capture its arguments
    # for enabled configuration tests without creating a socket or importing SDKs.
    class CapturePublisher(module.EgoFramePublisher):
        def __init__(self, endpoint, key, stride, quality):
            self.config = (endpoint, key, stride, quality)

    module.CapturePublisher = CapturePublisher
    return module


def test_disabled_stream_needs_no_optional_transport(stream, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "zmq", None)
    monkeypatch.setitem(sys.modules, "sim2real", None)
    assert stream.EgoFramePublisher.maybe() is None


def test_current_variables_take_precedence_for_all_stream_options(stream, monkeypatch):
    for name, value in {
        "LOCOMANILAB_EGO_STREAM": "6000",
        "LOCOMANILAB_EGO_STREAM_KEY": "current-camera",
        "LOCOMANILAB_EGO_STREAM_STRIDE": "3",
        "LOCOMANILAB_EGO_STREAM_QUALITY": "91",
        "SOMA_EGO_STREAM": "5999",
        "SOMA_EGO_STREAM_KEY": "old-camera",
        "SOMA_EGO_STREAM_STRIDE": "2",
        "SOMA_EGO_STREAM_QUALITY": "80",
    }.items():
        monkeypatch.setenv(name, value)
    assert stream.CapturePublisher.maybe().config == (
        "tcp://127.0.0.1:6000", "current-camera", 3, 91,
    )


def test_legacy_only_settings_remain_readable(stream, monkeypatch):
    for name, value in {
        "SOMA_EGO_STREAM": "5999", "SOMA_EGO_STREAM_KEY": "old-camera",
        "SOMA_EGO_STREAM_STRIDE": "2", "SOMA_EGO_STREAM_QUALITY": "80",
    }.items():
        monkeypatch.setenv(name, value)
    assert stream.CapturePublisher.maybe().config == (
        "tcp://127.0.0.1:5999", "old-camera", 2, 80,
    )


def test_fallback_is_per_option_not_per_configuration(stream, monkeypatch):
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM", "6000")
    monkeypatch.setenv("SOMA_EGO_STREAM_KEY", "existing-camera")
    monkeypatch.setenv("SOMA_EGO_STREAM_STRIDE", "4")
    assert stream.CapturePublisher.maybe().config == (
        "tcp://127.0.0.1:6000", "existing-camera", 4, 85,
    )


@pytest.mark.parametrize("value", ["", "   "])
def test_explicit_empty_current_endpoint_disables_legacy_stream(stream, monkeypatch, value):
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM", value)
    monkeypatch.setenv("SOMA_EGO_STREAM", "5999")
    assert stream.EgoFramePublisher.maybe() is None


@pytest.mark.parametrize("value, expected", [
    ("5557", "tcp://127.0.0.1:5557"),
    ("127.0.0.1:5558", "tcp://127.0.0.1:5558"),
    ("tcp://*:5559", "tcp://*:5559"),
])
def test_endpoint_forms_and_default_options(stream, monkeypatch, value, expected):
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM", value)
    assert stream.CapturePublisher.maybe().config == (expected, "ego_view", 1, 85)


def test_empty_current_key_does_not_restore_legacy_key(stream, monkeypatch):
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM", "6000")
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM_KEY", "")
    monkeypatch.setenv("SOMA_EGO_STREAM_KEY", "old-camera")
    assert stream.CapturePublisher.maybe().config[1] == ""


@pytest.mark.parametrize("option", ["STRIDE", "QUALITY"])
def test_invalid_current_numeric_option_never_falls_back_to_legacy(stream, monkeypatch, option):
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM", "6000")
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM_" + option, "")
    monkeypatch.setenv("SOMA_EGO_STREAM_" + option, "2")
    assert stream.CapturePublisher.maybe() is None


def test_current_option_overrides_invalid_legacy_setting(stream, monkeypatch):
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM", "6000")
    monkeypatch.setenv("LOCOMANILAB_EGO_STREAM_STRIDE", "3")
    monkeypatch.setenv("SOMA_EGO_STREAM_STRIDE", "not-an-integer")
    assert stream.CapturePublisher.maybe().config[2] == 3
