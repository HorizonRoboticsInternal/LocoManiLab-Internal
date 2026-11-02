"""CPU-only download selection; no simulator, network, or weight payload is used."""
import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "robocasa/scripts"
PLANNER = "sonic/low_latency/g1/planner_sonic.onnx"


@pytest.fixture
def helpers(monkeypatch):
    # The downloader's small UI dependencies are irrelevant to file selection.
    color = ModuleType("termcolor")
    color.colored = lambda text, *args, **kwargs: text
    progress = ModuleType("tqdm")
    progress.tqdm = type("Progress", (), {})
    monkeypatch.setitem(sys.modules, "termcolor", color)
    monkeypatch.setitem(sys.modules, "tqdm", progress)
    loaded = {}
    for name in ("download_g1_checkpoints", "download_kitchen_assets"):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(weights=loaded["download_g1_checkpoints"],
                           kitchen=loaded["download_kitchen_assets"])


def hub_listing(monkeypatch, paths):
    hub = ModuleType("huggingface_hub")
    hub.HfApi = lambda: SimpleNamespace(list_repo_tree=lambda *args, **kwargs: [
        SimpleNamespace(path="checkpoint/controller/" + path, size=10) for path in paths])
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_v1_1_selects_its_weights_and_only_the_shared_planner(helpers, monkeypatch, tmp_path):
    own = [f"sonic/v1_1/g1/model_{part}.onnx" for part in ("encoder", "decoder")]
    unrelated = ["sonic/low_latency/g1/model_encoder.onnx", "amo/policy.pt",
                 "heft/pmg/policy.onnx", "sonic/v1_1/g1/policy.yaml"]
    hub_listing(monkeypatch, own + [PLANNER] + unrelated)
    selected, skipped, size = helpers.weights.plan("test", "pin", ["sonic_v1_1"], tmp_path, False)
    assert set(selected) == set(own + [PLANNER])
    assert skipped == [] and size == 30


def test_heft_does_not_request_sonic_or_other_controllers(helpers, monkeypatch, tmp_path):
    hub_listing(monkeypatch, ["heft/pmg/policy.onnx", "heft/pmg/policy.yaml", PLANNER, "amo/policy.pt"])
    selected, skipped, size = helpers.weights.plan("test", "pin", ["heft"], tmp_path, False)
    assert selected == ["heft/pmg/policy.onnx"] and skipped == [] and size == 10


def test_existing_planner_is_preserved_and_missing_remote_dependency_is_reported(helpers, monkeypatch, tmp_path):
    own = "sonic/v1_1/g1/model_encoder.onnx"
    hub_listing(monkeypatch, [own])
    with pytest.raises(ValueError, match="Missing shared controller weights"):
        helpers.weights.plan("test", "pin", ["sonic_v1_1"], tmp_path, False)
    local = tmp_path / PLANNER
    local.parent.mkdir(parents=True)
    local.write_bytes(b"existing controller")
    hub_listing(monkeypatch, [own, PLANNER])
    selected, skipped, _ = helpers.weights.plan("test", "pin", ["sonic_v1_1"], tmp_path, False)
    assert selected == [own] and skipped == [PLANNER]
    assert local.read_bytes() == b"existing controller"


def test_offline_weight_plan_never_probes_runtime_or_hub(helpers, monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("offline plan attempted runtime or network discovery")
    monkeypatch.setattr(helpers.weights, "plan", forbidden)
    monkeypatch.setattr(helpers.weights, "warn_if_backends_will_miss_it", forbidden)
    destination = tmp_path / "not-created"
    helpers.weights.download_checkpoints(["sonic_v1_1"], dest=destination, plan_only=True)
    assert not destination.exists()
    output = capsys.readouterr().out
    assert PLANNER in output and "sonic/v1_1/g1" in output
    assert "low_latency/g1/model_encoder" not in output


def test_kitchen_plan_only_lists_selected_packs_without_prompt_or_download(helpers, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("offline asset plan attempted input or download")
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr(helpers.kitchen, "download_and_extract_zip", forbidden)
    helpers.kitchen.download_kitchen_assets(["tex", "objs_objaverse"], dry_run=True)
    output = capsys.readouterr().out
    assert "tex:" in output and "objs_objaverse:" in output
    assert "fixtures_lw:" not in output and "objs_lw:" not in output


@pytest.mark.parametrize("selection,count", [(["tex", "objs_objaverse"], 2), (["all"], 6)])
def test_kitchen_download_keeps_the_explicit_selection(helpers, monkeypatch, selection, count):
    selected = []
    monkeypatch.setattr("builtins.input", lambda *args: "y")
    monkeypatch.setattr(helpers.kitchen, "download_and_extract_zip", lambda **config: selected.append(config))
    helpers.kitchen.download_kitchen_assets(selection)
    assert len(selected) == count
    if "all" not in selection:
        assert selected == [helpers.kitchen.DOWNLOAD_ASSET_REGISTRY[key] for key in selection]


@pytest.mark.parametrize("script,options", [
    ("download_g1_checkpoints.py", []),
    ("download_g1_checkpoints.py", ["--family", "heft", "--all"]),
    ("download_kitchen_assets.py", []),
])
def test_missing_or_ambiguous_selection_exits_before_network(helpers, monkeypatch, script, options):
    monkeypatch.setattr(sys, "argv", [script, *options])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    assert error.value.code == 2


@pytest.mark.parametrize("script,options,expected", [
    ("download_g1_checkpoints.py", ["--family", "sonic_v1_1", "--plan"], PLANNER),
    ("download_kitchen_assets.py", ["--type", "tex", "--dry-run"], "tex:"),
])
def test_offline_cli_plan_runs_without_importing_the_simulator(helpers, monkeypatch, tmp_path, capsys,
                                                              script, options, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("offline CLI plan attempted a network request or prompt")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setitem(sys.modules, "robocasa", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    if script == "download_g1_checkpoints.py":
        options = options + ["--dest", str(tmp_path / "not-created")]
    monkeypatch.setattr(sys, "argv", [script, *options])
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    assert expected in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
