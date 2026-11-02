"""Check resource selection without importing Kimodo, Torch, or the simulator."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
H6 = ROOT / "rl/kimodo/heft/scripts/h6_kimodo_empty.py"
H7 = ROOT / "rl/kimodo/heft/scripts/h7_scene_rollout.py"
BRIDGE = ROOT / "rl/kimodo/heft_bridge.py"
EXPORT = ROOT / "rl/kimodo/heft/export.py"
HEFT_CKPT = ROOT / "rl/kimodo/heft/heft_ckpt.py"


def isolated_definitions(path, *names):
    """Execute only the selected definitions, excluding heavyweight module imports."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {"os": os, "Path": Path, "sys": sys, "KIMODO_FPS": 30.0, "HEFT_FPS": 50.0}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def checkout(tmp_path, name):
    work = tmp_path / name
    (work / "kimodo_rl").mkdir(parents=True)
    cache = work / "assets/text_cache.pt"
    cache.parent.mkdir()
    cache.write_bytes(b"existing-cache")
    return work, cache


def test_actor_cache_defaults_to_selected_checkout_and_honors_overrides(tmp_path, monkeypatch):
    resolve = isolated_definitions(H6, "resolve_actor_resources")["resolve_actor_resources"]
    work, cache = checkout(tmp_path, "env-work")
    explicit_work, explicit_cache = checkout(tmp_path, "explicit-work")
    monkeypatch.setenv("HEFT_WORK", str(work))
    monkeypatch.delenv("KIMODO_TEXT_CACHE", raising=False)
    assert resolve() == (work, cache)
    monkeypatch.setenv("KIMODO_TEXT_CACHE", str(explicit_cache))
    assert resolve() == (work, explicit_cache)
    assert resolve(explicit_work, cache) == (explicit_work, cache)


def test_actor_missing_resources_fail_before_loading_models(tmp_path, monkeypatch):
    namespace = isolated_definitions(H6, "resolve_actor_resources", "build_actor")
    monkeypatch.delenv("HEFT_WORK", raising=False)
    monkeypatch.delenv("KIMODO_TEXT_CACHE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="Set HEFT_WORK"):
        namespace["build_actor"](None, 6)
    with pytest.raises(ValueError, match="No kimodo_rl package"):
        namespace["resolve_actor_resources"](tmp_path)
    work, cache = checkout(tmp_path, "work")
    cache.unlink()
    with pytest.raises(ValueError, match="Kimodo text cache not found"):
        namespace["resolve_actor_resources"](work)


def test_actor_loads_the_explicit_cache_from_the_selected_checkout(tmp_path, monkeypatch):
    namespace = isolated_definitions(H6, "resolve_actor_resources", "build_actor")
    work, cache = checkout(tmp_path, "selected-work")
    other_work, _ = checkout(tmp_path, "other-work")
    monkeypatch.setenv("HEFT_WORK", str(other_work))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    loaded = []
    package = ModuleType("kimodo_rl")
    package.__path__ = []
    actor_module = ModuleType("kimodo_rl.actor_kimodo")
    actor_module.KimodoActorConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    actor_module.KimodoActor = lambda gen, config, text_cache: SimpleNamespace(
        gen=gen, config=config, text_cache=text_cache
    )
    gen_module = ModuleType("kimodo_rl.gen")
    gen_module.KimodoG1Generator = lambda **kwargs: SimpleNamespace(**kwargs)
    cache_module = ModuleType("kimodo_rl.text_cache")
    cache_module.TextCache = SimpleNamespace(load=lambda path: loaded.append(path) or "selected-cache")
    for name, module in {
        "torch": ModuleType("torch"),
        "kimodo_rl": package,
        "kimodo_rl.actor_kimodo": actor_module,
        "kimodo_rl.gen": gen_module,
        "kimodo_rl.text_cache": cache_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    actor = namespace["build_actor"](None, 6, device="cpu", heft_work=work, text_cache=cache)
    assert loaded == [str(cache)]
    assert sys.path[0] == str(work)
    assert actor.text_cache == "selected-cache"
    assert actor.gen.device == "cpu"
    assert actor.config.num_denoising_steps == 6
    assert actor.config.finetune_mode == "full"


def test_scene_rollout_forwards_its_selected_checkout_and_cache():
    tree = ast.parse(H7.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build_actor"
    ]
    assert len(calls) == 1
    received = []
    namespace = {
        "build_actor": lambda *args, **kwargs: received.append((args, kwargs)),
        "args": SimpleNamespace(ckpt="model.pt", steps=6, text_cache="selected-cache.pt"),
        "heft_work": "/selected/checkout",
    }
    exec(compile(ast.Expression(calls[0]), str(H7), "eval"), namespace)
    assert received == [
        (("model.pt", 6), {"heft_work": "/selected/checkout", "text_cache": "selected-cache.pt"})
    ]


def test_reference_xml_requires_a_matching_explicit_resource(tmp_path, monkeypatch):
    resolve = isolated_definitions(BRIDGE, "resolve_reference_xml")["resolve_reference_xml"]
    monkeypatch.delenv("HEFT_G1_XML", raising=False)
    with pytest.raises(ValueError, match="HEFT_G1_XML.*g1_5.xml"):
        resolve()
    model = tmp_path / "g1_5.xml"
    model.write_text("<mujoco/>")
    monkeypatch.setenv("HEFT_G1_XML", str(model))
    assert resolve() == model
    other = tmp_path / "matching-copy.xml"
    other.write_text("<mujoco/>")
    assert resolve(other) == other
    with pytest.raises(FileNotFoundError, match="g1.xml is not interchangeable"):
        resolve(tmp_path / "missing.xml")


def test_reference_constructor_passes_selected_xml_to_mujoco(tmp_path, monkeypatch):
    namespace = isolated_definitions(BRIDGE, "resolve_reference_xml", "HeftReference")
    model = tmp_path / "g1_5.xml"
    model.write_text("<mujoco/>")
    monkeypatch.setenv("HEFT_G1_XML", str(tmp_path / "unrelated.xml"))
    loaded = []
    mujoco = ModuleType("mujoco")
    mujoco.MjModel = SimpleNamespace(
        from_xml_path=lambda path: loaded.append(path) or SimpleNamespace(nq=36)
    )
    mujoco.MjData = lambda model: SimpleNamespace()
    mujoco.mjtObj = SimpleNamespace(mjOBJ_BODY=1)
    mujoco.mj_name2id = lambda model, kind, name: 0
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    reference = namespace["HeftReference"](["pelvis"], xml_path=model)
    assert loaded == [str(model)]
    assert reference.body_names == ["pelvis"]


def test_export_requires_an_explicit_g1_backend_checkout(tmp_path, monkeypatch):
    namespace = isolated_definitions(EXPORT, "_ensure_g1_bridge")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    # Even a g1_bridge folder in the current directory must not replace an unset selection.
    (tmp_path / "g1_bridge").mkdir()
    namespace["G1_BACKEND"] = ""
    with pytest.raises(SystemExit, match="G1_BACKEND is unset") as error:
        namespace["_ensure_g1_bridge"]()
    assert "https://github.com/LocoManiLab/g1_backend" in str(error.value)
    namespace["G1_BACKEND"] = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="Set G1_BACKEND"):
        namespace["_ensure_g1_bridge"]()
    work = tmp_path / "selected-backend"
    (work / "g1_bridge").mkdir(parents=True)
    namespace["G1_BACKEND"] = "selected-backend"
    namespace["_ensure_g1_bridge"]()
    assert sys.path[0] == str(work)
    assert namespace["G1_BACKEND"] == str(work)


def test_missing_tracker_onnx_requests_an_existing_artifact(tmp_path, monkeypatch):
    namespace = isolated_definitions(HEFT_CKPT, "resolve_onnx")
    namespace.update(
        ENV_VAR="HEFT_POLICY_ONNX", DEFAULT_ARTIFACT_DIR=tmp_path, ONNX_NAME="heft_policy_loc.onnx"
    )
    monkeypatch.delenv("HEFT_POLICY_ONNX", raising=False)
    with pytest.raises(SystemExit, match="Set HEFT_POLICY_ONNX") as error:
        namespace["resolve_onnx"]()
    assert "export_policy" not in str(error.value)
    assert "/home/" not in str(error.value)
    model = tmp_path / "selected.onnx"
    model.write_bytes(b"existing-policy")
    monkeypatch.setenv("HEFT_POLICY_ONNX", str(model))
    assert namespace["resolve_onnx"]() == model
    explicit = tmp_path / "explicit.onnx"
    explicit.write_bytes(b"explicit-policy")
    assert namespace["resolve_onnx"](explicit) == explicit
