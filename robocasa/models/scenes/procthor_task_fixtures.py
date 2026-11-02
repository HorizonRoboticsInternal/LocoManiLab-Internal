"""Generic task-local fixture assembly for baked ProcTHOR scenes.

ProcTHOR homes arrive as baked MJCF arenas.  Some tasks still need to add a
small amount of furniture or a target receptacle.  Keep that operation
Kitchen-like: tasks provide fixture configs, and this module translates those
configs into MJCF without task files touching XML.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.models.scenes.procthor_scene_builder import ProcTHORFixtureRecord
from robocasa.models.scenes.procthor_yaml import config_path
from robocasa.models.scenes.procthor_yaml import load_yaml_config


_MODEL_ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets")
)
_OBJECT_ASSET_ROOT = os.path.join(_MODEL_ASSET_ROOT, "objects")
_TASK_FIXTURE_ROOT = os.path.join(_MODEL_ASSET_ROOT, "scenes", "procthor_task_fixtures")


def task_fixture_config_path(name):
    return config_path(_TASK_FIXTURE_ROOT, name)


def load_task_fixture_configs(name, **variables):
    """Load task fixture configs from a YAML file under scene assets."""
    variables.setdefault("asset_root", _OBJECT_ASSET_ROOT)
    data = load_yaml_config(_TASK_FIXTURE_ROOT, name, **variables)
    return data.get("fixtures", [])


def _fmt(values):
    if isinstance(values, str):
        return values
    if np.isscalar(values):
        return f"{float(values):.4f}"
    return " ".join(f"{float(v):.4f}" for v in values)


def _attrs(attrs):
    out = {}
    for key, value in attrs.items():
        if value is None:
            continue
        out[key] = (
            _fmt(value) if isinstance(value, (list, tuple, np.ndarray)) else str(value)
        )
    return out


def _ensure_child(parent, tag, unique_attr="name", **attrs):
    unique_value = attrs.get(unique_attr)
    if unique_value is not None:
        existing = parent.find(f"./{tag}[@{unique_attr}='{unique_value}']")
        if existing is not None:
            return existing
    return ET.SubElement(parent, tag, **_attrs(attrs))


def _expand_meshes(asset, mesh_cfgs):
    mesh_sets = {}
    for cfg in mesh_cfgs:
        scale = cfg.get("scale", "1 1 1")
        set_name = cfg.get("set")
        names = []

        if "file_glob" in cfg:
            for path in sorted(glob.glob(cfg["file_glob"])):
                stem = os.path.splitext(os.path.basename(path))[0]
                name = cfg.get("name_template", "{prefix}_{stem}").format(
                    prefix=cfg.get("name_prefix", ""),
                    stem=stem,
                )
                _ensure_child(asset, "mesh", name=name, file=path, scale=scale)
                names.append(name)
        else:
            name = cfg["name"]
            _ensure_child(asset, "mesh", name=name, file=cfg["file"], scale=scale)
            names.append(name)

        if set_name is not None:
            mesh_sets.setdefault(set_name, []).extend(names)
    return mesh_sets


def _add_geoms(parent, geom_cfgs, mesh_sets):
    for cfg in geom_cfgs:
        if "for_each_mesh_set" not in cfg:
            _ensure_child(parent, "geom", **cfg)
            continue

        mesh_set = mesh_sets[cfg["for_each_mesh_set"]]
        geom_attrs = {
            k: v
            for k, v in cfg.items()
            if k not in {"for_each_mesh_set", "name_template"}
        }
        for mesh_name in mesh_set:
            name = cfg.get("name_template", "{mesh}").format(mesh=mesh_name)
            _ensure_child(parent, "geom", name=name, mesh=mesh_name, **geom_attrs)


def _add_body(parent, body_cfg, mesh_sets, body_names):
    body_attrs = {k: v for k, v in body_cfg.items() if k not in {"bodies", "geoms"}}
    body = _ensure_child(parent, "body", **body_attrs)
    if body.get("name") is not None:
        body_names.append(body.get("name"))

    _add_geoms(body, body_cfg.get("geoms", []), mesh_sets)
    for child_cfg in body_cfg.get("bodies", []):
        _add_body(body, child_cfg, mesh_sets, body_names)
    return body


def add_task_fixture(env, cfg):
    """Add one task fixture from a generic MJCF fixture config.

    Required keys:
        ``name``: fixture name and default root body name.
        ``fixture_type``: semantic type recorded in ``env.fixtures``.
        ``body``: nested body/geom config to append to ``worldbody``.

    Optional keys:
        ``materials``: list of MJCF material attrs.
        ``meshes``: list of MJCF mesh attrs, with optional ``file_glob``.
        ``asset_id`` and ``pos``: metadata stored in the fixture record.
    """
    name = cfg["name"]
    worldbody = env.mujoco_arena.worldbody
    if worldbody.find(f"./body[@name='{name}']") is not None:
        return env.fixtures.get(name)

    asset = env.mujoco_arena.asset
    for material_cfg in cfg.get("materials", []):
        _ensure_child(asset, "material", **material_cfg)
    mesh_sets = _expand_meshes(asset, cfg.get("meshes", []))

    body_names = []
    root_body = _add_body(worldbody, cfg["body"], mesh_sets, body_names)
    root_body_name = root_body.get("name")
    record = ProcTHORFixtureRecord(
        name=name,
        fixture_type=cfg["fixture_type"],
        body_name=root_body_name,
        asset_id=cfg.get("asset_id"),
        pos=np.array(cfg["pos"], dtype=float) if "pos" in cfg else None,
        body_names=body_names,
    )
    env.fixtures[name] = record
    return record
