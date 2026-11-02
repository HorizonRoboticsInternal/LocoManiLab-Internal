"""Shared YAML helpers for ProcTHOR scene/task configs."""

from __future__ import annotations

import os

import yaml


class TemplateVars(dict):
    """Keep unknown format variables intact during partial templating."""

    def __missing__(self, key):
        return "{" + key + "}"


def config_path(root, name):
    """Resolve a YAML config name under ``root``."""
    if os.path.isabs(name):
        return name
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    return os.path.join(root, name)


def template(value, variables):
    """Recursively format YAML-loaded data with partial variable support."""
    if isinstance(value, str):
        return value.format_map(TemplateVars(variables))
    if isinstance(value, list):
        return [template(v, variables) for v in value]
    if isinstance(value, tuple):
        return tuple(template(v, variables) for v in value)
    if isinstance(value, dict):
        return {k: template(v, variables) for k, v in value.items()}
    return value


def load_yaml_config(root, name, **variables):
    with open(config_path(root, name), "r") as f:
        data = yaml.safe_load(f) or {}
    return template(data, variables)
