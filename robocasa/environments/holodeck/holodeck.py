"""Holodeck-Objaverse scene environment.

Holodeck train exports use the same baked MJCF/JSON/metadata contract as the
existing ProcTHOR loader, but are kept in a separate scene family and registry.
"""

import os
import re

import robocasa
from robocasa.environments.procthor.procthor import (
    REGISTERED_PROCTHOR_ENVS,
    ProcTHOREnv,
    ProcTHOREnvMeta,
)
from robosuite.environments.base import REGISTERED_ENVS


REGISTERED_HOLODECK_ENVS = {}


class HolodeckEnvMeta(ProcTHOREnvMeta):
    """Register Holodeck classes separately while preserving robosuite registration."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        # ProcTHOREnvMeta also writes into the ProcTHOR registry.  Holodeck
        # classes belong only to the Holodeck catalog.
        REGISTERED_PROCTHOR_ENVS.pop(cls.__name__, None)
        if class_dict.get("ABSTRACT", False):
            # The parent EnvMeta registers every class globally.  Keep
            # implementation bases out of both public environment registries.
            REGISTERED_ENVS.pop(cls.__name__, None)
        else:
            REGISTERED_HOLODECK_ENVS[cls.__name__] = cls
        return cls


_SCENE_RE = re.compile(r"(?:train[_-])(\d+)$")


def get_holodeck_scenes_dir():
    return os.path.join(
        robocasa.models.assets_root, "scenes", "holodeck-objaverse-train"
    )


def list_holodeck_scenes():
    root = get_holodeck_scenes_dir()
    if not os.path.isdir(root):
        return {}
    return {
        f"train_{match.group(1)}": os.path.join(root, filename)
        for filename in os.listdir(root)
        if (match := re.fullmatch(r"train[_-](\d+)\.xml", filename))
    }


def normalize_holodeck_scene_name(scene_id):
    if isinstance(scene_id, int):
        return f"train_{scene_id}"
    match = _SCENE_RE.fullmatch(str(scene_id))
    if match is None:
        raise ValueError(f"not a Holodeck scene name/id: {scene_id!r}")
    return f"train_{match.group(1)}"


def get_holodeck_scene_xml(scene_id):
    name = normalize_holodeck_scene_name(scene_id)
    path = os.path.join(get_holodeck_scenes_dir(), f"{name}.xml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Holodeck scene is not downloaded: {path}")
    return path


class HolodeckEnv(ProcTHOREnv, metaclass=HolodeckEnvMeta):
    """Free-roam Holodeck environment using the ProcTHOR scene machinery."""

    ABSTRACT = True
    SCENE_FAMILY = "Holodeck-Objaverse train"
    # Scene defaults may be opted out by tasks with a deliberately different
    # tabletop layout, such as a cup-only or jar-only workflow.
    USE_DEFAULT_ASSET_LAYOUT = True
    LAPTOP_CLEARED_SCENES = ()
    # Permanently omit these exact bodies from all Holodeck environment loads.
    # Keep downloaded assets intact and scope cleanup to the named scenes.
    HIDDEN_SCENE_BODIES = {
        "train_1018": ("obja_528a83a41ce547199fc1be3bfec5c0f6_3_0_0",),
        "train_2209": (
            "statue_59b4e93be5e649fe613e418b442aec0b_2_0_0",
            "obja_e55514b2057c4ad7853722d0cdcf52f7_1_0_0",
        ),
    }
    NOISE_FREE_ROBOT_SPAWN = False

    def __init__(self, *args, **kwargs):
        if self.NOISE_FREE_ROBOT_SPAWN:
            kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, **kwargs)

    def _setup_baked_scene_fixtures(self):
        self._default_asset_layout = None
        if self.USE_DEFAULT_ASSET_LAYOUT:
            from robocasa.environments.holodeck.tasks._laptop_layout import (
                get_default_laptop_layout,
                replace_baked_laptop,
            )

            self._default_asset_layout = get_default_laptop_layout(self.scene_name)
            replace_baked_laptop(self, self._default_asset_layout)
        hidden = self.HIDDEN_SCENE_BODIES.get(self.scene_name, ())
        if hidden:
            removed = self.remove_scene_bodies(hidden)
            missing = set(hidden) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured hidden scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        if self._default_asset_layout is not None:
            from robocasa.environments.holodeck.tasks._laptop_layout import (
                spawn_calibrated_laptop,
            )

            self.laptop_fixture = spawn_calibrated_laptop(
                self, self._default_asset_layout
            )
        super()._setup_task_fixtures()

    def _list_downloaded_scenes(self):
        return list_holodeck_scenes()

    def _scene_ids_to_names(self, scene_ids, downloaded):
        ids = [scene_ids] if isinstance(scene_ids, (int, str)) else list(scene_ids)
        names = [normalize_holodeck_scene_name(scene_id) for scene_id in ids]
        missing = [name for name in names if name not in downloaded]
        if missing:
            raise ValueError(
                f"Holodeck scenes are not downloaded: {missing}; "
                f"available: {sorted(downloaded)}"
            )
        return names

    def _get_scene_xml(self, scene_name):
        # Explicitly select train_117.xml; the optional *_ceiling.xml is not a
        # runnable replacement and must never be selected by globbing.
        return get_holodeck_scene_xml(scene_name)


__all__ = [
    "HolodeckEnv",
    "REGISTERED_HOLODECK_ENVS",
    "get_holodeck_scene_xml",
    "list_holodeck_scenes",
]
