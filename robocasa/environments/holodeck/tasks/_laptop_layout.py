"""Reusable calibrated Laptop_26 layout for Holodeck desk tasks."""

import os

import numpy as np

import robocasa
from robocasa.models.scenes.holodeck_task_specs import (
    get_holodeck_default_asset_layout,
    get_holodeck_task_spec,
)


LAPTOP_ASSET = (
    "objects/thor/Living Room and Bedroom Objects/Laptop_Full/"
    "Prefabs/Laptop_26/Laptop_26.xml"
)
CALIBRATED_LAPTOP_SCENES = ("train_1664", "train_2025")
# The 1664 desk has too little usable top area for both the calibrated laptop
# and the jar workflow. Jar tasks opt into this scene-specific removal below.
LAPTOP_CLEARED_FOR_JAR_SCENES = ("train_1664",)


def laptop_is_cleared_for_task(env):
    """Return whether this task intentionally omits the calibrated laptop."""
    cleared_scenes = getattr(
        env, "LAPTOP_CLEARED_SCENES", ()
    )
    return env.scene_name in cleared_scenes


def get_open_laptop_layout(scene_name):
    """Return the shared Laptop_26 layout for scenes that require it."""
    if scene_name not in CALIBRATED_LAPTOP_SCENES:
        return None
    return get_holodeck_task_spec(
        scene_name,
        "open_laptop",
        required=(
            "fixture_mode",
            "source_body",
            "desk_body",
            "laptop_pose",
            "laptop_support_geom",
        ),
    )


def get_default_laptop_layout(scene_name):
    """Return the scene-configured stable laptop layout, if any."""
    layout_task = get_holodeck_default_asset_layout(scene_name)
    if layout_task is None:
        return None
    if layout_task != "open_laptop":
        raise ValueError(
            f"{scene_name}: unsupported default asset layout {layout_task!r}"
        )
    return get_open_laptop_layout(scene_name)


def replace_baked_laptop(env, layout):
    """Remove the embedded source laptop before the scene is compiled."""
    if layout is None:
        return
    if layout["fixture_mode"] != "spawn_asset":
        raise ValueError(
            f"{env.scene_name}: open_laptop must use fixture_mode=spawn_asset"
        )
    source_body = layout["source_body"]
    removed = env.remove_scene_bodies((source_body,))
    if source_body not in removed:
        raise RuntimeError(
            f"{env.scene_name}: calibrated laptop source is missing: {source_body}"
        )


def spawn_calibrated_laptop(env, layout):
    """Create the same supported Laptop_26 fixture used by OpenLaptop."""
    if layout is None or laptop_is_cleared_for_task(env):
        return None
    support = layout["laptop_support_geom"]
    env.add_scene_body_geom(
        layout["desk_body"],
        name=f"{env.scene_name}_laptop_support",
        type="box",
        pos=" ".join(str(float(value)) for value in support["pos"]),
        size=" ".join(str(float(value)) for value in support["size"]),
        contype="8",
        conaffinity="15",
        group="0",
        friction="0.9 0.9 0.001",
    )
    pose = layout["laptop_pose"]
    return env.create_scene_fixture(
        xml_path=os.path.join(robocasa.models.assets_root, LAPTOP_ASSET),
        pos=np.asarray(pose["pos"], dtype=float),
        quat=np.asarray(pose["quat"], dtype=float),
        name="spawned_laptop",
        category="Laptop",
        source_category="Laptop",
        asset_id="Laptop_26",
        front=np.asarray(layout.get("laptop_front", [0.0, -1.0])),
        # Laptop_26 has a free base and a hinged lid. The source asset omits
        # explicit inertials, so retain the fixture helper's inertial repair
        # before MuJoCo compiles the moving bodies.
        ensure_inertials=True,
    )
