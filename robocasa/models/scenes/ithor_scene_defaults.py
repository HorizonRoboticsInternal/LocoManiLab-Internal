"""Default furniture replacements shared by scene loading and semantics."""

import copy
from pathlib import Path
import xml.etree.ElementTree as ET


BEDROOM_301_OLD_SEAT_BODY = "pad_40f0405be70264b3ffe8b32e9347537b_1_0_0"
BEDROOM_301_SOFA_BODY = "bedroom_301_sofa"
BEDROOM_301_SOFA_ID = "FloorPlan301:Armchair_201_1_19"
_SOFA_ASSET = "Living Room Objects/Armchair/Prefabs/Armchair_201_1/Armchair_201_1.xml"


def apply_default_scene_furniture(root, scene_xml):
    """Replace the 301 ottoman with a fixed, full-size single-seat sofa."""
    if Path(scene_xml).name != "FloorPlan301_physics.xml":
        return

    # Reuse the loader's class expansion and geom conventions for this asset.
    from robocasa.models.arenas.molmospaces_arena import (
        inline_defaults,
        remap_geom_groups,
    )

    worldbody = root.find("worldbody")
    old_seat = worldbody.find(f"body[@name='{BEDROOM_301_OLD_SEAT_BODY}']")
    if old_seat is None:
        raise ValueError("FloorPlan301: missing default seat to replace")
    asset_path = Path(scene_xml).parent.parent.parent / "objects/thor" / _SOFA_ASSET
    sofa = ET.parse(asset_path).getroot()
    inline_defaults(sofa)
    remap_geom_groups(sofa)

    # Prefix the asset names to avoid collisions with the room's materials.
    names = {
        element.get("name"): f"{BEDROOM_301_SOFA_BODY}_{element.get('name')}"
        for element in sofa.iter()
        if element.get("name")
    }
    for element in sofa.iter():
        for key in ("name", "mesh", "material", "texture"):
            if element.get(key) in names:
                element.set(key, names[element.get(key)])
        if element.get("file"):
            element.set(
                "file", str((asset_path.parent / element.get("file")).resolve())
            )
    root.find("asset").extend(sofa.find("asset"))

    body = sofa.find("worldbody/body")
    # Keep receptacle sites for semantics without rendering their debug boxes.
    for site in body.iter("site"):
        site.set("rgba", "1 0 0 0")
        site.set("group", "5")
    body.set("name", BEDROOM_301_SOFA_BODY)
    # Local Y is up; local +Z faces into the room (+world Y).
    body.set("pos", "0.93 -0.99 0.461151517")
    body.set("quat", "0 0 0.7071067811865476 0.7071067811865476")
    body.remove(body.find("joint"))
    worldbody.remove(old_seat)
    worldbody.append(body)


def apply_default_scene_metadata(metadata, scene_name):
    """Describe the same replacement in every task's semantic registry."""
    if scene_name != "FloorPlan301":
        return
    objects = metadata["objects"]
    old_seat = objects.pop(BEDROOM_301_OLD_SEAT_BODY)
    sofa = copy.deepcopy(old_seat)
    sofa.update(
        hash_name=BEDROOM_301_SOFA_BODY,
        asset_id="Armchair_201_1",
        object_id=BEDROOM_301_SOFA_ID.split(":", 1)[1],
        category="Sofa",
        is_static=True,
        name_map={
            "bodies": {
                BEDROOM_301_SOFA_BODY: "Armchair_201_1",
                f"{BEDROOM_301_SOFA_BODY}_Armchair_201_1_armchair_201": "Armchair_201_1_armchair_201",
            },
            "joints": {},
            "sites": {
                f"{BEDROOM_301_SOFA_BODY}_Armchair_201_1_armchair_201_ReceptacleCollider_0": "Armchair_201_1_armchair_201_ReceptacleCollider_0",
            },
        },
    )
    objects[BEDROOM_301_SOFA_BODY] = sofa
