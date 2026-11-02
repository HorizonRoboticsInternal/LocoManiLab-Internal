"""Task-free TeleLab environment containing manipulation fixtures."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from robocasa.environments.telelab.telelab_env import TeleLabEnv
from robocasa.models.arenas.molmospaces_arena import inline_defaults
from robocasa.models.fixtures import (
    Fixture,
    FixtureType,
    Fridge,
    Shelf,
    fixture_is_type,
)
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


_ASSET_ROOT = Path(__file__).resolve().parents[2] / "models" / "assets"
_MANIPULATION_SCENE_XML = (
    _ASSET_ROOT / "scenes" / "teleai-lab" / "manipulation_scene.xml"
)
_FRIDGE_XML = _ASSET_ROOT / "objects" / "manual" / "fridge" / "fridge_1p5m.xml"
_SHELF_XML = _ASSET_ROOT / "objects" / "manual" / "shelf" / "shelf.xml"
_DESK_XML = _ASSET_ROOT / "objects" / "manual" / "desk" / "desk.xml"


# Target exterior dimensions (width, depth, height), in meters, measured for
# the physical TeleLab manipulation scene.  These scales are intentionally
# local to TeleLabMani so other consumers of the hand-authored fixture assets
# retain their existing dimensions.
_FRIDGE_SCALE = (
    0.52 / 0.633941756066,
    0.54 / 0.614683858722,
    1.60 / 1.501670622214,
)
# The desk mesh uses local Y as height and local Z as depth before its
# upright rotation in desk.xml.
_DESK_SCALE = (1.20 / 1.14, 0.75 / 0.739, 0.60 / 0.652)
# Shelf height is measured across its lowest and highest support boards.
_SHELF_SCALE = (1.00 / 0.9836, 1.50 / 1.4506, 0.40 / 0.495)
_SHELF_MIDDLE_SURFACE_HEIGHT = 0.72

_DESK_POS = (0.066553, 2.0055, 0.0)
# Keep the desk fixed and leave 90 cm between its positive-X collision edge
# and the refrigerator's negative-X collision edge.
_FRIDGE_POS = (1.825876, 2.011, 0.80)


def _materialize_fixture_defaults(source_xml):
    """Return a temporary XML that robosuite can parse as a fixture."""
    tree = ET.parse(source_xml)
    root = tree.getroot()
    inline_defaults(root)
    for geom in root.findall(".//geom"):
        if geom.get("group") is None:
            geom.set("group", "0")

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".telelab_fixture_",
        suffix=".xml",
        dir=os.path.dirname(source_xml),
        delete=False,
    )
    try:
        tree.write(handle, encoding="utf-8", xml_declaration=True)
    finally:
        handle.close()
    return handle.name


class TeleLabMani(TeleLabEnv):
    """Load the TeleLab room with its fridge, shelf, and desk fixtures."""

    def __init__(self, *args, scene_xml=_MANIPULATION_SCENE_XML, **kwargs):
        super().__init__(*args, scene_xml=scene_xml, **kwargs)

    def _load_model(self):
        super()._load_model()

        prepared_fridge_xml = _materialize_fixture_defaults(str(_FRIDGE_XML))
        try:
            fridge = Fridge(
                xml=prepared_fridge_xml,
                name="telelab_fridge",
                joints=None,
                scale=_FRIDGE_SCALE,
                # The authored refrigerator is centered at half its height.
                # Keep its base on the floor after increasing it to 1.60 m.
                pos=_FRIDGE_POS,
            )
        finally:
            os.unlink(prepared_fridge_xml)

        self.fixtures = {
            "fridge": fridge,
            "shelf": Shelf(
                xml=str(_SHELF_XML),
                name="telelab_shelf",
                scale=_SHELF_SCALE,
                pos=(-0.75, 1.20, 0.0),
            ),
            "desk": Fixture(
                xml=str(_DESK_XML),
                name="telelab_desk",
                joints=None,
                duplicate_collision_geoms=False,
                scale=_DESK_SCALE,
                pos=_DESK_POS,
            ),
        }

        self._set_shelf_middle_surface_height(self.fixtures["shelf"])

        for fixture in self.fixtures.values():
            self.model.merge_assets(fixture)
            self.model.worldbody.append(fixture.get_obj())
            self.model.mujoco_objects.append(fixture)

            # ManipulationTask.merge_objects intentionally omits these full
            # MJCF sections. Preserve them for articulated fixtures such as
            # the refrigerator doors and drawer.
            for section_name in (
                "actuator",
                "sensor",
                "tendon",
                "equality",
                "contact",
            ):
                source = getattr(fixture, section_name)
                target = getattr(self.model, section_name)
                for element in source:
                    target.append(deepcopy(element))

    @staticmethod
    def _set_shelf_middle_surface_height(shelf):
        """Set the second usable shelf surface to 72 cm above the floor."""
        middle_board = shelf.get_obj().find(
            f".//geom[@name='{shelf.naming_prefix}shelf_collision_board_2']"
        )
        if middle_board is None:
            raise RuntimeError("TeleLab shelf is missing its middle support board")

        position = string_to_array(middle_board.get("pos"))
        half_size = string_to_array(middle_board.get("size"))
        position[1] = _SHELF_MIDDLE_SURFACE_HEIGHT - half_size[1]
        middle_board.set("pos", array_to_string(position))

    def get_fixture(self, id, full_name_check=False):
        """Look up a fixture by instance, registered name, or FixtureType."""
        if isinstance(id, Fixture):
            return id
        if isinstance(id, str) and id in self.fixtures:
            return self.fixtures[id]

        if isinstance(id, (FixtureType, int)):
            matches = [
                fixture
                for fixture in self.fixtures.values()
                if fixture_is_type(fixture, id)
            ]
        elif isinstance(id, str):
            matches = [
                fixture
                for name, fixture in self.fixtures.items()
                if name == id or (not full_name_check and id in name)
            ]
        else:
            matches = []
        return matches[0] if matches else None

    def get_ep_meta(self):
        return {
            "lang": "freely explore the TeleLab manipulation scene",
            "scene_xml": str(self.scene_xml),
        }


__all__ = ["TeleLabMani"]
