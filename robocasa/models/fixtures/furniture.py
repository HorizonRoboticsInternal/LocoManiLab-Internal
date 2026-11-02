"""
Living-room furniture as kitchen-format fixtures.

Furniture pieces (sofa, coffee table, TV stand, ...) are non-manipulable static
objects -- exactly like kitchen fixtures -- so they live in the fixture system
rather than as free objects. They reuse ``Stool``'s behaviour (a floor-standing
``WallAccessory`` that supports ``attach_to`` a floor + ``z_rot``), and their
meshes are loaded by path, just like every other fixture.

``FurnitureType`` mirrors ``FixtureType``: it lets living-room tasks claim a
piece of furniture by semantic type (``env.get_fixture(FurnitureType.COFFEE_TABLE)``)
instead of by concrete instance, the same way kitchen tasks reference counters /
stoves. ``furniture_is_type`` is the matching predicate, used by the living-room
``get_fixture`` lookup.

So objects can be sampled *onto* furniture by the same placement sampler kitchen
uses, ``Furniture`` exposes a top-surface reset region synthesized from the
asset's ``reg_bbox`` (the convert script only emits a full bounding box, whose
region resolves to the floor; ``get_reset_regions`` here lifts it to the top).
"""

from enum import IntEnum

import numpy as np

from robocasa.models.fixtures.accessories import Stool


class FurnitureType(IntEnum):
    """Semantic types for living-room furniture (peer of ``FixtureType``)."""

    SOFA = 101
    COFFEE_TABLE = 102
    TV_STAND = 103
    SIDE_TABLE = 104


def furniture_is_type(furniture, furniture_type):
    """Return True if ``furniture`` matches ``furniture_type``.

    Mirrors ``fixture_is_type`` but for the ``FurnitureType`` enum: each concrete
    class declares its type via the ``furniture_type`` attribute.
    """
    if not isinstance(furniture_type, FurnitureType):
        return False
    return getattr(furniture, "furniture_type", None) == furniture_type


class Furniture(Stool):
    """Generic floor-standing furniture fixture (welded mesh)."""

    furniture_type = None

    @property
    def nat_lang(self):
        return "furniture"

    def get_reset_region_names(self):
        return ("top",)

    def get_reset_regions(self, env=None, reset_region_names=None, z_range=None):
        """Top-surface region for placing objects on this furniture.

        Synthesized from the asset's ``reg_bbox``: the placement sampler positions
        objects at ``fixture.pos + [0, 0, offset_z]``, so we lift the region to the
        top of the bounding box. Ignores ``z_range`` (furniture tops sit below the
        kitchen counter band that the base default assumes).
        """
        bbox = self._regions.get("bbox", None)
        if bbox is None:
            return super().get_reset_regions(
                env=env, reset_region_names=reset_region_names
            )
        p0, px, py, pz = bbox["p0"], bbox["px"], bbox["py"], bbox["pz"]
        return {
            "top": {
                "offset": (
                    float(np.mean((p0[0], px[0]))),
                    float(np.mean((p0[1], py[1]))),
                    float(pz[2]),
                ),
                "size": (float(px[0] - p0[0]), float(py[1] - p0[1])),
                # open clearance above the surface (objects rest on an open top)
                "height": 0.5,
            }
        }


class Sofa(Furniture):
    furniture_type = FurnitureType.SOFA

    @property
    def nat_lang(self):
        return "sofa"


class CoffeeTable(Furniture):
    furniture_type = FurnitureType.COFFEE_TABLE

    @property
    def nat_lang(self):
        return "coffee table"


class TVStand(Furniture):
    furniture_type = FurnitureType.TV_STAND

    @property
    def nat_lang(self):
        return "TV stand"


class SideTable(Furniture):
    """A small side table / stool (a flat-topped surface to set objects on).
    Backed by a reused lightwheel stool mesh via the style config."""

    furniture_type = FurnitureType.SIDE_TABLE

    @property
    def nat_lang(self):
        return "side table"
