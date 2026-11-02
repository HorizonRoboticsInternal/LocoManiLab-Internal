"""
Living-room furniture builder -- the living-room analogue of
``scene_builder.create_fixtures``.

Turns a (layout_config, style_config) pair into instantiated, welded furniture
fixtures. Kept deliberately simple compared to the kitchen builder (no
group/alignment/stacking/auxiliary logic): each layout entry names a furniture
type + planar pose, and the style says which mesh asset to use for that type.

``FURNITURE`` is the type-string -> class map (the living-room peer of
``scene_builder.FIXTURES``). Add new furniture types here.
"""

from robocasa.models.fixtures import (
    Sofa,
    CoffeeTable,
    TVStand,
    SideTable,
    Blender,
    Oven,
    Dishwasher,
    ArticulatedObject,
    Microwave,
    ToasterOven,
    Toaster,
    StandMixer,
    ElectricKettle,
    CoffeeMachine,
    Stove,
    Fridge,
)
from robocasa.models.fixtures.windows import Window

# furniture type string (used in layout/style yaml) -> fixture class
FURNITURE = dict(
    sofa=Sofa,
    coffee_table=CoffeeTable,
    tv_stand=TVStand,
    side_table=SideTable,
    blender=Blender,
    oven=Oven,
    dishwasher=Dishwasher,
    microwave_midea=Microwave,
    green_folder=ArticulatedObject,
    toaster_oven=ToasterOven,
    toaster=Toaster,
    stand_mixer=StandMixer,
    electric_kettle=ElectricKettle,
    coffee_machine=CoffeeMachine,
    stove=Stove,
    fridge=Fridge,
)


def _resting_z(fixture):
    """Z at which the fixture sits on the floor (= lift to align bottom with z=0).

    Tries ``bbox`` (furniture meshes), then ``main`` (kitchen fixtures), then
    falls back to ``bottom_offset``.  p0[2] is the z-coordinate of the
    bottom corner of the region — negative when the fixture origin is above
    the bottom, so ``-p0[2]`` is the lift needed.
    """
    for key in ("bbox", "main"):
        reg = fixture._regions.get(key, None)
        if reg is not None:
            return float(-reg["p0"][2])
    # last resort: the ``bottom_offset`` property (may raise ValueError)
    return float(-fixture.bottom_offset[2])


def create_furniture(layout_config, style_config, rng=None):
    """Instantiate furniture fixtures for a (layout, style) pair.

    Args:
        layout_config (dict): parsed layout yaml -- ``furniture`` is a list of
            ``{name, type, pos: [x, y], z_rot}`` entries.
        style_config (dict): parsed style yaml -- ``furniture`` maps each type to
            a mesh asset path (relative to the asset root).
        rng: unused for now (kept for parity with ``create_fixtures``).

    Returns:
        dict: name -> furniture ``Fixture``.
    """
    asset_for_type = style_config["furniture"]

    furniture = {}
    for entry in layout_config["furniture"]:
        ftype = entry["type"]
        cls = FURNITURE[ftype]
        xml = asset_for_type[ftype]
        x, y = entry["pos"]
        z_rot = entry.get("z_rot", 0.0)

        # --- window: mount at wall height, not on the floor ---
        if ftype == "window":
            z = entry.get("z_height", 0.85)  # default: eye-level on 1.5m wall
            fxtr = cls(name=entry["name"], xml=xml, pos=(x, y, z))
            if z_rot:
                fxtr.set_euler([0.0, 0.0, float(z_rot)])
        else:
            # --- floor-standing fixtures ---
            fxtr = cls(name=entry["name"], xml=xml, pos=(x, y, 0.0))
            fxtr.set_pos((x, y, _resting_z(fxtr)))
            if z_rot:
                fxtr.set_euler([0.0, 0.0, float(z_rot)])

        furniture[entry["name"]] = fxtr
    return furniture
