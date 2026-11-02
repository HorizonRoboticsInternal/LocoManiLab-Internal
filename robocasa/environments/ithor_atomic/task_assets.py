"""Canonical fixed assets shared by Bedroom and Bathroom benchmark tasks."""

import os

from robocasa.models import assets_root


CAN_MODEL = "/objects/objaverse/can/can_0/model.xml"
CANDLE_MODEL = "/objects/objaverse/candle/candle_10/model.xml"
SOAP_MODEL = "/objects/objaverse/bar_soap/bar_soap_0/model.xml"
SPONGE_MODEL = "/objects/objaverse/sponge/sponge_0/model.xml"
MUG_MODEL = "/objects/objaverse/mug/mug_1/model.xml"
BOXED_FOOD_MODEL = "/objects/objaverse/boxed_food/boxed_food_0/model.xml"
BOWL_MODEL = "/objects/objaverse/bowl/bowl_14/model.xml"
BASKET_MODEL = "/objects/lightwheel/basket/Basket034/model.xml"
LIFT_BASKET_MODEL = "/objects/lightwheel/basket/Basket039/model.xml"


# Small hand-held objects must present a span greater than 7 cm and no more
# than 8 cm across the hand's closing direction. 7.5 cm is the canonical target
# for fixed small assets; containers and other large objects are calibrated
# separately.
MIN_GRASP_SPAN = 0.07
TARGET_GRASP_SPAN = 0.075
MAX_GRASP_SPAN = 0.08

# Native spans describe the useful closing direction, not the longest AABB
# edge. Keep all scales here so the same MJCF asset cannot silently change
# size between rooms or between atomic and composite tasks.
NATIVE_GRASP_SPANS = {
    CANDLE_MODEL: 0.0484761660,
    SOAP_MODEL: 0.0646494780,
    # The task sampler applies the Objaverse object's intrinsic 0.9 scale;
    # this is the measured compiled closing span at object_scale=1.0.
    SPONGE_MODEL: 0.0689451682,
    CAN_MODEL: 0.0690627916,
    BOXED_FOOD_MODEL: 0.0723618216,
    MUG_MODEL: 0.0785635518,
}
CANONICAL_ASSET_SCALES = {
    model: TARGET_GRASP_SPAN / span
    for model, span in NATIVE_GRASP_SPANS.items()
}

SOAP_SCALE = CANONICAL_ASSET_SCALES[SOAP_MODEL]
SPONGE_SCALE = CANONICAL_ASSET_SCALES[SPONGE_MODEL]
CAN_SCALE = CANONICAL_ASSET_SCALES[CAN_MODEL]
CANDLE_SCALE = CANONICAL_ASSET_SCALES[CANDLE_MODEL]
BOXED_FOOD_SCALE = CANONICAL_ASSET_SCALES[BOXED_FOOD_MODEL]
MUG_SCALE = CANONICAL_ASSET_SCALES[MUG_MODEL]

# Containers are calibrated independently from the hand-held-object span.
# Their exact model and scale are nevertheless canonical across all consumers.
BASKET_SCALE = 0.75
BOWL_SCALE = 1.0
LIFT_BASKET_SCALE = 1.0


def asset_xml_path(model_path, *, require_exists=False):
    """Resolve a canonical task asset without requiring downloaded asset packs.

    Task classes store these paths while the catalog is imported. Checking at
    that point would make even unrelated tasks unavailable when an optional
    asset pack is absent. Asset loaders validate files when constructing a
    task; callers that need an explicit check can pass ``require_exists=True``.
    """

    path = os.path.abspath(os.path.join(assets_root, model_path.lstrip("/")))
    if require_exists and not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing iTHOR task asset: {path}. "
            "Provision the required asset pack before constructing this task."
        )
    return path


__all__ = [
    "BASKET_MODEL",
    "BASKET_SCALE",
    "BOWL_MODEL",
    "BOWL_SCALE",
    "BOXED_FOOD_MODEL",
    "BOXED_FOOD_SCALE",
    "CANONICAL_ASSET_SCALES",
    "CAN_MODEL",
    "CAN_SCALE",
    "CANDLE_MODEL",
    "CANDLE_SCALE",
    "LIFT_BASKET_MODEL",
    "LIFT_BASKET_SCALE",
    "MAX_GRASP_SPAN",
    "MIN_GRASP_SPAN",
    "MUG_MODEL",
    "MUG_SCALE",
    "NATIVE_GRASP_SPANS",
    "SOAP_MODEL",
    "SOAP_SCALE",
    "SPONGE_MODEL",
    "SPONGE_SCALE",
    "TARGET_GRASP_SPAN",
    "asset_xml_path",
]
