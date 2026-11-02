"""Bedroom object-in-bowl and object-in-basket tasks."""

from robocasa.environments.ithor_atomic.pick_place import (
    SurfaceObjectInReceptacle,
)
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)

from .assets import (
    BASKET_MODEL,
    BASKET_SCALE,
    BOXED_FOOD_MODEL,
    BOXED_FOOD_SCALE,
    CAN_MODEL,
    CAN_SCALE,
    MUG_MODEL,
    MUG_SCALE,
    asset_xml_path,
)
from .surface_placement import BedroomSurfacePlacementMixin

PNP_SPEC_REQUIRED = ("lang",)


class BedroomPnPMixin(
    BedroomSurfacePlacementMixin, SurfaceObjectInReceptacle
):
    """Shared task behavior; concrete room tasks provide registration."""

    # Several Bedroom receptacles have collidable feet above a loose visual
    # bbox lower face. Strict consumers still require named-support contact;
    # this tolerance only accommodates that asset-local bbox offset.
    RECEPTACLE_SUPPORT_Z_TOLERANCE = 0.08


class BedroomSimpleReceptaclePlacementMixin:
    """Accept a released item once its center and bottom enter its target."""

    RECEPTACLE_XY_CONTAINMENT = "center"

    def _raw_place_success(self):
        source = self.objects[self.SOURCE_NAME]
        receptacle = self.objects[self.RECEPTACLE_NAME]
        return bool(
            self.check_contact(source, receptacle)
            and self._source_bbox_inside_receptacle()
            and self.object_released(
                self.SOURCE_NAME, th=self.RELEASE_DISTANCE
            )
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["success_definition"] = (
            "the released source center and bottom must be inside and contact "
            f"the {self.RECEPTACLE_GROUP} for {self.HOLD_STEPS} control steps"
        )
        return ep_meta


class BedroomNearSurfaceStartMixin:
    """Start slightly closer to the work surface when collision-free."""

    STAND_DISTANCES = (0.40, 0.45, 0.48, 0.58, 0.68, 0.80)


class BedroomRightHandSourceLayoutMixin:
    """Place the grasped source to the operator's right of its receptacle."""

    SOURCE_ON_OPERATOR_RIGHT = True


class IThorBedroomCanIntoBasket(
    BedroomSimpleReceptaclePlacementMixin,
    BedroomNearSurfaceStartMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomPnPMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "can_into_basket"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK, required=PNP_SPEC_REQUIRED
    )
    SOURCE_GROUP, RECEPTACLE_GROUP = "can", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(CAN_MODEL)
    RECEPTACLE_MJCF_PATH = asset_xml_path(BASKET_MODEL)
    SOURCE_SCALE, RECEPTACLE_SCALE = CAN_SCALE, BASKET_SCALE


class IThorBedroomMugIntoBasket(
    BedroomSimpleReceptaclePlacementMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomPnPMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "mug_into_basket"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK, required=PNP_SPEC_REQUIRED
    )
    SOURCE_GROUP, RECEPTACLE_GROUP = "mug", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(MUG_MODEL)
    RECEPTACLE_MJCF_PATH = asset_xml_path(BASKET_MODEL)
    SOURCE_SCALE, RECEPTACLE_SCALE = MUG_SCALE, BASKET_SCALE
    STAND_DISTANCES = (0.35, 0.40, 0.48, 0.58, 0.68, 0.80)
    # FloorPlan302 needs one wider along-edge candidate to stay 0.35 m from
    # the table without colliding with nearby scene geometry.
    STAND_LATERAL_OFFSETS = (0.0, 0.15, -0.15, 0.30, -0.30, 0.40, -0.40)
    # A physically settled upright mug sits about 2.8 cm below the Basket034
    # semantic lower bound even while contacting its interior floor.
    RECEPTACLE_BOTTOM_TOLERANCE = 0.035


class IThorBedroomBoxedFoodIntoBasket(
    BedroomSimpleReceptaclePlacementMixin,
    BedroomNearSurfaceStartMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomPnPMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "boxed_food_into_basket"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK, required=PNP_SPEC_REQUIRED
    )
    SOURCE_GROUP, RECEPTACLE_GROUP = "boxed_food", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(BOXED_FOOD_MODEL)
    RECEPTACLE_MJCF_PATH = asset_xml_path(BASKET_MODEL)
    SOURCE_SCALE, RECEPTACLE_SCALE = BOXED_FOOD_SCALE, BASKET_SCALE


__all__ = [
    "IThorBedroomCanIntoBasket",
    "IThorBedroomMugIntoBasket",
    "IThorBedroomBoxedFoodIntoBasket",
]
