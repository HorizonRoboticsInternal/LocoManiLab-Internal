"""Calibrated fixture-navigation tasks for maintained iTHOR bathrooms."""

from robocasa.environments.ithor_atomic.navigation import ExactSceneNavigation
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)

class IThorBathroomApproachToilet(ExactSceneNavigation, IThorBathroomEnv):
    """Navigate to the calibrated toilet in any maintained bathroom."""

    SUPPORTED_SCENE_IDS = get_supported_scene_ids("approach_toilet")
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        "approach_toilet",
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


class IThorBathroomApproachSink(ExactSceneNavigation, IThorBathroomEnv):
    """Navigate to the calibrated sink in any maintained bathroom."""

    SUPPORTED_SCENE_IDS = get_supported_scene_ids("approach_sink")
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        "approach_sink",
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


class IThorBathroomApproachGarbageCan(ExactSceneNavigation, IThorBathroomEnv):
    SUPPORTED_SCENE_IDS = get_supported_scene_ids("approach_garbage_can")
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        "approach_garbage_can",
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


class IThorBathroomNavigateToBathingArea(
    ExactSceneNavigation,
    IThorBathroomEnv,
):
    SUPPORTED_SCENE_IDS = get_supported_scene_ids("navigate_to_bathing_area")
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        "navigate_to_bathing_area",
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


__all__ = [
    "IThorBathroomApproachGarbageCan",
    "IThorBathroomApproachSink",
    "IThorBathroomApproachToilet",
    "IThorBathroomNavigateToBathingArea",
]
