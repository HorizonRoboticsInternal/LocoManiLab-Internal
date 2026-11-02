"""FloorPlan310 desk uses basket-walk start; bed needs a safe indoor offset."""

from robocasa.environments.ithor_bedroom.tasks.articulation_composites import (
    IThorBedroomApproachDeskAndCloseLaptop,
    IThorBedroomApproachDeskAndOpenLaptop,
)
from robocasa.environments.ithor_bedroom.tasks.navigation import (
    IThorBedroomApproachBed,
    IThorBedroomApproachDesk,
)
from robocasa.environments.ithor_bedroom.tasks.navigation_composites import (
    IThorBedroomBedThenDesk,
    IThorBedroomBedThenStorage,
    IThorBedroomDeskThenBed,
    IThorBedroomDeskThenStorage,
    IThorBedroomStorageThenBed,
    IThorBedroomStorageThenDesk,
)
from robocasa.environments.ithor_bedroom.tasks.routine_composites import (
    IThorBedroomApproachDeskAndCloseLaptopThenApproachBed,
)
from robocasa.environments.ithor_bedroom.tasks.workspace_composites import (
    IThorBedroomApproachDeskAndLiftBasket,
)
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
)


def test_floorplan310_bedroom_starts_share_safe_atomic_specs():
    scene = "FloorPlan310"
    basket_start = IThorBedroomApproachDeskAndLiftBasket.SCENE_SPECS[scene][
        "far_start_pose"
    ]
    assert basket_start == {"xy": [0.65, -0.70], "yaw": -2.486158}
    for atomic, routes in (
        (
            IThorBedroomApproachBed,
            (IThorBedroomBedThenDesk, IThorBedroomBedThenStorage),
        ),
        (
            IThorBedroomApproachDesk,
            (IThorBedroomDeskThenBed, IThorBedroomDeskThenStorage),
        ),
    ):
        spec = atomic.SCENE_SPECS[scene]
        assert (
            spec["start_pose"]
            == get_ithor_bedroom_task_spec(scene, atomic.SPEC_TASK)["start_pose"]
        )
        for route in routes:
            assert route.SCENE_SPECS[scene]["start_pose"] == spec["start_pose"]
            assert (
                route.SCENE_SPECS[scene]["minimum_start_distance"]
                == spec["minimum_start_distance"]
            )

    # Basket-walk XY is already within 0.05 m of the bed, so bed needs an offset.
    bed = IThorBedroomApproachBed.SCENE_SPECS[scene]
    assert bed["start_pose"] == {"xy": [0.60, -1.90], "yaw": 1.335}
    assert bed["minimum_start_distance"] == 1.00
    desk = IThorBedroomApproachDesk.SCENE_SPECS[scene]
    assert desk["start_pose"] == basket_start
    assert desk["minimum_start_distance"] == 0.70
    for composite in (
        IThorBedroomApproachDeskAndOpenLaptop,
        IThorBedroomApproachDeskAndCloseLaptop,
    ):
        assert (
            composite.SCENE_SPECS[scene]["navigation"]["start_pose"]
            == desk["start_pose"]
        )
    routine = IThorBedroomApproachDeskAndCloseLaptopThenApproachBed.SCENE_SPECS[scene]
    assert routine["routine_targets"]["approach_bed"]["start_pose"] == (
        IThorBedroomApproachBed.SCENE_SPECS[scene]["start_pose"]
    )

    storage = get_ithor_bedroom_task_spec(scene, "approach_storage")["start_pose"]
    assert IThorBedroomStorageThenBed.SCENE_SPECS[scene]["start_pose"] == storage
    assert IThorBedroomStorageThenDesk.SCENE_SPECS[scene]["start_pose"] == storage
