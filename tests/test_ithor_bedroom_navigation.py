"""Static contracts for calibrated iTHOR Bedroom navigation starts."""

import math

import pytest

from robocasa.environments.ithor_bedroom.tasks.navigation import (
    IThorBedroomApproachBed,
    IThorBedroomApproachDesk,
    IThorBedroomNavigateToStorage,
)


_EXPECTED_STARTS = {
    IThorBedroomApproachBed: {
        "FloorPlan302": ((-0.90, 0.90), -0.734327, 1.25),
        "FloorPlan306": ((-0.75, 0.60), -0.499158, 1.65),
        "FloorPlan310": ((-2.25, -2.85), 0.732255, 2.70),
        "FloorPlan323": ((1.50, 1.35), -2.349079, 2.50),
        "FloorPlan329": ((1.80, 0.75), -2.268386, 1.40),
    },
    IThorBedroomApproachDesk: {
        "FloorPlan302": ((0.75, 0.60), -2.340163, 1.35),
        "FloorPlan306": ((2.85, 0.30), -2.660245, 2.60),
        "FloorPlan310": ((2.70, 1.80), -2.326290, 3.80),
        "FloorPlan323": ((0.00, 1.35), -1.124490, 4.00),
        "FloorPlan329": ((1.95, -1.95), 2.245555, 2.40),
    },
    IThorBedroomNavigateToStorage: {
        "FloorPlan302": ((-0.45, -0.60), 0.609252, 1.50),
        "FloorPlan306": ((2.85, 0.30), -2.516810, 2.40),
        "FloorPlan310": ((-1.35, -0.75), -0.079549, 2.30),
        "FloorPlan323": ((0.30, -2.85), 1.075583, 3.90),
        "FloorPlan329": ((1.95, -2.10), 1.332943, 1.40),
    },
}


@pytest.mark.parametrize("task_class", _EXPECTED_STARTS)
def test_bedroom_navigation_uses_exact_distant_start_specs(task_class):
    expected = _EXPECTED_STARTS[task_class]
    assert tuple(task_class.SCENE_SPECS) == tuple(expected)
    assert task_class.SUPPORTED_SCENE_IDS == tuple(
        int(scene_name.removeprefix("FloorPlan")) for scene_name in expected
    )

    for scene_name, (xy, yaw, minimum_distance) in expected.items():
        spec = task_class.SCENE_SPECS[scene_name]
        assert set(spec) == {
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        }
        assert tuple(spec["start_pose"]["xy"]) == pytest.approx(xy)
        assert float(spec["start_pose"]["yaw"]) == pytest.approx(yaw)
        assert math.isfinite(float(spec["start_pose"]["yaw"]))
        assert float(spec["minimum_start_distance"]) == pytest.approx(
            minimum_distance
        )
        assert minimum_distance >= 1.25
        assert spec["remove_fixtures"] == []
