import pytest

from robocasa.environments.holodeck.tasks.shelf_folder import (
    HolodeckInsertFolderIntoShelf,
    HolodeckRemoveFolderFromShelf,
    _HolodeckShelfFolderMixin,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckRemoveFolderFromShelfToDesk,
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _FakeParent:
    def _post_action(self, action):
        return action


class _FakeInsertTask(_HolodeckShelfFolderMixin, _FakeParent):
    MODE = "insert"

    def __init__(self):
        self._success_hold = 0
        self.control_freq = 20
        self.center_in_region = False
        self.shelf_contact = False

    def g1_hand_grasping_object(self, name):
        raise AssertionError("shelf success must not use the grasp predicate")

    def _folder_center_in_region(self):
        return self.center_in_region

    def _folder_contacts_shelf(self):
        return self.shelf_contact


class _FakeRemoveTask(_HolodeckShelfFolderMixin, _FakeParent):
    MODE = "remove"

    def __init__(self):
        self._success_hold = 0
        self.control_freq = 20
        self.shelf_contact = True

    def g1_hand_grasping_object(self, name):
        raise AssertionError("shelf success must not use the grasp predicate")

    def _folder_contacts_shelf(self):
        return self.shelf_contact


def test_folder_tasks_support_the_configured_shelf_scenes():
    expected = (1018, 1197, 2025)
    assert HolodeckInsertFolderIntoShelf.SUPPORTED_SCENE_IDS == expected
    assert HolodeckRemoveFolderFromShelf.SUPPORTED_SCENE_IDS == expected


def test_scene_1197_folder_calibrations_use_the_lower_shelf():
    insert_spec = get_holodeck_task_spec(
        "train_1197", "insert_folder_into_shelf"
    )
    remove_spec = get_holodeck_task_spec(
        "train_1197", "remove_folder_from_shelf"
    )

    lower_shelf_center = [5.25, 0.20, 0.6749]
    assert insert_spec["robot_pose"]["pos"] == remove_spec["robot_pose"]["pos"]
    assert insert_spec["robot_pose"]["yaw"] == (
        remove_spec["robot_pose"]["yaw"] + 3.141592653589793
    )
    assert insert_spec["region_center"] == lower_shelf_center
    assert remove_spec["folder_pose"]["pos"] == lower_shelf_center
    assert remove_spec["region_center"] == lower_shelf_center
    assert remove_spec["source_center"] == lower_shelf_center

    for composite_task in (
        "remove_folder_from_shelf_to_desk",
        "remove_folder_from_shelf_to_desk_then_open_laptop",
        "remove_folder_from_shelf_to_desk_then_close_laptop",
    ):
        composite_spec = get_holodeck_task_spec("train_1197", composite_task)
        assert composite_spec["folder_pose"]["pos"] == lower_shelf_center
        assert composite_spec["region_center"] == lower_shelf_center


def test_insert_folder_requires_center_and_shelf_contact_for_one_second():
    task = _FakeInsertTask()
    task.center_in_region = True
    task.shelf_contact = True
    task._post_action(None)
    assert task._success_hold == 1

    for _ in range(task.hold_steps - 2):
        task._post_action(None)
    assert task._success_hold == task.hold_steps - 1
    assert not task._check_success()

    task._post_action(None)
    assert task._success_hold == task.hold_steps
    assert task._check_success()


def test_insert_folder_hold_resets_when_folder_leaves_target_region():
    task = _FakeInsertTask()
    task.center_in_region = True
    task.shelf_contact = True
    task._post_action(None)
    assert task._success_hold == 1

    task.center_in_region = False
    task._post_action(None)
    assert task._success_hold == 0


def test_insert_folder_hold_resets_without_shelf_contact():
    task = _FakeInsertTask()
    task.center_in_region = True
    task.shelf_contact = True
    task._post_action(None)
    assert task._success_hold == 1

    task.shelf_contact = False
    task._post_action(None)
    assert task._success_hold == 0


def test_remove_folder_requires_no_shelf_contact_for_one_second():
    task = _FakeRemoveTask()
    task.shelf_contact = False
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()

    task._post_action(None)
    assert task._check_success()


def test_remove_folder_hold_resets_when_contact_returns():
    task = _FakeRemoveTask()
    task.shelf_contact = False
    task._post_action(None)
    assert task._success_hold == 1

    task.shelf_contact = True
    task._post_action(None)
    assert task._success_hold == 0


@pytest.mark.parametrize(
    "env_name",
    ("HolodeckInsertFolderIntoShelf", "HolodeckRemoveFolderFromShelf"),
)
def test_folder_tasks_reset_without_robot_collision(env_name):
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name=env_name,
        robots="G1",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=1018,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env._robot_footprint_is_in_room(radius=0.25)
        assert not env._check_success()
        if env_name == "HolodeckInsertFolderIntoShelf":
            assert env._folder_on_desk(require_released=False)
    finally:
        env.close()


@pytest.mark.parametrize("scene_id", (1197, 2025))
def test_folder_to_desk_composite_resets_and_exposes_language(scene_id):
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name="HolodeckRemoveFolderFromShelfToDesk",
        robots="G1",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=scene_id,
    )
    try:
        env.reset()
        assert env.source_desk_region is not None
        assert env.get_ep_meta()["lang"] == (
            "remove the folder from the shelf, then place it on the desk"
        )
        assert not env._check_success()
    finally:
        env.close()


@pytest.mark.parametrize("scene_id", (1197, 2025))
def test_folder_to_desk_open_laptop_composite_resets(scene_id):
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name="HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop",
        robots="G1",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=scene_id,
    )
    try:
        env.reset()
        assert env.target_fixture.body_name == "spawned_laptop_Laptop_26"
        assert env.source_desk_region is not None
        assert not env._check_success()
    finally:
        env.close()
