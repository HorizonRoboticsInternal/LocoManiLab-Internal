"""Regression checks for Holodeck task success predicates."""

from types import SimpleNamespace

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.ice_cube_in_glass import (
    HolodeckIceCubeInGlass,
)
from robocasa.environments.holodeck.tasks.move_cup import HolodeckMoveCupToCenter
from robocasa.environments.holodeck.tasks.multi_hop.desk_workflows import (
    HolodeckApproachDeskThenMoveCupToCenter,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckApproachDeskThenCloseLaptopMoveCup,
    HolodeckMoveCupToCenterThenOpenLaptop,
)
from robocasa.environments.holodeck.tasks.throw_bottle_into_trash_can import (
    HolodeckThrowBottleIntoTrashCan,
)


def _stub_parent_post_action(monkeypatch):
    monkeypatch.setattr(
        HolodeckEnv,
        "_post_action",
        lambda _self, _action: (0.0, False, {}),
    )


def _bottle_task_with_contacts(contact_pairs):
    task = object.__new__(HolodeckThrowBottleIntoTrashCan)
    task.trash_can_body = "trash_can"
    task.objects = {"bottle": SimpleNamespace(root_body="bottle")}
    geom_ids = {
        "trash_can_inner_floor": 0,
        "bottle_collision": 1,
        "trash_can_inner_wall_x0": 2,
    }
    model = SimpleNamespace(
        body_name2id=lambda _name: 1,
        geom_name2id=lambda name: geom_ids[name],
        geom_bodyid=[0, 1, 0],
        body_parentid=[0, 0],
        nbody=2,
    )
    contacts = [
        SimpleNamespace(geom1=first, geom2=second)
        for first, second in contact_pairs
    ]
    task.sim = SimpleNamespace(
        model=model,
        data=SimpleNamespace(
            ncon=len(contacts),
            contact=contacts,
        )
    )
    return task


def test_move_cup_success_requires_contact_with_the_desk(monkeypatch):
    _stub_parent_post_action(monkeypatch)
    task = object.__new__(HolodeckMoveCupToCenter)
    task.control_freq = 20
    task._success_hold_steps = 0
    task._cup_base_center = lambda: object()
    task._cup_in_target_region = lambda _center: True
    task._cup_contacts_desk = lambda: False

    for _ in range(task.hold_steps):
        task._post_action(None)

    assert not task._check_success()


def test_move_cup_success_holds_target_region_and_desk_contact(monkeypatch):
    _stub_parent_post_action(monkeypatch)
    task = object.__new__(HolodeckMoveCupToCenter)
    task.control_freq = 20
    task._success_hold_steps = 0
    task._cup_base_center = lambda: object()
    task._cup_in_target_region = lambda _center: True
    task._cup_contacts_desk = lambda: True

    for _ in range(task.hold_steps):
        task._post_action(None)

    assert task._check_success()


def test_move_cup_composites_use_the_shared_grounded_predicate():
    approach_task = object.__new__(HolodeckApproachDeskThenMoveCupToCenter)
    approach_task._cup_base_center = lambda: object()
    approach_task._cup_in_target_region = lambda _center: True
    approach_task._cup_contacts_desk = lambda: False
    assert not approach_task._cup_in_target_state()

    laptop_task = object.__new__(HolodeckMoveCupToCenterThenOpenLaptop)
    laptop_task.task_spec = object()
    laptop_task.cup_task_spec = object()
    laptop_task._cup_base_center = lambda: object()
    laptop_task._cup_in_target_region = lambda _center: True
    laptop_task._cup_contacts_desk = lambda: False
    assert not laptop_task._cup_centered()

    corner_task = object.__new__(HolodeckApproachDeskThenCloseLaptopMoveCup)
    corner_task.task_spec = object()
    corner_task.cup_task_spec = object()
    corner_task._cup_base_center = lambda: object()
    corner_task._cup_in_target_region = lambda _center: True
    corner_task._cup_contacts_desk = lambda: False
    assert not corner_task._cup_in_target_state()


def test_bottle_success_requires_contact_with_the_bin_floor():
    task = _bottle_task_with_contacts(((0, 1),))

    assert task._bottle_in_bin()


def test_bottle_sidewall_contact_is_not_success():
    task = _bottle_task_with_contacts(((2, 1),))

    assert not task._bottle_in_bin()


def test_bottle_floor_contact_must_be_held_for_one_second(monkeypatch):
    _stub_parent_post_action(monkeypatch)
    task = _bottle_task_with_contacts(((0, 1),))
    task.control_freq = 20
    task._success_hold = 0

    for _ in range(task.hold_steps - 1):
        reward, _, info = task._post_action(None)
    assert reward == 0.0
    assert info["task_success"] is False

    reward, _, info = task._post_action(None)
    assert reward == 1.0
    assert info["task_success"] is True


def test_ice_cube_success_requires_one_second_in_the_target_region(monkeypatch):
    _stub_parent_post_action(monkeypatch)
    task = object.__new__(HolodeckIceCubeInGlass)
    task.control_freq = 20
    task._success_hold = 0
    task._ice_in_cup_region = lambda: True

    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()

    task._post_action(None)
    assert task._check_success()
