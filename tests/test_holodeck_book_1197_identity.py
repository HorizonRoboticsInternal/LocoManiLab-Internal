"""The 1197 book workflow uses the same object as scene 1664, only locally."""

import xml.etree.ElementTree as ET
from pathlib import Path

from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckPickBookFromSourceToDesk,
)
from robocasa.environments.holodeck.tasks.pick_book_from_dresser import (
    HolodeckPickBookFromDresser,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


def test_1197_book_workflows_use_1664_asset_and_physics_without_scene_leak():
    book_1197 = get_holodeck_task_spec("train_1197", "pick_book_from_dresser")
    book_1664 = get_holodeck_task_spec("train_1664", "pick_book_from_dresser")
    native_book = "book_fde77405a8546edd1a9cd35621b51508_1_0_0"
    lookalike = "obja_27e4472ffcdb4b49b9feda691ded2704_1_0_0"

    assert "target_body" not in book_1197
    assert book_1197["remove_scene_bodies"] == [native_book, lookalike]
    assert (book_1197["book_scale"], book_1197["book_density"]) == (
        book_1664["book_scale"],
        book_1664["book_density"],
    )
    assert HolodeckPickBookFromSourceToDesk.SPEC_TASK == (
        HolodeckPickBookFromDresser.SPEC_TASK
    )
    assert book_1197["book_pose"]["pos"][:2] == [5.46986, 5.11871]
    other_task_removals = get_holodeck_task_spec(
        "train_1197", "pick_cup_from_desk"
    ).get("remove_scene_bodies", ())
    assert native_book not in other_task_removals
    assert lookalike not in other_task_removals


def test_1197_task_removes_native_book_and_dresser_lookalike():
    scene_xml = (
        Path(__file__).resolve().parents[1]
        / "robocasa/models/assets/scenes/holodeck-objaverse-train/train_1197.xml"
    )
    worldbody = ET.parse(scene_xml).getroot().find("worldbody")
    scene_bodies = {body.get("name") for body in worldbody.iter("body")}
    baked_book_roots = {
        body.get("name")
        for body in worldbody.iter("body")
        if body.get("name", "").startswith("book_")
        and body.get("name", "").endswith("_1_0_0")
    }
    spec = get_holodeck_task_spec("train_1197", "pick_book_from_dresser")

    # Other books listed in scene metadata are not instantiated in the
    # runnable XML. Book_3 is spawned only after this one baked book is gone.
    lookalike = "obja_27e4472ffcdb4b49b9feda691ded2704_1_0_0"
    assert lookalike in scene_bodies
    assert baked_book_roots | {lookalike} == set(spec["remove_scene_bodies"])
    assert HolodeckPickBookFromDresser.BOOK_ASSET.endswith("Book_3/Book_3.xml")
