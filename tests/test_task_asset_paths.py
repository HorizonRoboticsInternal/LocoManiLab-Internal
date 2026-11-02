"""Catalog paths and missing-asset diagnostics without scene packs or policies."""

import os

import pytest

from robocasa.environments.ithor_atomic import task_assets
from robocasa.environments.ithor_bathroom.tasks.floor_lift import (
    _BathroomCandleStoolTask,
)
from robocasa.environments.ithor_bedroom.tasks.lift_basket import (
    IThorBedroomLiftBasket,
)
from robocasa.models.objects.objects import MJCFObject


@pytest.mark.parametrize("model", [
    task_assets.BASKET_MODEL,
    task_assets.BOWL_MODEL,
    task_assets.BOXED_FOOD_MODEL,
    task_assets.CAN_MODEL,
    task_assets.CANDLE_MODEL,
    task_assets.LIFT_BASKET_MODEL,
    task_assets.MUG_MODEL,
    task_assets.SOAP_MODEL,
    task_assets.SPONGE_MODEL,
])
def test_catalog_asset_paths_resolve_without_accessing_asset_files(
    monkeypatch, tmp_path, model,
):
    monkeypatch.setattr(task_assets, "assets_root", str(tmp_path))

    def forbidden(path):
        pytest.fail(f"Catalog path resolution checked an optional asset: {path}")

    monkeypatch.setattr(task_assets.os.path, "isfile", forbidden)
    expected = os.path.abspath(os.path.join(tmp_path, model.lstrip("/")))
    assert task_assets.asset_xml_path(model) == expected


def test_imported_task_classes_keep_the_same_canonical_asset_paths():
    assert IThorBedroomLiftBasket.TARGET_XML == task_assets.asset_xml_path(
        task_assets.LIFT_BASKET_MODEL
    )
    assert _BathroomCandleStoolTask.TARGET_XML == task_assets.asset_xml_path(
        task_assets.CANDLE_MODEL
    )


def test_explicit_asset_validation_reports_the_required_missing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(task_assets, "assets_root", str(tmp_path))
    expected = task_assets.asset_xml_path(task_assets.LIFT_BASKET_MODEL)
    with pytest.raises(FileNotFoundError, match="Missing iTHOR task asset") as error:
        task_assets.asset_xml_path(task_assets.LIFT_BASKET_MODEL, require_exists=True)
    assert expected in str(error.value)
    assert "required asset pack" in str(error.value)


def test_explicit_asset_validation_accepts_a_provisioned_file(monkeypatch, tmp_path):
    monkeypatch.setattr(task_assets, "assets_root", str(tmp_path))
    path = tmp_path / task_assets.LIFT_BASKET_MODEL.lstrip("/")
    path.parent.mkdir(parents=True)
    path.write_text("<mujoco/>")
    assert task_assets.asset_xml_path(
        task_assets.LIFT_BASKET_MODEL, require_exists=True,
    ) == str(path)


def test_actual_object_loading_reports_missing_asset_without_scene_setup(tmp_path):
    path = tmp_path / "unprovisioned-model.xml"
    with pytest.raises(FileNotFoundError, match="Missing object MJCF asset") as error:
        MJCFObject(name="missing-task-object", mjcf_path=str(path))
    assert str(path) in str(error.value)
    assert "required asset pack" in str(error.value)
