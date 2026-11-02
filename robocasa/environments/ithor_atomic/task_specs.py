"""Scene-calibrated task specification helpers for iTHOR tasks."""

from copy import deepcopy


def _scene_name(scene_id):
    if isinstance(scene_id, str) and scene_id.startswith("FloorPlan"):
        return scene_id
    return f"FloorPlan{int(scene_id)}"


class SceneSpecMixin:
    """Resolve an exact, immutable-by-convention specification per scene.

    Concrete tasks declare both ``SUPPORTED_SCENE_IDS`` and ``SCENE_SPECS``.
    Explicit ``scene_ids`` are restricted to that calibrated set, so callers
    cannot accidentally launch a task in an unvalidated room.
    """

    SUPPORTED_SCENE_IDS = ()
    SCENE_SPECS = {}

    def __init__(self, *args, scene_ids=None, **kwargs):
        supported = tuple(int(scene_id) for scene_id in self.SUPPORTED_SCENE_IDS)
        if len(set(supported)) != len(supported):
            raise ValueError(
                f"{type(self).__name__}: duplicate SUPPORTED_SCENE_IDS"
            )
        supported_names = {_scene_name(scene_id) for scene_id in supported}
        spec_names = set(self.SCENE_SPECS)
        if supported_names != spec_names:
            missing = sorted(supported_names - spec_names)
            extra = sorted(spec_names - supported_names)
            raise ValueError(
                f"{type(self).__name__}: SUPPORTED_SCENE_IDS and SCENE_SPECS "
                f"disagree (missing={missing}, extra={extra})"
            )

        requested = supported if scene_ids is None else scene_ids
        requested_ids = (
            (int(requested),)
            if isinstance(requested, (int, str))
            else tuple(int(scene_id) for scene_id in requested)
        )
        if not requested_ids:
            raise ValueError(f"{type(self).__name__}: scene_ids cannot be empty")
        unsupported = sorted(set(requested_ids) - set(supported))
        if unsupported:
            raise ValueError(
                f"{type(self).__name__}: uncalibrated scene ids {unsupported}; "
                f"supported={list(supported)}"
            )
        super().__init__(*args, scene_ids=requested_ids, **kwargs)

    def scene_task_spec(self):
        """Return a defensive copy of the active scene's exact task spec."""

        try:
            return deepcopy(self.SCENE_SPECS[self.scene_name])
        except KeyError as exc:
            raise RuntimeError(
                f"{type(self).__name__} has no task spec for {self.scene_name}"
            ) from exc


__all__ = ["SceneSpecMixin"]
