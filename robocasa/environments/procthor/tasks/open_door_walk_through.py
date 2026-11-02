"""ProcTHOR semantic door traversal task."""

import numpy as np


class ProcTHOROpenDoorWalkThroughMixin:
    ATOMIC_SKILL = "WalkThroughDoor"
    DIFFICULTY_LEVEL = "L3"
    TASK_FAMILY = "navigation"
    HOLD_STEPS = 5

    def _door_rooms(self, fixture):
        object_id = fixture.object_id or ""
        parts = object_id.split("|")
        if len(parts) >= 3 and parts[0].lower() == "door":
            return str(parts[-2]), str(parts[-1])
        room_id = str(fixture.room_id) if fixture.room_id is not None else None
        adjacent = self.semantic_scene.adjacent_rooms(room_id) if room_id else []
        return (room_id, adjacent[0]) if room_id and adjacent else (None, None)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        candidates = self.get_fixture(
            category=("Doorway", "Door"),
            capabilities=("openable",),
            return_all=True,
        )
        candidates = [
            fixture for fixture in candidates if all(self._door_rooms(fixture))
        ]
        replay_id = (
            self._ep_meta.get("semantic_refs", {}).get("fixtures", {}).get("door")
        )
        replay = self.get_fixture(id=replay_id) if replay_id else None
        self.door_fixture = (
            replay if replay in candidates else candidates[0] if candidates else None
        )
        if self.door_fixture is None:
            return
        self.fixture_refs["door"] = self.door_fixture
        self.set_robot_base_from_fixture(self.door_fixture, distance=0.8)

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and getattr(self, "door_fixture", None) is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold = 0
        self.door_semantic_joint = self.door_fixture.semantic_joint()
        self.door_joint = self.door_semantic_joint.name
        self.sim.data.set_joint_qpos(
            self.door_joint, self.door_semantic_joint.closed_qpos
        )
        self.sim.forward()
        if not self.place_robot_at_fixture(self.door_fixture):
            raise RuntimeError(f"{self.scene_name}: no collision-free G1 pose for door")
        room_a, room_b = self._door_rooms(self.door_fixture)
        self.source_room = self.room_of_point(self._robot_base_xy())
        if self.source_room == room_a:
            self.target_room = room_b
        elif self.source_room == room_b:
            self.target_room = room_a
        else:
            raise RuntimeError(
                f"{self.scene_name}: G1 is not in either room adjacent to door"
            )

    def _raw_success(self):
        door_open = self.door_semantic_joint.fraction(self.sim) >= 0.40
        return bool(
            door_open and self.room_of_point(self._robot_base_xy()) == self.target_room
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._raw_success() else 0
        return ret

    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["door_fixture"] = self.door_fixture.name
        ep_meta["door_joint"] = self.door_joint
        ep_meta["source_room"] = self.source_room
        ep_meta["target_room"] = self.target_room
        target_center = np.mean(
            self.semantic_scene.rooms[self.target_room]["poly"], axis=0
        )
        ep_meta["target_center"] = [float(v) for v in target_center]
        ep_meta["lang"] = (
            "open the closed door in front of you and walk through it into the "
            "next room"
        )
        return ep_meta


from robocasa.environments.procthor.procthor import ProcTHOREnv


class ProcTHOROpenDoorWalkThrough(ProcTHOROpenDoorWalkThroughMixin, ProcTHOREnv):
    pass


__all__ = ["ProcTHOROpenDoorWalkThrough"]
