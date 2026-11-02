"""
``EmptyEnv`` — an asset-free native MuJoCo ground plane + robot, nothing else.

The minimal kitchen-free environment: a robot on a collision-infinite MuJoCo
plane and nothing else. It is the default free-teleoperation benchmark for
checking that a robot loads, balances, controls, and drives without importing
an external scene asset library. It is a plain
``ManipulationEnv`` subclass (NOT a ``Kitchen``), pulling in none of the
fixture/layout/style machinery.

Concrete scenes (e.g. the ``living_room`` stack) are their own independent
``ManipulationEnv`` subclasses and do NOT build on this env.

robosuite's ``EnvMeta`` auto-registers it under ``EmptyEnv`` at import time, so
``robosuite.make("EmptyEnv", ...)`` works once this module has been imported.

Teleop entry point: ``python -m robocasa.demos.demo_empty_teleop``.
"""

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask

from robocasa.models.arenas import NativeFloorArena


class EmptyEnv(ManipulationEnv):
    """
    A bare floor + robot, for testing. No scene content.

    Args:
        robots: G1 robot variant (e.g. "G1FloatingBody"). A
            mobile/floating base is recommended if you want to drive the robot
            around the scene.
        robot_base_pos (3-tuple): where to place the robot base on the floor.
        robot_base_ori (3-tuple): base roll, pitch, and yaw in radians.
        All other args mirror the standard robosuite environment constructor.
    """

    # There is intentionally no success predicate in a free-teleoperation
    # benchmark. The shared viewer uses this opt-in to expose S as an explicit
    # episode boundary without allowing incomplete formal-task demos to save.
    manual_episode_save = True

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        robot_base_pos=(0.0, 0.0, 0.0),
        robot_base_ori=(0.0, 0.0, 0.0),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.robot_base_pos = np.array(robot_base_pos)
        self.robot_base_ori = np.array(robot_base_ori)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_model(self):
        """Build the scene: one native ground plane + robot, no objects."""
        super()._load_model()

        for robot in self.robots:
            robot.robot_model.set_base_xpos(self.robot_base_pos)
            robot.robot_model.set_base_ori(self.robot_base_ori)

        mujoco_arena = self._make_arena()
        mujoco_arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )

    def _make_arena(self):
        """Return the arena merged with the robot.

        Scene-only subclasses can replace the floor while reusing EmptyEnv's
        robot and controller setup.
        """
        return NativeFloorArena()

    def reward(self, action=None):
        """No task -- always zero."""
        return 0.0

    def _check_success(self):
        """No success condition (teleop runs until you quit)."""
        return False

    def get_ep_meta(self):
        """Episode metadata. ``lang`` is the natural-language instruction shown
        by the teleop/data-collection scripts."""
        return {"lang": "freely teleoperate the robot on flat ground"}
