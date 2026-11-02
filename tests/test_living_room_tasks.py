"""Validity test for the living-room env stack (peer of ``test_tasks_validity``).

Living-room envs are an independent stack (``REGISTERED_LIVING_ROOM_ENVS``) that
does NOT go through kitchen's ``create_env``/split machinery -- they are built
directly with ``robosuite.make`` on the ``G1FloatingBody`` humanoid, exactly the
way ``collect_demos_g1`` builds them. This iterates every registered
living-room env (the base + all task subclasses), constructs it, resets it
``num_rollouts`` times, and runs a few zero-action steps per reset (so object
placement + ``_check_success`` are exercised), printing per-env PASS/FAIL with
tracebacks and a final summary.

    MUJOCO_GL=egl python tests/test_living_room_tasks.py
    MUJOCO_GL=egl python tests/test_living_room_tasks.py --envs TabletopPickPlace
"""

import argparse
import traceback

import numpy as np
from termcolor import colored

import robosuite

import robocasa  # noqa: F401  -- registers living-room envs
from robocasa.environments import ALL_LIVING_ROOM_ENVIRONMENTS


def test_living_room_tasks(
    env_names=None, robot="G1FloatingBody", num_rollouts=5, num_steps=10
):
    successful = []
    unsuccessful = []
    error_dict = {}

    if env_names is None or len(env_names) == 0:
        env_names = list(ALL_LIVING_ROOM_ENVIRONMENTS)

    for i, env_name in enumerate(env_names):
        print(colored(f"[{i+1}/{len(env_names)}] Testing {env_name}", "green"))

        completed = True
        try:
            env = robosuite.make(
                env_name=env_name,
                robots=robot,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=20,
                ignore_done=True,
                renderer="mujoco",
            )
            action = np.zeros(env.action_dim)
            for _ in range(num_rollouts):
                env.reset()
                for _ in range(num_steps):
                    env.step(action)
                # exercise the task's success check (must not raise)
                env._check_success()
            env.close()
        except KeyboardInterrupt:
            print(colored("Exiting Test Early.", "yellow"))
            break
        except Exception:
            print(colored(f"Test {env_name} Failed:\n{traceback.format_exc()}", "red"))
            error_dict.setdefault(env_name, []).append(traceback.format_exc())
            completed = False

        (successful if completed else unsuccessful).append(env_name)
        print()

    print(colored(f"The following tests ran successfully:\n{successful}\n", "green"))
    print(colored(f"The following tests ran unsuccessfully:\n{unsuccessful}", "red"))
    if error_dict:
        print(colored("With the following errors:", "red"))
        for key in error_dict:
            print(colored(f"{key}:", "red"))
            for j, error in enumerate(error_dict[key]):
                print(colored(f"Error #{j+1}:\n{error}\n", "red"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="(optional) living-room envs to test; default = all registered.",
    )
    parser.add_argument("--robot", type=str, default="G1FloatingBody")
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    test_living_room_tasks(
        env_names=args.envs,
        robot=args.robot,
        num_rollouts=args.num_rollouts,
        num_steps=args.num_steps,
    )
