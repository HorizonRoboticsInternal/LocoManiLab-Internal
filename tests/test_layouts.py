"""Optional G1 fixture/layout inspection using already-installed scene assets.

This command-line utility is not part of the CPU unit-test selection.
"""

import argparse
import os
import traceback
from termcolor import colored

from robocasa.scripts.browse_mjcf_model import read_model
from robocasa.models.fixtures.fixture import FixtureType

from robocasa.utils.env_utils import create_env, run_random_rollouts
from robocasa.models.scenes.scene_registry import get_layout_path, get_style_path


TEST_ENVS = [
    dict(env_name="PnPCounterToMicrowave"),
    dict(env_name="PnPMicrowaveToCounter"),
    dict(env_name="OpenSingleDoor", door_id=FixtureType.MICROWAVE),
    dict(env_name="CloseSingleDoor", door_id=FixtureType.MICROWAVE),
    dict(env_name="TurnOnMicrowave"),
    dict(env_name="TurnOffMicrowave"),
    dict(env_name="PnPCounterToStove"),
    dict(env_name="PnPStoveToCounter"),
    dict(env_name="TurnOnStove"),
    dict(env_name="TurnOffStove"),
    dict(env_name="PnPCounterToSink"),
    dict(env_name="PnPSinkToCounter"),
    dict(env_name="TurnOnSinkFaucet"),
    dict(env_name="TurnOffSinkFaucet"),
    dict(env_name="TurnSinkSpout"),
    dict(env_name="CoffeeSetupMug"),
    dict(env_name="CoffeeServeMug"),
    dict(env_name="CoffeePressButton"),
    dict(env_name="Kitchen", init_robot_base_ref=FixtureType.DISHWASHER),
    dict(env_name="Kitchen", init_robot_base_ref=FixtureType.COFFEE_MACHINE),
    dict(env_name="Kitchen", init_robot_base_ref=FixtureType.FRIDGE),
    dict(env_name="Kitchen", init_robot_base_ref=FixtureType.TOASTER),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--layouts", type=str, nargs="+")
    parser.add_argument("--envs", type=str, nargs="+")

    args = parser.parse_args()


    for layout in args.layouts:
        for env_i, env_kwargs in enumerate(TEST_ENVS):
            env_name = env_kwargs["env_name"]

            # if envs is specified, only eval specified envs
            if args.envs is not None:
                if env_name not in args.envs:
                    continue

            log_dir = (
                f"/tmp/robocasa_test_layouts/layout_{layout}/env{env_i}_{env_name}"
            )
            os.makedirs(log_dir, exist_ok=True)

            env = None
            try:
                print(
                    colored(f"\n\nLayout: {layout}, Env{env_i}: {env_name}", "yellow")
                )
                env = create_env(
                    render_onscreen=False,
                    seed=0,  # set seed=None to run unseeded
                    layout_ids=layout,
                    translucent_robot=True,
                    **env_kwargs,
                )

                info = run_random_rollouts(
                    env,
                    num_rollouts=3,
                    num_steps=30,
                    video_path=f"{log_dir}/layout{layout}_env{env_i}_{env_name}.mp4",
                )

                env.close()
            except Exception as e:
                error = traceback.format_exc()
                print(error)
                error_log_path = os.path.join(log_dir, "error_log.txt")
                with open(error_log_path, "w") as f:
                    f.write(error)
                print(colored(f"Saved error log to: {error_log_path}", "yellow"))

                if env is not None:
                    env.close()
