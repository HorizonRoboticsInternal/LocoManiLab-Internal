"""Browse inherited kitchen layouts with the G1 whole-body viewer.

Scene assets and the selected controller weights must already be provisioned.
This command never downloads them. Use collect_demos_g1 / collect_fixed_task
for training-data collection.
"""

import argparse
from collections import OrderedDict

import numpy as np
import robosuite
from termcolor import colored

from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("{}s:".format(option_name.capitalize()))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i + 1, k, v))
        else:
            print("[{}] {}".format(i + 1, v))
    print()
    try:
        s = input(
            "Choose an option 1 to {}, or 0/other for default ({}): ".format(
                len(options),
                default_message,
            )
        )
        num = int(s)
        # 0 means use default (e.g. random layouts/styles)
        if num == 0:
            choice = default
        else:
            # parse 1-based input into 0-based index
            k = min(max(num - 1, 0), len(options) - 1)
            choice = list(options.keys())[k]
    except Exception as e:  # noqa: F841
        print(f"Invalid input: {e}")
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


from robocasa.demos.demo_empty_teleop import (
    _G1_CONTROL_DESCRIPTIONS,
    load_g1_controller_config,
    run_sonic_viewer_loop,
)
from robocasa.utils.robot_utils import DEFAULT_ROBOT, G1_ROBOTS
from g1_bridge.sonic_utils.ckpt import DEFAULT_SONIC_CKPT, SONIC_CKPT_CHOICES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Kitchen", help="registered kitchen task")
    parser.add_argument("--layout", type=int, help="kitchen layout ID")
    parser.add_argument("--style", type=int, help="kitchen style ID")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=G1_ROBOTS)
    parser.add_argument("--show-walls", action="store_true")
    parser.add_argument("--g1-control", default="keyboard", choices=list(_G1_CONTROL_DESCRIPTIONS))
    parser.add_argument("--pico-source", default="mock", choices=("mock", "real"))
    parser.add_argument("--sonic-ckpt", default=DEFAULT_SONIC_CKPT, choices=SONIC_CKPT_CHOICES)
    args = parser.parse_args()

    layouts = OrderedDict((item.value, item.name) for item in LayoutType if item.value >= 0)
    styles = OrderedDict((item.value, item.name) for item in StyleType if item.value >= 0)
    layout = args.layout if args.layout is not None else choose_option(
        layouts, "kitchen layout", default=-1, default_message="random layout"
    )
    style = args.style if args.style is not None else choose_option(
        styles, "kitchen style", default=-1, default_message="random style"
    )
    if layout == -1:
        layout = int(np.random.choice(list(layouts)))
    if style == -1:
        style = int(np.random.choice(list(styles)))

    env = robosuite.make(
        env_name=args.task,
        robots=args.robot,
        controller_configs=load_g1_controller_config(
            args.g1_control, args.pico_source, args.sonic_ckpt
        ),
        layout_ids=layout,
        style_ids=style,
        translucent_robot=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = EnclosingWallRenderWrapper(env, alpha=0.1, enabled=not args.show_walls)
    install_enclosing_wall_hotkeys(env)
    run_sonic_viewer_loop(env, show_ego=False)


if __name__ == "__main__":
    main()
