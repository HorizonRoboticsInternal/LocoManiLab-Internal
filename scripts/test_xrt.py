"""XRoboToolkit SDK stream health check."""

import argparse
import time

import numpy as np
import xrobotoolkit_sdk as xrt


def _abs_sum(value):
    return float(np.abs(np.asarray(value)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument(
        "--buttons",
        action="store_true",
        help="Also poll controller button/axis getters used by pico-manager.",
    )
    args = parser.parse_args()

    xrt.init()
    try:
        deadline = time.monotonic() + args.seconds
        i = 0
        while time.monotonic() < deadline:
            headset = np.asarray(xrt.get_headset_pose())
            body = np.asarray(xrt.get_body_joints_pose())
            motion = np.asarray(xrt.get_motion_tracker_pose())
            button_text = ""
            if args.buttons:
                button_text = (
                    f" A={xrt.get_A_button()} B={xrt.get_B_button()} "
                    f"X={xrt.get_X_button()} Y={xrt.get_Y_button()} "
                    f"Laxis={xrt.get_left_axis()} Raxis={xrt.get_right_axis()} "
                    f"Lclick={xrt.get_left_axis_click()} "
                    f"Rclick={xrt.get_right_axis_click()} "
                    f"Ltrig={xrt.get_left_trigger():.3f} "
                    f"Rtrig={xrt.get_right_trigger():.3f} "
                    f"Lgrip={xrt.get_left_grip():.3f} "
                    f"Rgrip={xrt.get_right_grip():.3f}"
                )
            print(
                f"[{i:03d}] "
                f"head_ts={xrt.get_time_stamp_ns()} "
                f"head_sum={_abs_sum(headset):.4f} "
                f"body_avail={xrt.is_body_data_available()} "
                f"body_ts={xrt.get_body_timestamp_ns()} "
                f"body_shape={body.shape} "
                f"body_sum={_abs_sum(body):.4f} "
                f"motion_n={xrt.num_motion_data_available()} "
                f"motion_ts={xrt.get_motion_timestamp_ns()} "
                f"motion_sum={_abs_sum(motion):.4f} "
                f"serials={xrt.get_motion_tracker_serial_numbers()} "
                f"Lhand={xrt.get_left_hand_is_active()} "
                f"Rhand={xrt.get_right_hand_is_active()}"
                f"{button_text}",
                flush=True,
            )
            i += 1
            time.sleep(args.period)
    finally:
        xrt.close()


if __name__ == "__main__":
    main()
