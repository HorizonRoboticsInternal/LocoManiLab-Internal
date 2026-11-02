import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Type
import sched

import tyro
import yaml
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.controllers.keyboard import KeyboardController
from sim2real.rl_policy.controllers.passive import PassiveController
from sim2real.rl_policy.controllers.pico import PicoController
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController
from sim2real.rl_policy.inference import Timer, build_inference_module
from sim2real.rl_policy.control_mode import (
    EpisodeCommand,
    resolve_keyboard_episode_command,
)
from sim2real.rl_policy.observations import Observation, ObsGroup, normalize_observation_array
from sim2real.rl_policy.recording import EpisodeRecorderBase, copy_array
from sim2real.rl_policy.recording.npz_writer import NpzEpisodeRecorder
from sim2real.rl_policy.robot_io import create_robot_io
from sim2real.rl_policy.utils.command_sender import ActionManager
from sim2real.rl_policy.utils.state_processor import StateProcessor
from sim2real.utils.common import PORTS
from sim2real.utils.dex3 import (
    DEFAULT_DOMAIN_ID as DEX3_DEFAULT_DOMAIN_ID,
    DEFAULT_INTERFACE as DEX3_DEFAULT_INTERFACE,
    DEFAULT_KD as DEX3_DEFAULT_KD,
    DEFAULT_KP as DEX3_DEFAULT_KP,
    DEFAULT_MAX_DELTA_Q as DEX3_DEFAULT_MAX_DELTA_Q,
    DEFAULT_THUMB_KD as DEX3_DEFAULT_THUMB_KD,
    DEFAULT_THUMB_KP as DEX3_DEFAULT_THUMB_KP,
)
from sim2real.utils.profiling import ScopedTimer
from sim2real.utils.strings import resolve_matching_names_values


class BasePolicy:
    def __init__(
        self,
        args: "BasePolicyArgs",
    ):
        self.args = args
        self.robot_cfg = get_robot_cfg(args.robot)
        with open(args.policy_config) as file:
            policy_config = yaml.load(file, Loader=yaml.FullLoader)
        policy_config = self.prepare_policy_config(policy_config)
        model_path = policy_config.get("model_path")
        if model_path is None:
            model_path = args.policy_config.replace(".yaml", ".onnx")
        else:
            model_path = str(Path(args.policy_config).parent / str(model_path))
        self.policy_config = policy_config
        self.model_path = model_path
        # initialize robot related processes
        self.joint_names_simulation = list(policy_config["joint_names_simulation"])
        self.body_names_simulation = list(policy_config["body_names_simulation"])

        self.robot_io_mode = args.robot_io
        self.robot_io = create_robot_io(
            mode=args.robot_io,
            robot_name=args.robot,
            robot_cfg=self.robot_cfg,
            interface=args.robot_interface,
        )

        self.state_processor = StateProcessor(
            self.robot_cfg,
            robot_io=self.robot_io,
        )
        self.action_manager = ActionManager(
            self.robot_cfg,
            policy_config,
            robot_io=self.robot_io,
        )
        self.rl_dt = 1.0 / float(args.rl_rate)
        self.inference_backend = args.inference_backend

        self.num_dofs = len(self.joint_names_simulation)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.action_manager.joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.joint_names_simulation))
        self.default_dof_angles[joint_indices] = default_joint_pos

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.action_manager.joint_names.index(name)
            for name in self.policy_joint_names
        ]

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_actions))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.policy_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        elif isinstance(action_scale_cfg, list):
            if len(action_scale_cfg) != self.num_actions:
                raise ValueError(
                    f"Action scale list length {len(action_scale_cfg)} does not match num actions {self.num_actions}"
                )
            self.action_scale[:] = np.array(action_scale_cfg)
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.init_count = 0
        # Perf metrics dict is reused; initialize early so background threads can record.
        self.perf_dict: Dict[str, float] = {}
        self.key_pressed: set[str] = set()
        self.state_dict = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "zero",
        }

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                self.robot_cfg.joint_pos_lower_limit,
                self.joint_names_simulation,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                self.robot_cfg.joint_pos_upper_limit,
                self.joint_names_simulation,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        self.controller_type = args.controller
        self.controller = self._build_controller()
        self.use_joystick = self.controller_type == "joystick"
        self.wc_msg = None

        self.dex3_driver = None
        self._dex3_grip_client = None
        if args.dex3_hand_dir is not None:
            from sim2real.rl_policy.recording.hand_grip_client import HandGripClient

            # Falls back to the same local default scripts/dh116s_control.py and
            # scripts/dex3_control.py already use, so --dex3-hand-dir alone is
            # enough when the Pico publisher runs on this machine; only pass
            # --hand-grip-connect explicitly for a remote publisher.
            hand_grip_connect = args.hand_grip_connect or f"tcp://127.0.0.1:{PORTS['hand_grip']}"
            self._dex3_grip_client = HandGripClient(connect=hand_grip_connect)
            if args.dex3_dry_run:
                from sim2real.utils.dex3 import DryRunDex3HandDriver

                self.dex3_driver = DryRunDex3HandDriver(hand_dir=args.dex3_hand_dir)
            else:
                from sim2real.utils.dex3 import Dex3HandDriver

                self.dex3_driver = Dex3HandDriver(
                    hand_dir=args.dex3_hand_dir,
                    domain_id=args.dex3_domain_id,
                    interface=args.dex3_interface,
                    kp=args.dex3_kp,
                    kd=args.dex3_kd,
                    thumb_kp=args.dex3_thumb_kp,
                    thumb_kd=args.dex3_thumb_kd,
                    max_delta_q=args.dex3_max_delta_q,
                )

        # Setup observations after state processor is initialized
        self.setup_policy(model_path)
        self.setup_observations(policy_config["observation"])
        self._record_enabled = bool(args.record)
        self._record_output = self._resolve_record_output(args.record_output)
        self._recorder: EpisodeRecorderBase | None = (
            self._build_recorder() if self._record_enabled else None
        )
        self._last_extra_keys: list[str] = []

    def prepare_policy_config(self, policy_config):
        return policy_config

    def _build_controller(self) -> ControllerBase:
        self.keyboard_controller = None
        self.joystick_controller = None
        self.pico_controller = None

        controller_type = self.controller_type
        if controller_type == "keyboard":
            print("Using keyboard")
            self.keyboard_controller = KeyboardController()
            self.key_pressed = self.keyboard_controller.key_pressed
            return self.keyboard_controller

        if controller_type == "joystick":
            print("Using joystick")
            self.joystick_controller = UnitreeJoystickController(
                self.robot_cfg,
                self.perf_dict,
            )
            print("Wireless Controller Initialized")
            return self.joystick_controller

        if controller_type == "pico":
            self.pico_controller = PicoController(connect=self.args.pico_zmq_connect)
            return self.pico_controller

        if controller_type == "passive":
            # No operator input at all: mode changes come from whoever embeds
            # this runtime (a simulator host, a VLA driver) calling
            # set_policy_mode()/set_zero_mode() directly. Unlike "keyboard",
            # this starts no stdin listener thread.
            return PassiveController()

        raise ValueError(f"Unsupported controller_type: {controller_type}")

    def setup_policy(self, model_path):
        runtime_module = build_inference_module(model_path, self.inference_backend)
        runtime_label = self.inference_backend
        if self.inference_backend == "tensorrt":
            runtime_label = (
                "tensorrt"
                f"[fp16={runtime_module.use_fp16}, "
                f"workspace={runtime_module.workspace_size}]"
            )

        logger.info("Using policy inference backend {}", runtime_label)

        def policy(input_dict):
            output_dict = runtime_module(input_dict)
            action = np.asarray(output_dict["action"], dtype=np.float32)
            next_state_dict = {
                k[1]: v
                for k, v in output_dict.items()
                if isinstance(k, tuple) and len(k) == 2 and k[0] == "next"
            }
            input_dict.update(next_state_dict)

            q_target = self.default_dof_angles.copy()
            q_target[self.controlled_joint_indices] += action * self.action_scale

            return action, q_target, input_dict

        self.policy = policy

    def setup_observations(self, obs_cfg):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []

        # Create observation instances based on config
        for obs_group, obs_items in obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                print(f"\t{obs_name}: {obs_config}")
                obs_config = dict(obs_config)
                obs_key = obs_config.pop("_target_", obs_name)
                obs_class: Type[Observation] = Observation.resolve(obs_key)
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        self.state_dict["paused"] = True
        # A new episode has no previous action, and saying so is not a
        # formality. Every prev-action observation appends state_dict["action"]
        # in update(), which runs BEFORE inference -- so whatever is left here
        # is the first frame of the new episode's action history. Leave the old
        # value and episode N opens with episode N-1's final command fed back
        # into the policy, one tick after the robot was teleported somewhere
        # else entirely. ``run()`` rebuilds state_dict per session and never saw
        # this; a host that drives step() directly across episodes (the
        # g1_bridge composites) resets in place and does.
        #
        # Measured on gr00t bench_sonic_KitchenOpenFridge_v2 / sonic
        # (2026-08-09): the stale action moves the first decoded target by up to
        # 1.9 rad and costs ~30 points of task success. It is invisible in
        # episode 1 -- which is exactly why single-episode parity tests against
        # the vendored WBT controller (which zeroes its own action history on
        # reset) came back bit-identical.
        self.state_dict["action"] = np.zeros(self.num_actions, dtype=np.float32)
        [reset_callback() for reset_callback in self.reset_callbacks]

    def update(self):
        [update_callback(self.state_dict) for update_callback in self.update_callbacks]

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        obs_components: dict[str, dict[str, np.ndarray]] = {}
        for obs_group in self.observations.values():
            group_components: dict[str, np.ndarray] = {}
            group_values: list[np.ndarray] = []
            for obs_name, obs_func in obs_group.funcs.items():
                obs = normalize_observation_array(obs_func.compute())
                group_components[obs_name] = obs
                group_values.append(obs)

            obs_components[obs_group.name] = group_components
            obs_dict[obs_group.name] = np.concatenate(group_values, axis=-1)

        return obs_dict, obs_components

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def set_init_mode(self, *, source: str) -> None:
        self.init_count = 0
        self.state_dict["control_mode"] = "init"
        logger.info(f"Control mode set to init via {source}")

    def set_zero_mode(self, *, source: str) -> None:
        self.state_dict["control_mode"] = "zero"
        logger.info(f"Control mode set to zero via {source}")

    def set_policy_mode(self, *, source: str) -> None:
        self.reset()
        self.state_dict["control_mode"] = "policy"
        logger.info(f"Control mode set to policy via {source}")

    def process_controllers(self) -> None:
        if self.joystick_controller is not None:
            self.wc_msg = self.joystick_controller.state

        mode = self.controller.get_control_mode()
        if mode == "policy":
            self.set_policy_mode(source=self.controller.name)
        elif mode == "zero":
            self.set_zero_mode(source=self.controller.name)
        elif mode == "init":
            self.set_init_mode(source=self.controller.name)

        # Cached (not re-queried) so subclasses overriding process_controllers can
        # inspect the same edge-triggered keys without double-draining the listener's
        # pending queue (KeyboardController.get_extra_keys() clears it on each call).
        self._last_extra_keys = self.controller.get_extra_keys()
        episode_command: EpisodeCommand | None = None
        for keycode in self._last_extra_keys:
            resolved = resolve_keyboard_episode_command(keycode)
            if resolved is not None:
                episode_command = resolved

        if self.pico_controller is not None:
            pico_command = self.pico_controller.get_episode_command()
            if pico_command is not None:
                episode_command = pico_command

        if episode_command == "toggle":
            self._toggle_recording(source=self.controller.name)
        elif episode_command == "discard":
            self._discard_episode(source=self.controller.name)

    def _toggle_recording(self, *, source: str) -> None:
        """Start a new episode (if idle) or stop & save the in-progress one. No-op
        with no recorder attached, or for recorder backends/settings that don't
        support mid-run segmentation (see ``EpisodeRecorderBase.toggle``)."""
        if self._recorder is not None:
            self._recorder.toggle(source=source)

    def _discard_episode(self, *, source: str) -> None:
        """Drop the in-progress episode without saving it. No-op with no recorder."""
        if self._recorder is not None:
            self._recorder.discard(source=source)

    def _resolve_record_output(self, record_output: str | None) -> Path:
        if record_output:
            return Path(record_output).expanduser()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        policy_stem = Path(self.args.policy_config).stem
        if self.args.record_format == "lerobot":
            return Path.cwd() / f"policy_tracking_record_{policy_stem}_{timestamp}"
        return Path.cwd() / f"policy_tracking_record_{policy_stem}_{timestamp}.npz"

    def _build_recorder(self) -> EpisodeRecorderBase:
        if self.args.record_format == "npz":
            return NpzEpisodeRecorder(
                policy=self,
                output_path=self._record_output,
                episodes_enabled=self.args.record_episodes,
            )
        if self.args.record_format == "lerobot":
            from sim2real.rl_policy.recording.lerobot_writer import build_lerobot_recorder

            return build_lerobot_recorder(self)
        raise ValueError(f"Unsupported record_format: {self.args.record_format}")

    def _append_record_frame(
        self,
        *,
        obs_dict: Dict[str, np.ndarray],
        obs_components: dict[str, dict[str, np.ndarray]],
        action: np.ndarray,
        q_target: np.ndarray,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
    ) -> None:
        if self._recorder is None:
            return

        now_ns = time.time_ns()

        robot_state = self.state_processor.latest_state
        joint_torque = robot_state.joint_torque if robot_state is not None else None
        if joint_torque is None:
            joint_torque = np.full(self.num_dofs, np.nan, dtype=np.float32)

        self._recorder.append_frame(
            {
                "step_index": int(self.total_inference_cnt),
                "time_ns": int(now_ns),
                "robot": {
                    "joint_pos": copy_array(self.state_processor.joint_pos),
                    "joint_vel": copy_array(self.state_processor.joint_vel),
                    "joint_torque": copy_array(joint_torque),
                    "root_quat_w": copy_array(self.state_processor.root_quat_w),
                    "root_ang_vel_b": copy_array(self.state_processor.root_ang_vel_b),
                    "low_state_tick": int(robot_state.tick if robot_state is not None else -1),
                },
                "policy": {
                    "action": copy_array(action),
                    "q_target": copy_array(q_target),
                    "cmd_q": copy_array(cmd_q),
                    "cmd_dq": copy_array(cmd_dq),
                    "cmd_tau": copy_array(cmd_tau),
                    "obs": {
                        group_name: {
                            obs_name: copy_array(obs)
                            for obs_name, obs in group_obs.items()
                        }
                        for group_name, group_obs in obs_components.items()
                    },
                },
                "runtime": {
                    "control_mode": str(self.state_dict.get("control_mode", "")),
                    "paused": bool(self.state_dict.get("paused", False)),
                    **self._record_runtime_fields(),
                },
            }
        )

    def _record_runtime_fields(self) -> dict[str, Any]:
        return {}

    def _update_dex3_hand(self) -> None:
        if self.dex3_driver is None:
            return
        grips = self._dex3_grip_client.read()
        if grips is None:
            return
        left_grip, right_grip = grips
        self.dex3_driver.apply_grips(left_grip, right_grip)

    def _record_metadata_fields(self) -> dict[str, Any]:
        return {}

    def run(self):
        total_inference_cnt = 0

        self.state_dict = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "zero",
        }
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                scheduler.enterabs(next_run_time, 1, self.step, ())
                scheduler.run()
                
                next_run_time += self.rl_dt
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    print(f"total_inference_cnt: {self.total_inference_cnt}")
                    for key, value in self.perf_dict.items():
                        print(f"\t{key}: {value/100*1000:.3f} ms")
                    self.perf_dict = {}
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if self._recorder is not None:
                    self._recorder.close()
            finally:
                try:
                    self.controller.close()
                finally:
                    try:
                        if self.dex3_driver is not None:
                            self.dex3_driver.close()
                    finally:
                        try:
                            if self._dex3_grip_client is not None:
                                self._dex3_grip_client.close()
                        finally:
                            self.robot_io.close()

    def step(self):
        self._update_dex3_hand()
        with ScopedTimer("rl_policy.step") as step_timer:
            with ScopedTimer("rl_policy.step.prepare_low_state") as prepare_low_state_timer:
                with Timer(self.perf_dict, "prepare_low_state"):
                    with ScopedTimer(
                        "rl_policy.step.prepare_low_state.process_controllers"
                    ) as process_controllers_timer:
                        with Timer(self.perf_dict, "process_controllers"):
                            self.process_controllers()

                    with ScopedTimer(
                        "rl_policy.step.prepare_low_state.get_low_state"
                    ) as get_low_state_timer:
                        with Timer(self.perf_dict, "get_low_state"):
                            if not self.state_processor._prepare_low_state():
                                print("low state not ready.")
                                return

            try:
                with ScopedTimer("rl_policy.step.prepare_obs") as prepare_obs_timer:
                    with Timer(self.perf_dict, "prepare_obs"):
                        # Prepare observations
                        self.update()
                        obs_dict, obs_components = self.prepare_obs_for_rl()
                        runtime_control_mode = self.state_dict["control_mode"]
                        self.state_dict.update(obs_dict)
                        self.state_dict["is_init"] = np.zeros(1, dtype=bool)

                with ScopedTimer("rl_policy.step.policy") as policy_timer:
                    with Timer(self.perf_dict, "policy"):
                        # Inference
                        action, q_target, self.state_dict = self.policy(self.state_dict)
                        # for key, value in self.state_dict.items():
                        #     if key.endswith("_ood_ratio"):
                        #         print(key, value)
                        # Clip policy action
                        # action = action.clip(-100, 100)
                        self.state_dict["action"] = action
                        self.state_dict["q_target"] = q_target
                        self.state_dict["control_mode"] = runtime_control_mode
            except Exception as e:
                print(f"Error in policy inference: {e}")
                # print traceback for debugging
                import traceback
                traceback.print_exc()
                self.state_dict["action"] = np.zeros(self.num_actions)
                return

            with ScopedTimer("rl_policy.step.rule_based_control_flow") as rule_based_timer:
                with Timer(self.perf_dict, "rule_based_control_flow"):
                    with ScopedTimer(
                        "rl_policy.step.rule_based_control_flow.select_target"
                    ) as select_target_timer:
                        # rule based control flow
                        control_mode = self.state_dict["control_mode"]
                        if control_mode == "init":
                            q_target = self.get_init_target()
                        elif control_mode == "zero":
                            q_target = self.state_processor.joint_pos
                        elif control_mode == "policy":
                            q_target = self.state_dict["q_target"]
                        else:
                            raise ValueError(f"Invalid control mode: {control_mode}")

                        # # Clip q target
                        # q_target = np.clip(
                        #     q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
                        # )

                        # Send command
                        cmd_q = q_target
                        cmd_dq = np.zeros(self.num_dofs)
                        cmd_tau = np.zeros(self.num_dofs)

                    with ScopedTimer(
                        "rl_policy.step.rule_based_control_flow.send_command"
                    ) as send_command_timer:
                        self.action_manager.send_command(cmd_q, cmd_dq, cmd_tau)
                        self._append_record_frame(
                            obs_dict=obs_dict,
                            obs_components=obs_components,
                            action=action,
                            q_target=q_target,
                            cmd_q=cmd_q,
                            cmd_dq=cmd_dq,
                            cmd_tau=cmd_tau,
                        )

        elapsed = step_timer.last_time
        if elapsed > self.rl_dt:
            logger.warning(
                (
                    "RL step took {:.3f} ms, expected {:.3f} ms. "
                    "breakdown: prepare_low_state={:.3f} ms "
                    "(process_controllers={:.3f}, get_low_state={:.3f}), "
                    "prepare_obs={:.3f} ms, policy={:.3f} ms, "
                    "rule_based_control_flow={:.3f} ms "
                    "(select_target={:.3f}, send_command={:.3f})"
                ),
                elapsed * 1000.0,
                self.rl_dt * 1000.0,
                prepare_low_state_timer.last_time * 1000.0,
                process_controllers_timer.last_time * 1000.0,
                get_low_state_timer.last_time * 1000.0,
                prepare_obs_timer.last_time * 1000.0,
                policy_timer.last_time * 1000.0,
                rule_based_timer.last_time * 1000.0,
                select_target_timer.last_time * 1000.0,
                send_command_timer.last_time * 1000.0,
            )


@dataclass
class BasePolicyArgs:
    """Robot."""

    policy_config: str
    robot: str = "g1"
    rl_rate: float = 50.0
    inference_backend: Literal["onnx-gpu", "onnx-cpu", "tensorrt"] = "onnx-cpu"
    robot_io: Literal["inline", "zmq", "push"] = "zmq"
    """'push' has no robot behind it -- the embedding host pushes state in and
    reads the command back off PushRobotIO.last_command (see robot_io/push.py)."""
    robot_interface: str = "eth0"
    controller: Literal["keyboard", "joystick", "pico", "passive"] = "keyboard"
    """'passive' takes no operator input and starts no listener thread; the
    embedding host drives the control mode directly."""
    pico_zmq_connect: str = f"tcp://127.0.0.1:{PORTS['pico_controller']}"
    record: bool = False
    record_output: str | None = None
    record_format: Literal["npz", "lerobot"] = "npz"
    """'npz' preserves the original flat-dump behavior; 'lerobot' writes a LeRobot v2.1
    episode dataset (state/action/image), deriving its schema from this family's own
    `observation` config -- see rl_policy.recording.lerobot_writer."""
    record_episodes: bool = False
    """Segment recording into multiple episodes via the controller's toggle/discard
    command ('c'/'x' on keyboard, Y/Y+A on pico) instead of one implicit episode
    spanning the whole process. Always True when record_format == 'lerobot'."""
    record_task: str | None = None
    """LeRobot task string for recorded episodes (record_format == 'lerobot' only).
    Defaults to the policy_config filename stem."""
    camera_connect: str | None = None
    """ZMQ connect address for a sim2real.camera server (e.g. tcp://127.0.0.1:5594).
    Only used when record_format == 'lerobot'; omit to record without images."""
    hand_grip_connect: str | None = None
    """ZMQ connect address for the existing hand-grip channel (PORTS['hand_grip']).
    Used for recording (record_format == 'lerobot') and, when dex3_hand_dir is set,
    as the live source driving the in-process Dex3 hand. Omit to use the local
    default (tcp://127.0.0.1:5593); only needed when the Pico publisher runs on a
    different machine."""
    record_extra: list[str] = field(default_factory=list)
    """Extra controller-specific recording fields to enable (see
    rl_policy.recording.extras), on top of whatever this policy_config's own
    'recording_extras' key already declares. Only used when record_format == 'lerobot'."""
    dex3_hand_dir: Literal["left", "right", "double"] | None = None
    """Enable in-process Dex3 hand control (no separate scripts/dex3_control.py
    process needed): drives the Dex3 hand over DDS every tick from the same
    hand-grip stream Pico already publishes -- add just this one flag, everything
    else (hand_grip_connect, domain/interface, gains) defaults to the common case.
    When enabled with record_format == 'lerobot', the 'dex3' RecordingExtra (7x2
    joint targets and measured angles) is added automatically -- no need to also
    pass --record-extra dex3. Only combine with --robot-io inline after checking
    for DDS-participant conflicts with the debug-mode handshake in
    robot_io/g1_debug_mode.py; safe by default since robot_io defaults to 'zmq'."""
    dex3_dry_run: bool = False
    """Map grips to joint targets and record them (see dex3_hand_dir) without
    touching unitree_sdk2py/DDS at all -- no domain/interface/network-card needed.
    Use this for sim2sim/dev-machine testing where there's no real Dex3 hand or
    matching DDS network interface (e.g. no eth0)."""
    dex3_domain_id: int = DEX3_DEFAULT_DOMAIN_ID
    dex3_interface: str = DEX3_DEFAULT_INTERFACE
    dex3_kp: float = DEX3_DEFAULT_KP
    dex3_kd: float = DEX3_DEFAULT_KD
    dex3_thumb_kp: float = DEX3_DEFAULT_THUMB_KP
    dex3_thumb_kd: float = DEX3_DEFAULT_THUMB_KD
    dex3_max_delta_q: float = DEX3_DEFAULT_MAX_DELTA_Q

if __name__ == "__main__":
    args = tyro.cli(BasePolicyArgs)
    policy = BasePolicy(args=args)
    policy.run()
