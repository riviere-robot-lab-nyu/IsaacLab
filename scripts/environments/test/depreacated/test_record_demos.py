# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful.
                              (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument(
    "--num_demos", type=int, default=3, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=200,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import logging
import os
import time

import gymnasium as gym
import torch

import omni.ui as ui

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils import configclass
# import logger
logger = logging.getLogger(__name__)


class ObsActionRecorderTerm(RecorderTerm):
    def record_pre_step(self):
        return "actions", self._env._actions[:, :-1]  # exclude last unused dim

    def record_post_step(self):
        return "obs", self._env.obs_buf

@configclass
class ObsActionRecorderTermCfg(RecorderTermCfg):
    class_type: type = ObsActionRecorderTerm

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories(dataset_file: str) -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(dataset_file)
    output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    task_name: str, output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1)
        env_cfg.env_name = task_name.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    # if hasattr(env_cfg.terminations, "success"):
    #     success_term = env_cfg.terminations.success
    #     env_cfg.terminations.success = None
    # else:
    #     logger.warning(
    #         "No success termination term was found in the environment."
    #         " Will not be able to mark recorded demos as successful."
    #     )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    # env_cfg.terminations.time_out = None
    # env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: RecorderManagerBaseCfg = RecorderManagerBaseCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.obs_action = ObsActionRecorderTermCfg()

    return env_cfg, success_term


def create_environment(task_name: str, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(task_name, cfg=env_cfg).unwrapped
        env.recorder_manager = RecorderManager(env_cfg.recorders, env)
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(xr=False)
    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
        instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """

    if success_step_count >= args_cli.num_success_steps:
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        print("Success condition met! Recording completed.")
        return 0, True


    return success_step_count, False


def handle_reset(
    env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


B = torch.tensor([[1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],   # Fx contribution from each thruster
                      [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0],   # Fy contribution from each thruster
                      [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]]) # (3, 3) control allocation matrix for surge, sway, yaw
M = torch.linalg.pinv(B) # (3, 8) pseudo-inverse for control allocation
def PD_controller(obs: dict):
    KP_POS = 1.5
    KD_POS = 7.5
    KP_ORI = 0.5
    KD_ORI = 7.5
    robot_states = obs['policy']
    lin_vel_b = robot_states[:, :3]
    ang_vel_b = robot_states[:, 3:6]
    desired_pos_b = robot_states[:, 6:9]
    desired_ori_b = robot_states[:, 9:13]
    F_x_des_b = KP_POS * desired_pos_b[:, 0] - KD_POS * lin_vel_b[:, 0] # (num_envs, )
    F_y_des_b = KP_POS * desired_pos_b[:, 1] - KD_POS * lin_vel_b[:, 1]
    roll, pitch, yaw = euler_xyz_from_quat(desired_ori_b)
    tau_z_des_b = KP_ORI * yaw - KD_ORI * ang_vel_b[:, 2]  # yaw control
    return torch.matmul(torch.stack([F_x_des_b, F_y_des_b, tau_z_des_b], dim=1), M.T.to(obs['policy'].device)) # (num_envs, 8)

def run_simulation_loop(
    env: gym.Env,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr

    # Reset before starting
    env.sim.reset()
    env.reset()

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)  # type: ignore
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                env.recorder_manager.record_pre_step()
                obs, reward, terminated, truncated, info = env.step(actions)
                actions[:, :8] = PD_controller(obs)
                success_step_count += 1
                env.recorder_manager.record_post_step()
                reset_env_ids = terminated.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    env.recorder_manager.record_pre_reset(reset_env_ids)
                    # env.step() already did the actual reset internally,
                    # so just notify the recorder
                    env.recorder_manager.record_post_reset(reset_env_ids)     
                    success_step_count = 0  # env terminated early, start fresh         
            else:
                env.sim.render()

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0  and \
                env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    
    rate_limiter = RateLimiter(hz=30)
    
    dataset_file_path = "./datasets/dataset.hdf5"
    # Set up output directories
    output_dir, output_file_name = setup_output_directories(dataset_file_path)
    
    task_name = "RRL-M3-Direct-v0"
    # Create and configure environment
    env_cfg, success_term = create_environment_config(task_name, output_dir, output_file_name)

    # Create environment
    env = create_environment(task_name, env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, rate_limiter)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {dataset_file_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
