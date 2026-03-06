"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
# import multiprocessing

# if multiprocessing.get_start_method() != "spawn":
#     multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, gr00tn1.6, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet_camera.rrl_m3_cabinet_camera_env_cfg import M3CabinetCameraEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.sensors import TiledCamera
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul

import leisaac  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def convert_gripper_command_to_env_format(gripper_command: torch.Tensor) -> torch.Tensor:
    # Input:  0 = open, 1 = close
    # Output: 0 = close, 0.044 = open
    # return (1.0 - gripper_command) * 0.044
    return torch.where(gripper_command < 0.5, torch.tensor(0.044), torch.tensor(0.0))


def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    def slice_at_index(data, idx: int):
        """Take the idx-th frame from the nested data structure."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes

def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # load hdf5 dataset for replaying episode
    hdf5_file = "/home/cpw/workspace/IsaacLab/datasets/sm_dataset_1_episodes.hdf5"
    episode_name = "demo_0"
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(hdf5_file)
    # episode_names = dataset_file_handler.get_episode_names()
    episode = dataset_file_handler.load_episode(episode_name, device='cpu')
    all_data = episode.data
    # remove images to save memory 
    del all_data['observation']['images']['base']
    del all_data['observation']['images']['wrist']
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10 or num_frames > 2000:
        raise ValueError(f"Episode {episode_name} has less than 10 frames or more than 2000 frames, use another episode")

    episode_list = split_episode(episode, num_frames)

    # parse configuration
    env_cfg: M3CabinetCameraEnvCfg = parse_env_cfg(
            "RRL-M3-Cabinet-Camera-Direct-v0",
            device=args_cli.device,
            num_envs=1,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    ######################
    # copy from original 
    # policy inference 
    ######################
    # modify configuration
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    # env_cfg.recorders = None
    #######################


    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing
    env_cfg.decimation = 1
    env_cfg.debug_env = False
    env_cfg.debug_vis = False
    env_cfg.replay_episode = True

    env_cfg.robot.init_state.pos = (2.0, 0.0, 0.01)

    # cabinet setting to make it easier to open
    cabinet_cfg = getattr(env_cfg, "cabinet", getattr(env_cfg.scene, "cabinet", None))
    
    if cabinet_cfg is not None:
        for actuator_name in cabinet_cfg.actuators.keys():
            cabinet_cfg.actuators[actuator_name].stiffness = 0.0      # Kills the spring
            cabinet_cfg.actuators[actuator_name].effort_limit_sim = 0.001 # Limits the max force
    else:
        raise ValueError("Cabinet configuration not found in environment configuration")

    # create environment
    env: gym.Env = gym.make("RRL-M3-Cabinet-Camera-Direct-v0", cfg=env_cfg).unwrapped
    env_camera_keys = ["base", "wrist"]
    
    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        raise NotImplementedError("GR00T N1.5 policy is not implemented yet, please use GR00T N1.6 or LeRobot policy for now.")
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif args_cli.policy_type == "gr00tn1.6":
        from leisaac.policy import Gr00t16ServicePolicyClient

        policy = Gr00t16ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=env_camera_keys, # this expects ["front", "wrist"], hardcoded if necessary
        )

    elif "lerobot" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"
        camera_infos={
                'wrist': (480, 640), 'base': (480, 640)
            }
        # policy_type = args_cli.policy_type.split("-")[1]
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos=camera_infos,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif args_cli.policy_type == "openpi":
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=env_camera_keys,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    env.reset()
    # obs_dict, _ = env.reset()
    controller.reset()

    robot = env.robot
    wrist_camera = env.scene["tiled_camera_wrist"]
    base_camera  = env.scene["tiled_camera_base"]

    ############################################
    # load initial pose here
    ############################################ 
    robot_init_state = episode_list[0]._data["observation"]["state"][-1]
    # Write pose directly to sim
    root_pose = robot_init_state[7:14].unsqueeze(0)   # (1, 7) pos + quat
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.unwrapped.device))
    robot.data.joint_pos_target[:, :7] = robot_init_state[:7].unsqueeze(0)  # write joint positions to sim
    ############################################ 
    # Propagate the write
    env.sim.step()
    env.scene.update(dt=env.physics_dt)

    frame = 0
    sim_time = 0.0
    sim_dt = env.cfg.sim.dt

    # record the results
    success_count, episode_count = 0, 1
    # initialization
    obs_dict = {}
    forces = torch.zeros((1, 1, 3), device=env.device)
    torques = torch.zeros((1, 1, 3), device=env.device)
    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    env.reset()
                    # obs_dict, _ = env.reset()
                    episode_count += 1
                    break
                cabinet = env.scene["cabinet"]

                # build observation dictionary for policy
                obs_dict = {
                    "policy": {
                        "joint_pos": torch.cat([env.robot.data.joint_pos[:, :-1], 
                                                env.robot.data.root_lin_vel_b[:, :2], 
                                                env.robot.data.root_ang_vel_b], dim=1), #torch.zeros((1, 12)),

                        "task_description": args_cli.policy_language_instruction,
                        "base": base_camera.data.output["rgb"],
                        "wrist": wrist_camera.data.output["rgb"],
                    },
                }

                actions = policy.get_action(obs_dict["policy"]).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    episode_data = episode_list[frame]
                    _action = episode_data._data["actions"][-1]
                    _action[-1] = convert_gripper_command_to_env_format(_action[-1])
                    print("policy gripper output:", action[:, -1])
                    # Fx, Fy, Tau_z
                    forces[:, 0, 0] = _action[0]
                    forces[:, 0, 1] = _action[1]
                    torques[:, 0, 2] = _action[2] 
                    
                    # apply actions
                    env.robot.instantaneous_wrench_composer.set_forces_and_torques(
                        forces=forces,
                        torques= torques,
                        body_ids=[0],
                    )
                    # 6 dof arm command
                    env.robot.data.joint_pos_target[:, :7] = _action[3:]

                    # env step and render
                    env.robot.write_data_to_sim()
                    env.sim.step()
                    env.scene.update(dt=env.physics_dt)
                    frame += 1
                    # print(f"[Evaluation] Episode {episode_count} Step {frame}, Action: {action.cpu().numpy()}")

                    # obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    # step the sim and render
                    if cabinet.data.joint_pos[0, 1] > 0.35 or frame >= num_frames: # check if the drawer is opened, this threshold can be adjusted based on the actual drawer configuration
                        success = True
                        break
                    # if reset_terminated[0]:
                    #     success = True
                    #     break
                    # if reset_time_outs[0]:
                    #     time_out = True
                    #     break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                episode_count += 1
                break
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
