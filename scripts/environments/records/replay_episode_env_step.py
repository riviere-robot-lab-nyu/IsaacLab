"""Script to test the control of rrl m3 robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import random

import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet.rrl_m3_cabinet_env_cfg import M3CabinetEnvCfg
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet_camera.rrl_m3_cabinet_camera_env_cfg import M3CabinetCameraEnvCfg
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_vel_cabinet_camera.rrl_m3_vel_cabinet_camera_env_cfg import M3VelCabinetCameraEnvCfg

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


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


def convert_gripper_command_to_env_format(gripper_command: torch.Tensor) -> torch.Tensor:
    # Input:  0 = open, 1 = close
    # Output: 0 = close, 0.044 = open
    return (1.0 - gripper_command) * 0.044

def main():
    """Zero actions agent with Isaac Lab environment."""

    # load hdf5 dataset for replaying episode
    hdf5_file = "./datasets/sm_dataset_vel_1_episodes.hdf5"
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(hdf5_file)
    episode_names = dataset_file_handler.get_episode_names()
    episode_name = random.choice(list(episode_names))
    print(f"loading episode: {episode_name}")
    episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
    print("episode loaded.")
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10 or num_frames > 2000:
        raise ValueError(f"Episode {episode_name} has less than 10 frames or more than 2000 frames, use another episode")

    episode_list = split_episode(episode, num_frames)

    # parse configuration
    env_cfg: M3VelCabinetCameraEnvCfg = parse_env_cfg(
            "RRL-M3-Vel-Cabinet-Camera-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing
    # env_cfg.decimation = 2
    env_cfg.debug_env = False
    env_cfg.debug_vis = False

    env_cfg.robot.init_state.pos = (2.0, 0.0, 0.01)

    cabinet_cfg = getattr(env_cfg, "cabinet", getattr(env_cfg.scene, "cabinet", None))
    
    if cabinet_cfg is not None:
        for actuator_name in cabinet_cfg.actuators.keys():
            cabinet_cfg.actuators[actuator_name].stiffness = 0.0      # Kills the spring
            cabinet_cfg.actuators[actuator_name].effort_limit_sim = 0.001 # Limits the max force
    else:
        raise ValueError("Cabinet configuration not found in environment configuration")

    # create environment
    env = gym.make("RRL-M3-Vel-Cabinet-Camera-Direct-v0", 
                   cfg=env_cfg
                   ).unwrapped
    # reset environment at start
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # Initialize the robot state to match the first frame of the episode
    robot_init_state = episode_list[0]._data["observation"]["state"][-1]
    robot = env.robot
    # Write pose directly to sim
    root_pose = robot_init_state[7:14].unsqueeze(0)   # (1, 7) pos + quat
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    robot.data.joint_pos_target[:, :7] = robot_init_state[:7].unsqueeze(0)  # write joint positions to sim

    # Propagate the write
    env.sim.step()
    env.scene.update(dt=env.physics_dt)

    # start from 5th frame
    frame = 0
    sim_time = 0.0
    sim_dt = env.cfg.sim.dt
    
    # simulate environment
    actions = torch.zeros(env.action_space.shape, device=env.device) # type: ignore
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            episode_data = episode_list[frame]
            actions = episode_data._data["actions"][-1]
            obs, rews, _, _, _ =env.step(actions[None, :]) # type: ignore

            frame += 1
            if frame >= num_frames:
                break
            sim_time += sim_dt
            recorded_state = episode_list[frame]._data["observation"]["state"][-1][7:10]
            live_state = env.unwrapped.robot.data.root_state_w[0]
            pos_err = torch.norm(recorded_state[:3] - live_state[:3])
            print(f"Frame {frame}: pos_error={pos_err:.4f}m")
            # print(f"[INFO]: Step: {current_step}, action: {_action}, rewards: {rews}")
            
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
