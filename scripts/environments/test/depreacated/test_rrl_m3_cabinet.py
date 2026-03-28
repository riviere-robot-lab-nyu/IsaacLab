"""Script to test the control of rrl m3 robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to simulate.")

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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet.rrl_m3_cabinet_env_cfg import M3CabinetEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg: M3CabinetEnvCfg = parse_env_cfg(
            "RRL-M3-Cabinet-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing
    env_cfg.debug_env = True

    # create environment
    env = gym.make("RRL-M3-Cabinet-Direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    max_action_steps = 100
    current_step = 0
    sim_time = 0.0
    sim_dt = env.unwrapped.cfg.sim.dt
    # simulate environment
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device) # type: ignore
    cabinet = env.unwrapped.scene["cabinet"]
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Arm wave movement for testing gains 
            # actions[:, 2:4] = torch.ones(env.action_space.shape[0], 2, device=env.unwrapped.device)
            actions[:, 8:] = 1.0 * torch.sin(2 * torch.pi * torch.ones((env.action_space.shape[0], 7), device=env.unwrapped.device) * 0.5 * sim_time) # type: ignore

            ########## testing gripper open/close with sine wave ##########
            # signal = torch.zeros(env.action_space.shape[0], 7, device=env.unwrapped.device)
            # signal[:, -1] = 1
            # actions[:, 8:] = 0.004 * torch.sin(2 * torch.pi * signal * 0.5 * sim_time) # type: ignore
            
            # gripper_cmd = actions[:, -1]  # single scalar per env
            # close_mask = (gripper_cmd > 0.0).unsqueeze(-1)  # (num_envs, 1)
            # gripper_targets = torch.where(close_mask, 0, 0.04)
            ###################################################################
            
            # apply actions
            obs, rews, _, _, _ = env.step(actions)
            current_step += 1
            sim_time += sim_dt
            # print(f"[INFO]: Step: {current_step}, to_target: {obs['debug_to_target_dist']}, cabinet_joint_pos: {obs['debug_cabinet_joint_pos']}, rewards: {rews}")
            # print("gripper_targets:", gripper_targets)
            
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
