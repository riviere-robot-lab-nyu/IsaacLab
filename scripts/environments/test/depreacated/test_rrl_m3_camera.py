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
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_camera.rrl_m3_camera_env_cfg import M3CameraEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg: M3CameraEnvCfg = parse_env_cfg(
            "RRL-M3-Camera-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing

    # create environment
    env = gym.make("RRL-M3-Camera-Direct-v0", cfg=env_cfg)
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
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if (current_step // 120) % 2 == 0: # every 60 steps, switch thruster direction
                actions[:, 8] = torch.ones(env.action_space.shape[0], device=env.unwrapped.device)
            else:
                actions[:, 8] = -torch.ones(env.action_space.shape[0], device=env.unwrapped.device)
            obs, rews, _, _, _ = env.step(actions)
            current_step += 1
            sim_time += sim_dt
            print(f"[INFO]: Step: {current_step}, obs: {obs['policy'].shape}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
