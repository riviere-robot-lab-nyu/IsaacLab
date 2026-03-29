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

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, subtract_frame_transforms, sample_uniform, euler_xyz_from_quat, quat_apply_inverse
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3.rrl_m3_env_cfg import M3EnvCfg
from isaaclab_tasks.utils import parse_env_cfg


class PDPositionController:
    """PD controller for planar position and yaw control of the RRL-M3 robot.

    Controls x, y position and yaw in world frame. Goals and states are given in
    world frame; forces are computed in body frame and mapped to per-thruster commands
    via the actuation matrix pseudoinverse.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device.
        output_mode: Return ``"thrust"`` (normalised per-thruster commands, default)
            or ``"force"`` (body-frame force/torque vector).
    """

    def __init__(self, num_envs, device, output_mode="thrust"):
        self.device = device
        self.num_envs = num_envs
        self.output_mode = output_mode
        kp = 10.0
        kd = 40.0
        kp_orient = kp
        kd_orient = kd
        self.kps = torch.tensor([[kp, kp, kp_orient]], device=device)
        self.kds = torch.tensor([[kd, kd, kd_orient]], device=device)

        U_MAX=1.7
        D_MOMENT=0.12
        D = U_MAX * torch.tensor([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0],
                                  [D_MOMENT, -D_MOMENT, D_MOMENT, -D_MOMENT]], device=device)
        self.D_T = D.T
        self.D_inv_T = torch.linalg.pinv(D).T
    
    def step(self, state, goal):
        """Compute control output for one timestep.

        Args:
            state: Robot root state in world frame ``(num_envs, 13)`` —
                ``root_state_w`` from Isaac Lab (pos, quat, lin_vel, ang_vel).
            goal: Desired ``[x, y, yaw]`` in world frame ``(num_envs, 3)``.

        Returns:
            Tuple of ``(output, error, velocity)`` where ``output`` is either
            normalised thrust commands or body-frame forces depending on
            ``output_mode``, each shaped ``(num_envs, 4)`` or ``(num_envs, 3)``.
        """
        quat = state[:, 3:7]
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        p = torch.cat([state[:, :2], yaw.unsqueeze(1)],dim=1)
        error = goal - p 
        error[:,2] = (error[:,2] + torch.pi) % (2*torch.pi) - torch.pi
        velocity = torch.cat([state[:,7:9], state[:,-1].unsqueeze(1)],dim=1)
        force = self.kps*error - self.kds*velocity
        # desired force in body frame
        forces_body = quat_apply_inverse(quat, torch.cat([force[:,:2],torch.zeros_like(force[:,0:1])], dim=1))
        forces_body[:, -1] = force[:, -1]
        # actions 
        thrust = forces_body@self.D_inv_T
        thrust = torch.clip(thrust, -1., 1.)
        force_body_real = thrust@self.D_T
        if self.output_mode == "force":
            return force_body_real, error, velocity   
        elif self.output_mode == "thrust": 
            return thrust, error, velocity
        else:
            raise ValueError(f"Invalid output mode: {self.output_mode}")


def main():
    """Positional PD control with rrl m3 robot in Isaac Lab environment."""
    # parse configuration
    env_cfg: M3EnvCfg = parse_env_cfg(
            "RRL-M3-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 30.0 # 100 seconds per episode for testing

    # create environment
    env = gym.make("RRL-M3-Direct-v0", cfg=env_cfg).unwrapped   
    # reset environment at start
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    current_step = 0
    controller = PDPositionController(args_cli.num_envs, env.device, output_mode="thrust")
    # simulate environment
    actions = torch.zeros(env.action_space.shape, device=env.device)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # apply actions
            obs, rews, _, _, _ = env.step(actions)
            robot_states = obs['policy']
            lin_vel_b = robot_states[:, :3]
            ang_vel_b = robot_states[:, 3:6]
            _, _, yaw_goal_w = euler_xyz_from_quat(env._desired_ori_w)
            goal = torch.cat([env._desired_pos_w[:, :2], yaw_goal_w.unsqueeze(1)], dim=1)
            actions[:, :4], error, velocity = controller.step(state=env.robot.data.root_state_w,
                                                              goal=goal)
            current_step += 1
            err = error.squeeze(1).cpu().numpy()[0]
            print(f"[INFO]: Step: {current_step:4d} | error  x: {err[0]:+.4f}  y: {err[1]:+.4f}  z: {err[2]:+.4f}  |norm|: {(err**2).sum()**0.5:.4f}")


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
