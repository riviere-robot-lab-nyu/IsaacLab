"""Script to test the control of rrl m3 robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from sympy import im

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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3.rrl_m3_env_cfg import M3EnvCfg
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet.rrl_m3_cabinet_env_cfg import M3CabinetEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_mul, matrix_from_quat

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg: M3EnvCfg = parse_env_cfg(
            "RRL-M3-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing
    env_cfg.debug_vis = False
    env_cfg.terrain.physics_material.static_friction = 5.0
    env_cfg.terrain.physics_material.dynamic_friction = 5.0
    env_cfg.terrain.physics_material.restitution = 1.0
    env_cfg.robot = env_cfg.robot.replace(
        init_state=env_cfg.robot.init_state.replace(
            pos=(1.0, 0.0, 0.01),
            rot=(0.0, 0.0, 0.0, 1.0),  
        )
    )

    # create environment
    env = gym.make("RRL-M3-Direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")


    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals_b = [
        [0.45, 0.0, 0.7, 1, 0, 0, 0],
    ]
    ee_goals_b = torch.tensor(ee_goals_b, device=env.unwrapped.device)

    # Create buffers to store actions
    ik_commands = torch.zeros(env.unwrapped.num_envs, diff_ik_controller.action_dim, device=env.unwrapped.device)
    ik_commands[:] = ee_goals_b[0]
    scene = env.unwrapped.scene
    robot = env.unwrapped.scene["robot"]

    # Resolving the scene entities
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[0-5]"], body_names=["link_6"])
    robot_entity_cfg.resolve(scene) 
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        jacobi_joint_ids = robot_entity_cfg.joint_ids
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] 
        jacobi_joint_ids = [idx + 6 for idx in robot_entity_cfg.joint_ids]
    # ee_jacobi_idx = robot.find_bodies("link_6")[0][0]  # returns (indices, names)
    current_step = 0
    sim_dt = env.unwrapped.cfg.sim.dt
    sim_time = 0.0
    # simulate environment
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device) # type: ignore
    lower = robot.data.joint_limits[0, robot_entity_cfg.joint_ids, 0]
    upper = robot.data.joint_limits[0, robot_entity_cfg.joint_ids, 1]
    # reset controller
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)
    root_pose_w_init = robot.data.root_pose_w 

    ee_goals_pos_w, ee_goals_quat_w = combine_frame_transforms(
                root_pose_w_init[:, 0:3], root_pose_w_init[:, 3:7],
                ik_commands[:, 0:3], ik_commands[:, 3:7]
            )
    R = matrix_from_quat(root_pose_w_init[:, 3:7])
    R_jacobi = torch.zeros((env.unwrapped.num_envs, 6, 6), device=env.unwrapped.device)
    R_jacobi[:, :3, :3] = R
    R_jacobi[:, 3:, 3:] = R
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jacobi_joint_ids]
            # import pdb; pdb.set_trace()  # Set a breakpoint to inspect variables before IK computation
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute ee frame in body frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute ee goal in body frame
            ee_goals_pos_b, ee_goals_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_goals_pos_w, ee_goals_quat_w
            )
            # ik_commands[:, :3] = ee_goals_pos_b
            # ik_commands[:, 3:] = ee_goals_quat_b
            # diff_ik_controller.set_command(ik_commands)

            # compute the joint commands
            joint_pos_des_ik = diff_ik_controller.compute(ee_pos_b, ee_quat_b, R_jacobi @ jacobian, joint_pos)
            actions[:, 4:-1] = joint_pos_des_ik.clamp(lower, upper) 

            # apply actions
            obs, rews, _, _, _ = env.step(actions)
            current_step += 1
            sim_time += sim_dt
            pos_error = ik_commands[:, :3] - ee_pos_b
            print(f"pos_error: {pos_error[0].cpu().numpy()}, norm: {pos_error[0].norm().item():.4f}")
            # print(f"[INFO]: Step: {current_step}, actions: {actions[0].cpu().numpy()}")
        
            goal_marker.visualize(ee_goals_pos_w, ee_goals_quat_w)
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
