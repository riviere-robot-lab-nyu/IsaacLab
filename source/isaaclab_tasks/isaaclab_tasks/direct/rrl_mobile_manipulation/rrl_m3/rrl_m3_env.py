"""Thruster Cylinder Environment for Isaac Lab."""

from __future__ import annotations

import torch
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, subtract_frame_transforms, quat_from_euler_xyz
##
# Pre-defined configs
##
from isaaclab.markers import CUBOID_MARKER_CFG, POSITION_GOAL_MARKER_CFG, SPHERE_MARKER_CFG, FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.250, 0.250, 0.250)

from .rrl_m3_env_cfg import M3EnvCfg  # isort: skip


class M3Env(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: M3EnvCfg
    
    def __init__(self, cfg: M3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialization
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        

        # DEBUG: Print joint information
        print(f"Total joints: {self.robot.num_joints}")  
        print(f"Joint names: {self.robot.joint_names}")  
        # Check actuated joints
        actuated_joints = []
        for actuator in self.robot.actuators.values():
            actuated_joints.extend(actuator.joint_names)
        print(f"Actuated joints: {len(actuated_joints)}")  
        print(f"Actuated joint names: {actuated_joints}")  

        # Arm joint position targets
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        # Base goal position targets in world frame
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Goal orientation 
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device) 
        
        # Track RL episode statistics
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # Add thruster stuff for velocity control
        kp = 100
        kp_orient = 0.5 * kp
        self.kps = torch.tensor([[kp, kp, kp_orient]], device=self.device)
        U_MAX = 1.7
        D_MOMENT = 0.12
        D = U_MAX * torch.tensor([[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0],
                                  [D_MOMENT, -D_MOMENT, D_MOMENT, -D_MOMENT]], device=self.device)
        self.D_T = D.T
        self.D_inv_T = torch.linalg.pinv(D).T

        self._applied_forces = torch.zeros((self.num_envs, 3), device=self.device)
        self._applied_torques = torch.zeros((self.num_envs, 3), device=self.device)
    
    def _setup_scene(self):
        """Set up the scene with robot and ground."""
        # Add robot to scene
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self._terrain = self.scene["terrain"]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor): 
        """Process actions to compute thruster forces and torques."""
        self._actions = actions.clone()
        
        thrust = torch.clip(self._actions[:, :4], -1. ,1.)
        wrench = thrust @ self.D_T

        self._applied_forces[:, :2] = wrench[:, :2]
        self._applied_torques[:, 2] = wrench[:, -1]

        self.robot_dof_targets[:, :-1] = self._actions[:, 4:]  
        
    
    def _apply_action(self):
        """Apply the computed forces and torques to the robot."""
        # Get robot body indices
        body_ids = self.robot.find_bodies("alpha3_v4_part_glb")[0] # Cylinder, alpha3_v4_part_glb
        
        # Apply external wrench (force + torque) to robot
        forces = self._applied_forces.unsqueeze(1)  # (num_envs, 1, 3)
        torques = self._applied_torques.unsqueeze(1)  # (num_envs, 1, 3)
        
        
        # TODO: consider applying force to specific poistion, not just center of mass, see Isaac Lab docs.
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
        )

        # apply arm joint position targets
        self.robot.set_joint_position_target(self.robot_dof_targets) 
    

    def _get_observations(self) -> dict:
      
        desired_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, self._desired_pos_w
        )
        _, desired_ori_b = subtract_frame_transforms( 
            torch.zeros_like(self.robot.data.root_pos_w), self.robot.data.root_quat_w, torch.zeros_like(self._desired_pos_w), self._desired_ori_w
        )

        obs = torch.cat([
            self.robot.data.root_lin_vel_b,       # (num_envs, 3)
            self.robot.data.root_ang_vel_b,       # (num_envs, 3)
            desired_pos_b,                         # (num_envs, 3)
            desired_ori_b,                        # (num_envs, 4)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self.robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check for episode termination.
        
        Episodes terminate when:
        1. Time limit reached
        2. Robot falls below ground (z < -0.1) to prevent weird artifacts
        3. Robot tips over (large roll/pitch angle)
        """
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Robot fell
        fell = self.robot.data.root_pos_w[:, 2] < -0.1
        
        # Robot tipped over (check if up vector is pointing down)
        # Get the up vector in world frame by rotating [0, 0, 1] by robot orientation
        up_world = quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        )
        tipped = up_world[:, 2] < 0.3  # Cosine of ~73 degrees
        
        # Combine termination conditions
        terminated = fell | tipped
        truncated = time_out & ~terminated
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # Reset robot state
        num_resets = len(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-1.0, 1.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        # For orientation, we can sample a random yaw angle and convert to quaternion
        random_yaw = torch.zeros(num_resets, device=self.device).uniform_(0, torch.pi/3) # (N, )
        self._desired_ori_w[env_ids] = quat_from_euler_xyz(torch.zeros_like(random_yaw), torch.zeros_like(random_yaw), random_yaw) # (N, 4)


        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Goal marker (sphere)
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                ori_marker_cfg = FRAME_MARKER_SMALL_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                ori_marker_cfg.prim_path = "/Visuals/Command/goal_orientation"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                self.goal_ori_visualizer = VisualizationMarkers(ori_marker_cfg)
            
            # Robot frame marker (axes)
            if not hasattr(self, "robot_frame_visualizer"):
                marker_cfg = FRAME_MARKER_SMALL_CFG.copy()
                marker_cfg.prim_path = "/Visuals/RobotFrame"
                self.robot_frame_visualizer = VisualizationMarkers(marker_cfg)
            
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ori_visualizer.set_visibility(True)
            self.robot_frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                self.goal_ori_visualizer.set_visibility(False)
            if hasattr(self, "robot_frame_visualizer"):
                self.robot_frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        try:
            robot_pos = self.robot.data.root_pos_w.clone()
        except ReferenceError:
            return
        # Goal sphere
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        # Goal frame (orientation)
        self.goal_ori_visualizer.visualize(self._desired_pos_w, self._desired_ori_w)
        
        # Robot frame at robot position (offset to top of cylinder)
        robot_pos = self.robot.data.root_pos_w.clone()
        robot_pos[:, 2] += 0.35  # Offset to top of cylinder
        self.robot_frame_visualizer.visualize(robot_pos, self.robot.data.root_quat_w)
