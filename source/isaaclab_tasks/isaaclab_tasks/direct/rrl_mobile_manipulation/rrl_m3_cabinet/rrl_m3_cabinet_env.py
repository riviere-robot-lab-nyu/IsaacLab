"""Thruster Cylinder Environment for Isaac Lab."""

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms, sample_uniform
from isaaclab.sim.utils.stage import get_current_stage
##
# Pre-defined configs
##
from isaaclab.markers import CUBOID_MARKER_CFG, POSITION_GOAL_MARKER_CFG, SPHERE_MARKER_CFG, FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.250, 0.250, 0.250)
FRAME_MARKER_TINY_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_TINY_CFG.markers["frame"].scale = (0.100, 0.100, 0.100)


from .rrl_m3_cabinet_env_cfg import M3CabinetEnvCfg  # isort: skip


class M3CabinetEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: M3CabinetEnvCfg
    
    def __init__(self, cfg: M3CabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # Initialization
        self._setup_thrusters()
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        
        # Arm joint position targets
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        
        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/wxai_follower/link_6")),
            device=self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/wxai_follower/gripper_left")),
            device=self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/wxai_follower/gripper_right")),
            device=self.device,
        )   


        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0.04, 0, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        # Gripper's approach direction to drawer
        self.gripper_forward_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self.robot.find_bodies("link_6")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("gripper_left")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("gripper_right")[0][0]
        self.drawer_link_idx = self.cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
    

    def _setup_thrusters(self):
        thruster_names = self.cfg.thrusters.thruster_names
        
        # Positions: (8, 3)
        positions = []
        directions = []
        for name in thruster_names:
            positions.append(self.cfg.thrusters.positions[name])
            directions.append(self.cfg.thrusters.directions[name])
        
        self._thruster_positions = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        )  # (8, 3)
        
        self._thruster_directions = torch.tensor(
            directions, dtype=torch.float32, device=self.device
        )  # (8, 3)
        
        self._max_thrust = self.cfg.thrusters.max_thrust
        self._num_thrusters = len(thruster_names)

    
    def _setup_scene(self):
        """Set up the scene with robot and ground."""
        # Add robot to scene
        self.robot = Articulation(self.cfg.robot)
        self.cabinet = Articulation(self.cfg.cabinet)

        # add articulation and sensors to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["cabinet"] = self.cabinet
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
        
        # Map actions from [-1, 1] to [0, max_thrust]
        thrust_magnitudes = (self._actions[:, :8].clamp(-1.0, 1.0) + 1.0) * 0.5 * self._max_thrust  # (num_envs, 8)
        
        # Compute forces in body frame
        forces_body = thrust_magnitudes.unsqueeze(-1) * self._thruster_directions.unsqueeze(0) # (num_envs, 8, 3)
        
        # Sum all thruster forces to get total force
        total_force_body = forces_body.sum(dim=1)  # (num_envs, 3)
        
        # Compute torques: torque = position × force
        torques_body = torch.cross(
            self._thruster_positions.unsqueeze(0).expand(self.num_envs, -1, -1),
            forces_body,
            dim=-1
        )
        total_torque_body = torques_body.sum(dim=1)  # (num_envs, 3)
                
        # Apply external forces and torques
        self._applied_forces = total_force_body
        self._applied_torques = total_torque_body


        arm_targets = (
            self.robot_dof_targets[:, :-2]  # previous targets for arm joints
            + self.cfg.arm_speed_scale * self.dt * self._actions[:, 8:-1] * self.cfg.action_scale
        )

        # Clamp the joint targets to be within limits
        self.robot_dof_targets[:, :-2] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[:-2],
            self.robot_dof_upper_limits[:-2],
        )

        # Binary Gripper command
        close_mask = (actions[:, -1] < 0.0).unsqueeze(-1)  # (num_envs, 1)
        gripper_targets = torch.where(close_mask, 0, 0.04) # 0.04 is the open position, 0 is the closed position
        self.robot_dof_targets[:, -2] = gripper_targets.squeeze(-1)  
        # 8 joint targets for the arm, but ignore the last gripper joint (mimic joint)
        # self.robot_dof_targets[:, :-1] = self._actions[:, 8:]

    
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
      
        dof_pos_scaled = (
            2.0
            * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        root_pos_local = self.robot.data.root_pos_w - self.scene.env_origins  # (num_envs, 3)
        root_quat = self.robot.data.root_quat_w                               # (num_envs, 4)

        obs = torch.cat(
            (   root_pos_local,
                root_quat,
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                dof_pos_scaled,
                self.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self.cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                self.cabinet.data.joint_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        if self.cfg.debug_env:
            return {"policy": torch.clamp(obs, -5.0, 5.0), "debug_to_target_dist": to_target, "debug_cabinet_joint_pos": self.cabinet.data.joint_pos[:, 3].unsqueeze(-1)}
        else:
            return  {"policy": torch.clamp(obs, -5.0, 5.0)}
    
    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self.cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self.robot.data.joint_pos,
        )
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        cabinet_done = self.cabinet.data.joint_pos[:, 3] > 0.39
        
        # Robot tipped over (check if up vector is pointing down)
        # Get the up vector in world frame by rotating [0, 0, 1] by robot orientation
        up_world = quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        )
        tipped = up_world[:, 2] < 0.7  # Cosine of ~80 degrees, was 0.3

        # Robot drifted too far from cabinet
        root_pos_local = self.robot.data.root_pos_w - self.cabinet.data.root_pos_w
        drifted = torch.norm(root_pos_local[:, :2], dim=-1) > 1.5  # 1.5m from cabinet
        
        # Combine termination conditions
        terminated = tipped | cabinet_done | drifted
        truncated = time_out & ~terminated
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self.cabinet.num_joints), device=self.device)
        self.cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

         # cabinet state
        zeros = torch.zeros((len(env_ids), self.cabinet.num_joints), device=self.device)
        self.cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Robot frame marker (axes)
            if not hasattr(self, "robot_frame_visualizer"):
                marker_cfg = FRAME_MARKER_SMALL_CFG.copy()
                drawer_marker_cfg = FRAME_MARKER_TINY_CFG.copy()
                ee_grasp_marker = FRAME_MARKER_TINY_CFG.copy()
                marker_cfg.prim_path = "/Visuals/RobotFrame"
                drawer_marker_cfg.prim_path = "/Visuals/DrawerHandleFrame"
                ee_grasp_marker.prim_path = "/Visuals/EEGraspFrame"
                self.robot_frame_visualizer = VisualizationMarkers(marker_cfg)
                self.drawer_goal_frame_visualizer = VisualizationMarkers(drawer_marker_cfg)
                self.ee_grasp_frame_visualizer = VisualizationMarkers(ee_grasp_marker)
            
            self.robot_frame_visualizer.set_visibility(True)
            self.drawer_goal_frame_visualizer.set_visibility(True)
            self.ee_grasp_frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "robot_frame_visualizer"):
                self.robot_frame_visualizer.set_visibility(False)
            if hasattr(self, "drawer_goal_frame_visualizer"):
                self.drawer_goal_frame_visualizer.set_visibility(False)
            if hasattr(self, "ee_grasp_frame_visualizer"):
                self.ee_grasp_frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):        
        # Robot frame at robot position (offset to top of cylinder)
        robot_pos = self.robot.data.root_pos_w.clone()
        robot_pos[:, 2] += 0.35  # Offset to top of cylinder
        self.robot_frame_visualizer.visualize(robot_pos, self.robot.data.root_quat_w)

        # TCP = midpoint of fingers (live)
        # lfinger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        # rfinger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]
        # tcp_pos = (lfinger_pos + rfinger_pos) / 2.0
        # tcp_rot = self.robot.data.body_quat_w[:, self.left_finger_link_idx]  # use either finger's rot
        self.ee_grasp_frame_visualizer.visualize(self.robot_grasp_pos, self.robot_grasp_rot)
        
        # Drawer handle frame
        self.drawer_goal_frame_visualizer.visualize(self.drawer_grasp_pos, self.drawer_grasp_rot)



    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        hand_pos = self.robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self.robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self.cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self.cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
