import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for mobile cabinet environments.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.sensors import FrameTransformer

import isaaclab_tasks  # noqa: F401
#from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet import rrl_m3_cabinet_env, rrl_m3_cabinet_env_cfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul
from isaaclab.markers import VisualizationMarkers

from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.250, 0.250, 0.250)

RELATIVE_GRIP_POSE = torch.tensor([0.8, 0.0,torch.pi])

FINAL_POSE = torch.tensor([1.2, 0.0, 0.0])

class PositionController:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
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
    
    def pd(self, state, goal):
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
        forces_body = quat_apply_inverse(quat, torch.cat([force[:,:2],torch.zeros_like(force[:,0:1])], dim=1))
        forces_body[:, -1] = force[:, -1]
        thrust = forces_body@self.D_inv_T
        thrust = torch.clip(thrust, -1., 1.)
        force_body_real = thrust@self.D_T
        return force_body_real, error, velocity
    
class arm_ik():
    def __init__(self, env):
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device)
        scene = env.unwrapped.scene
        robot = env.unwrapped.scene["robot"]
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[0-5]"], body_names=["link_6"])
        robot_entity_cfg.resolve(scene)
        self.ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        self.jacobi_joint_ids = [idx + 6 for idx in robot_entity_cfg.joint_ids]
        self.lower = robot.data.joint_limits[0, robot_entity_cfg.joint_ids, 0]
        self.upper = robot.data.joint_limits[0, robot_entity_cfg.joint_ids, 1]
        root_pose_w_init = robot.data.root_pose_w
        R = matrix_from_quat(root_pose_w_init[:, 3:7])
        self.R_jacobi = torch.zeros((env.unwrapped.num_envs,6, 6), device=env.unwrapped.device)
        self.R_jacobi[:, :3, :3] = R
        self.R_jacobi[:, 3:, 3:] = R
    
    def set_command(self, ik_commands):
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(ik_commands)
    
    def compute(self, ee_pose_body, ee_quat_body, jacobian, joint_pos):
        jacobian = self.R_jacobi @ jacobian
        return self.diff_ik_controller.compute(ee_pose_body, ee_quat_body, jacobian, joint_pos)
## STATES : [ 0, 1, 2, 3, 4]
## 0 := Coarse approach (body only)
## 1 := Final approach and arm movement
## 2 := Get a grip
## 3 := pull out
## 4 := release grip
## 5 := back it up
class OpenDrawerSM:
    def __init__(self, env, randomize=True):
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.states = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.robot_goals = torch.zeros((self.num_envs, 3), device=self.device)
        self.joint_goals = torch.zeros((self.num_envs, 6), device=self.device)
        self.grip = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int)
        self.ik = arm_ik(env=env)
        drawer_index = env.unwrapped.cabinet.find_bodies("drawer_handle_top")[0][0]
        self.handle_pose = env.unwrapped.cabinet.data.body_pose_w[:, drawer_index]
        self.drawer_pos_b = torch.zeros((self.num_envs, 3),device=self.device)
        self.randomize=randomize
        self._init_goals()
        
    
    def _init_goals(self):
        self.robot_goals[:, :2] = self.env.unwrapped.scene.env_origins[:, :2] +  RELATIVE_GRIP_POSE[:2].to(device=self.device)
        self.robot_goals[:, -1] = RELATIVE_GRIP_POSE[-1].to(device=self.device).repeat(self.num_envs)
        goal_pos = torch.zeros((self.num_envs, 3),device=self.device)
        goal_pos[:, :2] = self.robot_goals[:, :2]
        goal_quats = torch.zeros((self.num_envs, 4), device=self.device)
        goal_quats[:, -1] =1.0
        self.drawer_pos_b, _ = subtract_frame_transforms(
            goal_pos, goal_quats, self.handle_pose[:, :3], self.handle_pose[:, 3:7]
        )
        self.drawer_pos_b[:, 0] -= 0.305
        self.drawer_pos_b[:, 2] += 0.02
        if self.randomize:
            self.drawer_pos_b[:, 1] = 2*0.05*(torch.rand(self.num_envs, device=self.device) - 0.5)
        self.drawer_pos_b[:, 0] -= 0.25376 - 0.1 -0.02
        ik_commands = torch.zeros((self.num_envs, 7), device=self.device)
        ik_commands[:, :3] = self.drawer_pos_b
        ik_commands[:, 3:5] = 0.7071
        self.ik.set_command(ik_commands)
        
        
    
    def update(self, robot_base_pose, ee_pose, joint_pos, jacobians):
        quat = robot_base_pose[:, 3:7]
     
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        p = torch.cat([robot_base_pose[:, :2], yaw.unsqueeze(1)], dim=1)
        error = self.robot_goals - p
        error[:,2] = (error[:,2] + torch.pi) % (2*torch.pi) - torch.pi
        velocity = torch.cat([robot_base_pose[:,7:9], robot_base_pose[:, -1].unsqueeze(1)],dim=1)
        b_norm = torch.linalg.norm(error, dim=1)
        body_err_norm =  b_norm + torch.linalg.norm(velocity, dim=1)
        
        mask_0 = (self.states==0)
        rand_thresh = torch.zeros(self.num_envs, device=self.device)
        if self.randomize:
            rand_thresh = 0.3*(torch.rand(self.num_envs, device=self.device) + 0.5)
        transition_0_1 = mask_0 & (torch.linalg.norm(p[:,:2]- self.env.unwrapped.scene.env_origins[:, :2],dim=1) + rand_thresh < 1.2)
        mask_1 = (self.states==1)
        transition_1_2 = mask_1 & (body_err_norm + torch.linalg.norm(self.drawer_pos_b[:,:3] - ee_pose[:,0:3],dim=1) < 0.04)
        mask_2 = (self.states==2)
        transition_2_3 = mask_2 & (joint_pos[:, -2] < 0.01)
        mask_3 = (self.states==3)
        transition_3_4 = mask_3 & (body_err_norm < 0.02+0.4)
        mask_4 = (self.states==4)
        transition_4_5 = mask_4 & (joint_pos[:, -2]>0.035)
        if transition_0_1.any():
            self.states[transition_0_1] = 1
        elif transition_1_2.any():
            self.states[transition_1_2] = 2
            self.grip[transition_1_2] = 1
        elif transition_2_3.any():
            self.states[transition_2_3] = 3
            self.robot_goals[transition_2_3, 0] += 0.8
        elif transition_3_4.any():
            self.states[transition_3_4] = 4
            self.grip[transition_3_4] = 0
            self.robot_goals[transition_3_4] += 0.03
        elif transition_4_5.any():
            self.states[transition_4_5] = 5
            self.joint_goals[transition_4_5, :6] = 0.0


        ik_states = (self.states > 0) & (self.states < 5)
        if ik_states.any():
            desired_joint_pos = self.ik.compute(ee_pose[:, 0:3], ee_pose[:, 3:7], jacobians, joint_pos[:,:6])
            n_err = torch.linalg.norm(desired_joint_pos - joint_pos[:,:6], dim=1)
            n_err_mask = (n_err > 0.05)
            if n_err_mask.any():
                damping_factor = (1/(1 + (15*n_err[n_err_mask]))).unsqueeze(1)
                desired_joint_pos[n_err_mask] = joint_pos[n_err_mask, :6] + damping_factor * (desired_joint_pos[n_err_mask] - joint_pos[n_err_mask, :6])
            self.joint_goals[ik_states, :6] = desired_joint_pos[ik_states]
    
def main():
    env_cfg = rrl_m3_cabinet_env_cfg.M3CabinetEnvCfg()
    env_cfg.debug_vis = False

    cabinet_cfg = getattr(env_cfg, "cabinet", getattr(env_cfg.scene, "cabinet", None))
    
    if cabinet_cfg is not None:
        for actuator_name in cabinet_cfg.actuators.keys():
            cabinet_cfg.actuators[actuator_name].stiffness = 0.0      # Kills the spring
            cabinet_cfg.actuators[actuator_name].effort_limit_sim = 0.001 # Limits the max force
    else:
        print("WARNING: Could not find the cabinet in the environment configuration to override actuators.")

    if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("RRL-M3-Cabinet-Direct-v0",
                cfg=env_cfg,
                )
    env.reset()

    if hasattr(env_cfg.scene, "cabinet"):
        for actuator_name, actuator_cfg in env_cfg.scene.cabinet.actuators.items():
            actuator_cfg.effort_limit = 0.001

    scene = env.unwrapped.scene
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] += 0.4 * (torch.rand(num_envs, device=device) - 0.5) 
    root_state[:, 1] += 1.0 * (torch.rand(num_envs, device=device) - 0.5)  

    root_state[:, :2] += scene.env_origins[:, :2] 

    yaw_noise = 2.0 * (torch.rand(num_envs, device=device) - 0.5)
    zeros = torch.zeros_like(yaw_noise)
    quat_noise = quat_from_euler_xyz(zeros, zeros, yaw_noise)
    root_state[:, 3:7] = quat_mul(quat_noise, root_state[:, 3:7])

    robot.write_root_state_to_sim(root_state)    

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    joint_pos[:, :6] += 0.15 * (torch.rand(num_envs, 6, device=device) - 0.5)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    env.unwrapped.sim.step()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

    state_machine = OpenDrawerSM(env=env)
    pd = PositionController(num_envs, device)
    forces = torch.zeros((num_envs, 1, 3), device=device)
    torques = torch.zeros((num_envs, 1, 3), device=device)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[0-5]"], body_names=["link_6"])
    robot_entity_cfg.resolve(scene)
    while simulation_app.is_running():
        root_state = env.unwrapped.robot.data.root_state_w
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pose_b, ee_quat_b = subtract_frame_transforms(
            root_state[:, 0:3], root_state[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        full_ee = torch.cat([ee_pose_b, ee_quat_b], dim=1)
        jacobian = robot.root_physx_view.get_jacobians()[:, state_machine.ik.ee_jacobi_idx, :, state_machine.ik.jacobi_joint_ids]
        state_machine.update(root_state, full_ee, robot.data.joint_pos, jacobian)
        wrench, error, velocity = pd.pd(root_state, state_machine.robot_goals)
        forces.zero_()
        torques.zero_()
        forces[:, 0, 0] = wrench[:, 0]
        forces[:, 0, 1] = wrench[:, 1]
        torques[:, 0, 2] = wrench[:, 2]
        env.unwrapped.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques= torques,
            body_ids=[0],
        )
        robot.data.joint_pos_target[:, :6] = state_machine.joint_goals[:, :6]
        robot.data.joint_pos_target[:, -2]=0.0
        is_open_mask = (state_machine.grip[:,0] == 0) 
        robot.data.joint_pos_target[is_open_mask, -2] = 0.04
        
        env.unwrapped.robot.write_data_to_sim()
        env.unwrapped.sim.step()
        env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

    env.close()

if __name__=="__main__":
    main()
    simulation_app.close()
