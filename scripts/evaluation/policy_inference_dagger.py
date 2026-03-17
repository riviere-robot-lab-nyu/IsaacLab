"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, matrix_from_quat, quat_from_euler_xyz, quat_mul, quat_apply_inverse

import leisaac  # noqa: F401

# TODO: write state machine in utils or something
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
        # desired force in body frame
        forces_body = quat_apply_inverse(quat, torch.cat([force[:,:2],torch.zeros_like(force[:,0:1])], dim=1))
        forces_body[:, -1] = force[:, -1]
        # actions 
        thrust = forces_body@self.D_inv_T
        thrust = torch.clip(thrust, -1., 1.)
        force_body_real = thrust@self.D_T
        return force_body_real, error, velocity

RELATIVE_GRIP_POSE = torch.tensor([0.8, 0.0,torch.pi])
class arm_ik():
    def __init__(self, env):
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)
        scene = env.scene
        robot = env.scene["robot"]
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[0-5]"], body_names=["link_6"])
        robot_entity_cfg.resolve(scene)
        self.ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        self.jacobi_joint_ids = [idx + 6 for idx in robot_entity_cfg.joint_ids]
        self.lower = robot.data.joint_limits[0, robot_entity_cfg.joint_ids, 0]
        self.upper = robot.data.joint_limits[0, robot_entity_cfg.joint_ids, 1]
        root_pose_w_init = robot.data.root_pose_w
        R = matrix_from_quat(root_pose_w_init[:, 3:7])
        self.R_jacobi = torch.zeros((env.num_envs, 6, 6), device=env.device)
        self.R_jacobi[:, :3, :3] = R
        self.R_jacobi[:, 3:, 3:] = R
    
    def set_command(self, ik_commands):
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(ik_commands)
    
    def compute(self, ee_pose_body, ee_quat_body, jacobian, joint_pos):
        jacobian = self.R_jacobi @ jacobian
        return self.diff_ik_controller.compute(ee_pose_body, ee_quat_body, jacobian, joint_pos)

class OpenDrawerSM:
    def __init__(self, env, randomize=True):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.states = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.robot_goals = torch.zeros((self.num_envs, 3), device=self.device)
        self.joint_goals = torch.zeros((self.num_envs, 6), device=self.device)
        self.grip = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int)
        self.ik = arm_ik(env=env)
        drawer_index = env.cabinet.find_bodies("drawer_handle_top")[0][0]
        self.handle_pose = env.cabinet.data.body_pose_w[:, drawer_index]
        self.drawer_pos_b = torch.zeros((self.num_envs, 3),device=self.device)
        self.randomize=randomize
        self._init_goals()
        
    
    def _init_goals(self):
        self.robot_goals[:, :2] = self.env.scene.env_origins[:, :2] +  RELATIVE_GRIP_POSE[:2].to(device=self.device)
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
        # wrap yaw error to [-pi, pi]
        error[:,2] = (error[:,2] + torch.pi) % (2*torch.pi) - torch.pi
        velocity = torch.cat([robot_base_pose[:,7:9], robot_base_pose[:, -1].unsqueeze(1)],dim=1)

        body_err_norm = torch.linalg.norm(error, dim=1) + torch.linalg.norm(velocity, dim=1)
        
        mask_0 = (self.states==0)
        rand_thresh = torch.zeros(self.num_envs, device=self.device)
        if self.randomize:
            rand_thresh = 0.3*(torch.rand(self.num_envs, device=self.device) + 0.5)
        transition_0_1 = mask_0 & (torch.linalg.norm(p[:,:2]- self.env.scene.env_origins[:, :2],dim=1) + rand_thresh < 1.2)
        mask_1 = (self.states==1)
        transition_1_2 = mask_1 & (body_err_norm + torch.linalg.norm(self.drawer_pos_b[:,:3] - ee_pose[:,0:3],dim=1) < 0.04)
        mask_2 = (self.states==2)
        transition_2_3 = mask_2 & (joint_pos[:, -2] < 0.01)
        mask_3 = (self.states==3)
        transition_3_4 = mask_3 & (body_err_norm < 0.02+0.4)
        mask_4 = (self.states==4)
        transition_4_5 = mask_4 & (joint_pos[:, -2] > 0.035)
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
    
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.states[env_ids] = 0
        self.grip[env_ids] = 0
        self.robot_goals[env_ids] = 0
        self.joint_goals[env_ids] = 0
        self._init_goals()  # recalculate goals from fresh env state

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


class KeyboardController:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False
        self.human_intervantion = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False
        self.human_intervantion = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
            if event.input.name == "I":
                self.human_intervantion = True
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

def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

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
    env_cfg.recorders = None
    #######################


    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing
    env_cfg.decimation = 1
    env_cfg.debug_env = False
    env_cfg.debug_vis = False
    env_cfg.replay_episode = True

    env_cfg.robot.init_state.pos = (2.0, 0.0, 0.01)

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
    keyboard_controller = KeyboardController()

    # reset environment
    env.reset()
    keyboard_controller.reset()

    robot = env.robot
    wrist_camera = env.scene["tiled_camera_wrist"]
    base_camera  = env.scene["tiled_camera_base"]

    ############################################
    # Randomize the initial pose here
    ############################################ 
    # Radomize initial robot base pose
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] += 0.4 * (torch.rand(1, device=env.device) - 0.5) 
    root_state[:, 1] += 1.0 * (torch.rand(1, device=env.device) - 0.5)  

    root_state[:, :2] += env.scene.env_origins[:, :2] 

    # Randomize orientation (yaw)
    yaw_noise = 1.0 * (torch.rand(1, device=env.device) - 0.5)
    zeros = torch.zeros_like(yaw_noise)
    quat_noise = quat_from_euler_xyz(zeros, zeros, yaw_noise)
    root_state[:, 3:7] = quat_mul(quat_noise, root_state[:, 3:7])

    robot.write_root_state_to_sim(root_state)    

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # Randomize joint positions for the arm 
    joint_pos[:, :6] += 0.15 * (torch.rand(1, 6, device=env.device) - 0.5) 

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
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
    # state_machine = OpenDrawerSM(env=env)
    controller = PositionController(1, env.device)
    forces = torch.zeros((1, 1, 3), device=env.device)
    torques = torch.zeros((1, 1, 3), device=env.device)
    is_sm_reset = False
    use_action_expert = False
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[0-5]"], body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if keyboard_controller.reset_state:
                    keyboard_controller.reset()
                    env.reset()
                    # obs_dict, _ = env.reset()
                    episode_count += 1
                    break
                elif keyboard_controller.human_intervantion:
                    use_action_expert = True
                    keyboard_controller.reset()

                cabinet = env.scene["cabinet"]
                
                # Build the observation dictionary 
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

                start = time.perf_counter()
                actions = policy.get_action(obs_dict["policy"]).to(env.device)
                print(f"[Evaluation] Policy inference time: {time.perf_counter() - start:.2f} seconds")

                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    if use_action_expert:
                        if not is_sm_reset:
                            state_machine = OpenDrawerSM(env=env)
                            is_sm_reset = True
                        root_state = env.robot.data.root_state_w
                        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                        ee_pose_b, ee_quat_b = subtract_frame_transforms(
                            root_state[:, 0:3], root_state[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                        )
                        full_ee = torch.cat([ee_pose_b, ee_quat_b], dim=1)
                        jacobian = robot.root_physx_view.get_jacobians()[:, state_machine.ik.ee_jacobi_idx, :, state_machine.ik.jacobi_joint_ids]
                        state_machine.update(root_state, full_ee, robot.data.joint_pos, jacobian)
                        wrench, error, velocity = controller.pd(root_state, state_machine.robot_goals)
                        # Fx, Fy, Tau_z
                        forces[:, 0, 0] = wrench[:, 0]
                        forces[:, 0, 1] = wrench[:, 1]
                        torques[:, 0, 2] = wrench[:, 2]
                        # 6 dof arm command
                        robot.data.joint_pos_target[:, :6] = state_machine.joint_goals[:, :6]
                        # gripper command
                        robot.data.joint_pos_target[:, -2] = 0.0
                        is_open_mask = (state_machine.grip[:,0] == 0) 
                        robot.data.joint_pos_target[is_open_mask, -2] = 0.04
                    else:
                        action = actions[i, :, :]
                        action[:, -1] = convert_gripper_command_to_env_format(action[:, -1]) # [0, 0.044]
                        # Fx, Fy, Tau_z
                        forces[:, 0, 0] = action[:, 0]
                        forces[:, 0, 1] = action[:, 1]
                        torques[:, 0, 2] = action[:, 2] 
                        # 6 dof arm command
                        env.robot.data.joint_pos_target[:, :7] = action[:, 3:]
                    
                    # apply actions
                    env.robot.instantaneous_wrench_composer.set_forces_and_torques(
                        forces=forces,
                        torques= torques,
                        body_ids=[0],
                    )

                    # env step and render
                    env.robot.write_data_to_sim()
                    env.sim.step()
                    env.scene.update(dt=env.physics_dt)
                    # print(f"[Evaluation] Episode {episode_count} Step {frame}, Action: {action.cpu().numpy()}")

                    # obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    # step the sim and render
                    if cabinet.data.joint_pos[0, 1] > 0.35: # check if the drawer is opened, this threshold can be adjusted based on the actual drawer configuration
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
