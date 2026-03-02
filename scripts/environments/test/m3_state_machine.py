import argparse
import contextlib

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for mobile cabinet environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--num_demos", type=int, default=65, help="Number of demonstrations to record. Set to 0 for infinite."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""


import gymnasium as gym
import torch

from isaaclab.sensors import FrameTransformer

import isaaclab_tasks  # noqa: F401
import time
import os 
import omni.ui as ui
#from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg
import isaaclab_mimic.envs # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay
from isaaclab.envs.ui import EmptyWindow
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3_cabinet_camera import rrl_m3_cabinet_camera_env, rrl_m3_cabinet_camera_env_cfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG  # isort: skip
from isaaclab.utils import configclass
# from test_record_demos import setup_output_directories, RateLimiter, ObsActionRecorderTermCfg, setup_ui  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.250, 0.250, 0.250)

RELATIVE_GRIP_POSE = torch.tensor([0.8, 0.0,torch.pi])

FINAL_POSE = torch.tensor([1.2, 0.0, 0.0])

class StateRecorderTerm(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._current_state = None

    def record_pre_step(self):
        return "observation/state", self._current_state.cpu()

    def record_post_step(self):
        return None, None   

class ObservationRecorderTerm(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._current_wrist_obs = None
        self._current_base_obs = None

    def record_pre_step(self):
        return "observation/images", {
        "wrist": self._current_wrist_obs.cpu(),
        "base": self._current_base_obs.cpu(),
    }
    # "actions", self._current_actions.cpu()

    def record_post_step(self):
        return None, None

class ActionRecorderTerm(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._current_actions = None

    def record_pre_step(self):
        return "actions", self._current_actions.cpu()

    def record_post_step(self):
        return None, None
@configclass
class StateRecorderTermCfg(RecorderTermCfg):
    class_type: type = StateRecorderTerm

@configclass
class ObservationRecorderTermCfg(RecorderTermCfg):
    class_type: type = ObservationRecorderTerm

@configclass
class ActionRecorderTermCfg(RecorderTermCfg):
    class_type: type = ActionRecorderTerm

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories(dataset_file: str) -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(dataset_file)
    output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name

def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(xr=False)
    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
        instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display

def process_success_condition(env: gym.Env, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """

    if success_step_count >= args_cli.num_success_steps:
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        print("Success condition met! Recording completed.")
        return 0, True


    return success_step_count, False

def handle_reset(
    env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str
    ) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count

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
        self.R_jacobi = torch.zeros((env.num_envs,6, 6), device=env.device)
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

def reset_env(env: gym.Env):
    scene = env.scene
    robot = env.scene["robot"]
    wrist_camera = env.scene["tiled_camera_wrist"]
    base_camera  = env.scene["tiled_camera_base"]
    num_envs = env.num_envs
    device = env.device
    # Radomize initial robot base pose
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] += 0.4 * (torch.rand(num_envs, device=device) - 0.5) 
    root_state[:, 1] += 1.0 * (torch.rand(num_envs, device=device) - 0.5)  

    root_state[:, :2] += scene.env_origins[:, :2] 

    # Randomize orientation (yaw)
    yaw_noise = 2.0 * (torch.rand(num_envs, device=device) - 0.5)
    zeros = torch.zeros_like(yaw_noise)
    quat_noise = quat_from_euler_xyz(zeros, zeros, yaw_noise)
    root_state[:, 3:7] = quat_mul(quat_noise, root_state[:, 3:7])

    robot.write_root_state_to_sim(root_state)    

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # Randomize joint positions for the arm 
    joint_pos[:, :6] += 0.15 * (torch.rand(num_envs, 6, device=device) - 0.5)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    env.sim.step()
    env.scene.update(dt=env.physics_dt)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[0-5]"], body_names=["link_6"])
    robot_entity_cfg.resolve(scene)

    return env, robot, scene, robot_entity_cfg, wrist_camera, base_camera


def main():
    env_cfg: DirectRLEnvCfg = rrl_m3_cabinet_camera_env_cfg.M3CabinetCameraEnvCfg()
    # env_cfg = parse_env_cfg("RRL-M3-Cabinet-Camera-Direct-v0", device=args_cli.device, num_envs=1)
    env_cfg.debug_vis = False
    env_cfg.write_image_to_file = False

    env_cfg.robot.init_state.pos = (2.0, 0.0, 0.01)

    cabinet_cfg = getattr(env_cfg, "cabinet", getattr(env_cfg.scene, "cabinet", None))
    
    if cabinet_cfg is not None:
        for actuator_name in cabinet_cfg.actuators.keys():
            cabinet_cfg.actuators[actuator_name].stiffness = 0.0      # Kills the spring
            cabinet_cfg.actuators[actuator_name].effort_limit_sim = 0.001 # Limits the max force
    else:
        print("WARNING: Could not find the cabinet in the environment configuration to override actuators.")

    if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs

    # Recording Stuff 
    rate_limiter = RateLimiter(hz=30)
    
    dataset_file_path = f"./datasets/sm_dataset_{args_cli.num_demos}_episodes.hdf5"
    # Set up output directories
    output_dir, output_file_name = setup_output_directories(dataset_file_path)
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

    env_cfg.recorders: RecorderManagerBaseCfg = RecorderManagerBaseCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.obs = ObservationRecorderTermCfg()
    env_cfg.recorders.state = StateRecorderTermCfg()
    env_cfg.recorders.action = ActionRecorderTermCfg()
    
    env = gym.make("RRL-M3-Cabinet-Camera-Direct-v0",
                cfg=env_cfg,
                ).unwrapped
    env.recorder_manager = RecorderManager(env_cfg.recorders, env)
    env.reset()
    instruction_display = setup_ui(label_text, env)

    if hasattr(env_cfg.scene, "cabinet"):
        for actuator_name, actuator_cfg in env_cfg.scene.cabinet.actuators.items():
            actuator_cfg.effort_limit = 0.001

    num_envs = env.num_envs
    device = env.device
    env, robot, scene, robot_entity_cfg, wrist_camera, base_camera = reset_env(env)

    # initialization 
    state_machine = OpenDrawerSM(env=env)
    controller = PositionController(num_envs, device)
    forces = torch.zeros((num_envs, 1, 3), device=device)
    torques = torch.zeros((num_envs, 1, 3), device=device)

    image_save_counter = 0
    
    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running():
            root_state = env.robot.data.root_state_w
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            ee_pose_b, ee_quat_b = subtract_frame_transforms(
                root_state[:, 0:3], root_state[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            full_ee = torch.cat([ee_pose_b, ee_quat_b], dim=1)
            jacobian = robot.root_physx_view.get_jacobians()[:, state_machine.ik.ee_jacobi_idx, :, state_machine.ik.jacobi_joint_ids]
            state_machine.update(root_state, full_ee, robot.data.joint_pos, jacobian)
            wrench, error, velocity = controller.pd(root_state, state_machine.robot_goals)
            forces.zero_()
            torques.zero_()
            # Fx, Fy, Tau_z
            forces[:, 0, 0] = wrench[:, 0]
            forces[:, 0, 1] = wrench[:, 1]
            torques[:, 0, 2] = wrench[:, 2]
            # apply actions
            env.robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces,
                torques= torques,
                body_ids=[0],
            )
            # 6 dof arm command
            robot.data.joint_pos_target[:, :6] = state_machine.joint_goals[:, :6]
            # gripper command
            robot.data.joint_pos_target[:, -2] = 0.0
            is_open_mask = (state_machine.grip[:,0] == 0) 
            robot.data.joint_pos_target[is_open_mask, -2] = 0.04
            
            # recording
            actions = torch.cat([wrench, state_machine.joint_goals, state_machine.grip], dim=1)
            env.recorder_manager._terms["action"]._current_actions = actions
            env.recorder_manager._terms["obs"]._current_wrist_obs = wrist_camera.data.output["rgb"]
            env.recorder_manager._terms["obs"]._current_base_obs = base_camera.data.output["rgb"]
            env.recorder_manager._terms["state"]._current_state = torch.cat([robot.data.joint_pos[:, :-1], robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b], dim=1)
            env.recorder_manager.record_pre_step()
            # step the sim and render
            env.robot.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)

            env.recorder_manager.record_post_step()

            # Check for success condition
            if state_machine.states[0] == 5: # should be 5 
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0  and \
                env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                # reset record manager 
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])

                print("Resetting environment...")
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                instruction_display.show_demo(label_text)

                env, robot, scene, robot_entity_cfg, wrist_camera, base_camera = reset_env(env)
                state_machine = OpenDrawerSM(env=env)
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)
            # TODO: check if the images need to be normalized for lerobot format
            camera_wrist_image = wrist_camera.data.output["rgb"] /255.0   # shape: (num_envs, H, W, 4)  -- RGBA uint8
            camera_base_image  = base_camera.data.output["rgb"] /255.0
            if env_cfg.write_image_to_file:
                save_images_to_file(camera_wrist_image.clone(), f"./images/wrist/wrist_img_{image_save_counter}.png")
                save_images_to_file(camera_base_image.clone(), f"./images/base/base_img_{image_save_counter}.png")
                image_save_counter += 1

    env.close()

if __name__=="__main__":
    main()
    simulation_app.close()