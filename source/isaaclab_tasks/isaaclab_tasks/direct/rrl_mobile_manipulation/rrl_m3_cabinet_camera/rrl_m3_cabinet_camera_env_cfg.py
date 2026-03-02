from __future__ import annotations
from dataclasses import MISSING, field

import torch
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import CYLINDER_CFG, RRLM3_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers  # isort: skip
from ..thruster_layout_cfg import ThrusterLayoutCfg  # isort: skip

from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.250, 0.250, 0.250)


##
# Environment Configuration
##

@configclass
class M3CabinetCameraEnvCfg(DirectRLEnvCfg):
    """Configuration for the RRL M3 environment."""
    # Environment settings
    decimation: int = 2  # Control frequency = sim_dt * decimation
    episode_length_s: float = 15  # was 8.33 500 timesteps 
    # debugging and testing settings
    debug_vis = True  
    write_image_to_file = False
    debug_env = False

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,  # 120 Hz simulation
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),  # Normal gravity
        physx=PhysxCfg(
            solver_type=1,  # TGS solver
            gpu_max_rigid_patch_count = 24 * 2**15,     # ~786432, well above 376832
            gpu_max_rigid_contact_count = 2**24,        # double the default
            enable_stabilization=True,
            enable_external_forces_every_iteration=True,  # for thusters to work properly
            min_velocity_iteration_count=1,  # stable velocity 
    
        ),
    )
    
    # Ground plane with zero friction
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="max",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
    ) 

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=3.0, 
    )

    # cameras
    tiled_camera_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wxai_follower/camera_link/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(1, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"], # ["rgb", "depth", "semantic_segmentation"]
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=640,
        height=480,
    )

    tiled_camera_base: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wxai_follower/base_link/base_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.025), rot=(1, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"], # ["rgb", "depth", "semantic_segmentation"]
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=640,
        height=480,
    )

    # robot 
    robot: ArticulationCfg = RRLM3_CFG.replace(
    prim_path="/World/envs/env_.*/Robot",
    init_state=RRLM3_CFG.init_state.replace(
        pos=(1.0, 0.0, 0.01),
        rot = (0.0, 0.0, 0.0, 1.0),  # 180 degrees rotation around Z-axis to face the cabinet
        joint_pos={
            "joint_0": 0.0,
            "joint_1": 0.39,
            "joint_2": 0.39,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,

            "left_carriage_joint": 0.04,
        },
    ),
    ) # type: ignore

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # spaces
    # 8 thrusters + 6 dof arm + 2 gripper = 16
    action_space: int = 15
    observation_space = [tiled_camera_wrist.height, tiled_camera_wrist.width, 3*2]
    # No state space for asymmetric actor-critic
    state_space = 0

    # Thruster configuration
    thrusters: ThrusterLayoutCfg = ThrusterLayoutCfg()
    

    action_scale = 1.0
    arm_speed_scale = 0.5
    dof_velocity_scale = 0.5
    # Reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0
    grasp_reward_scale = 5.0




    