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
class M3EnvCfg(DirectRLEnvCfg):
    """Configuration for the RRL M3 environment."""
    # Environment settings
    decimation: int = 2  # Control frequency = sim_dt * decimation
    episode_length_s: float = 15.0  # 15 seconds per episode
    # 4 thrusters + 6 dof arm + 2 gripper = 16
    action_space: int = 11
    # Observation space: position(3) + orientation(4) + linear_vel(3) + angular_vel(3) = 13 + 6 joint pos + 6 joint vel = 25
    observation_space: int = 13
    # No state space for asymmetric actor-critic
    state_space = 0
    debug_vis = True  
    terminations = None

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,  # 120 Hz simulation
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),  # Normal gravity
        physx=PhysxCfg(
            solver_type=1,  # TGS solver
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
        num_envs=4096, env_spacing=1.5, 
    )

    # robot 
    robot: ArticulationCfg = RRLM3_CFG.replace(prim_path="/World/envs/env_.*/Robot",
                                               init_state=RRLM3_CFG.init_state.replace(
        pos=(1.0, 0.0, 0.01),
        rot = (1.0, 0.0, 0.0, 0.0),
        )
    )  # 180 degrees rotation around Z-axis to face the cabinet) # type: ignore

    # Thruster configuration
    thrusters: ThrusterLayoutCfg = ThrusterLayoutCfg()
    
    # Reward scales
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    # reward_action_penalty: float = -0.001  # Small penalty for using thrusters



    