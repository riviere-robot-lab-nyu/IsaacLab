"""Standalone script to view USD scene - NO robot, NO gym env."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View USD scene standalone.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch with GUI
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ============ Imports after AppLauncher ============
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG 
from isaaclab_assets.robots.humanoid import HUMANOID_CFG
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG




USD_PATH = "/home/cpw/workspace/IsaacLab/assets/M3_Robot_v1000/M3_Robot_v1000.usd"

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=12.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


@configclass
class MyEnvSceneCfg(InteractiveSceneCfg):
    """Scene config with multiple parallel environments."""

    # Simple ground plane
    ground: TerrainImporterCfg = TerrainImporterCfg(
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
    
    # Light
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot") # FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")


def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cuda:0")
    sim = SimulationContext(sim_cfg)
        
    # Create scene config
    scene_cfg = MyEnvSceneCfg(
        num_envs=5, env_spacing=5.0
    )
    # Create the scene (this handles cloning!)
    scene = InteractiveScene(scene_cfg)
    
    print(f"\n{'='*60}")
    print(f"Loaded USD: {USD_PATH}")
    print(f"{'='*60}")
    print("\nUse Isaac Sim GUI to inspect the scene.")
    print("Check the Stage window to browse all prims.")
    print("Press Ctrl+C or close window to exit.\n")

    # Reset and play
    sim.reset()
    scene.reset()
        
    # Run loop
    while simulation_app.is_running():
        sim.step()
    
    simulation_app.close()


if __name__ == "__main__":
    main()