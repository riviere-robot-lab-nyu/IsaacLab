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
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# USD_PATH = "/home/cpw/workspace/IsaacLab/assets/scenes/Lab_rendering.usd"

USD_PATH = f"{ISAAC_NUCLEUS_DIR}/People/Characters/female_adult_police_02/female_adult_police_02.usd"

def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    
    # Spawn the USD scene
    cfg = sim_utils.UsdFileCfg(usd_path=USD_PATH)
    cfg.func("/World/Scene", cfg)

    
    print(f"\n{'='*60}")
    print(f"Loaded USD: {USD_PATH}")
    print(f"{'='*60}")
    print("\nUse Isaac Sim GUI to inspect the scene.")
    print("Check the Stage window to browse all prims.")
    print("Press Ctrl+C or close window to exit.\n")

    # Reset and play
    sim.reset()

    # Run loop
    while simulation_app.is_running():
        sim.step()
    
    simulation_app.close()


if __name__ == "__main__":
    main()