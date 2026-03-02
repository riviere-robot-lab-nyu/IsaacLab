"""
RRL Microgravity Mobile Manipulator (M3) in lab env 
"""

import gymnasium as gym

# RL config has yet configuerd
from . import agents 

gym.register(
    id="RRL-M3-Cabinet-Camera-Direct-v0",
    entry_point=f"{__name__}.rrl_m3_cabinet_camera_env:M3CabinetCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rrl_m3_cabinet_camera_env:M3CabinetCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RRLM3CabinetPPOCameraRunnerCfg",
    },
)



