
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg



raise NotImplementedError("This file is not yet implemented.")
@configclass
class RRLM3VelCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24          # was 16 — longer rollouts capture thruster→move→reach→grasp sequences
    max_iterations = 3000           # was 1500 — much harder task needs more time
    save_interval = 50             # save often so you can find the sweet spot
    experiment_name = "rrl_m3_vel_camera_cabinet_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,         # was 1.0 — thrusters map [-1,1]→[0,max], so 1.0 is too aggressive
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],  # bigger network for harder problem
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,          # was 0.0 — critical, prevents collapse
        num_learning_epochs=5,      # was 8 — less aggressive updates for stability
        num_mini_batches=4,         # was 8 — larger batches for less noisy gradients
        learning_rate=3.0e-4,       # was 5e-4 — slightly lower for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,           # was 0.008 — looser KL allows more exploration
        max_grad_norm=1.0,
    )