import os

# Try to import MaskablePPO (hybrid approach)
# Falls back to regular PPO if sb3-contrib not installed
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    HAS_MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO
    HAS_MASKABLE = False
    print("⚠️  sb3-contrib not installed. Using regular PPO.")
    print("   Install with: pip install sb3-contrib")


def train_agent(
    env,
    total_timesteps: int = 300000,
    save_path: str = "models/ppo_nyaya",
    verbose: int = 1,
):
    """
    Train using HYBRID approach:
    - MaskablePPO prevents invalid actions (Layer 1)
    - Heuristic bonus in env rewards guides exploration (Layer 2)
    - High entropy encourages discovering beyond heuristic
    """

    if HAS_MASKABLE:
        print("✅ Using MaskablePPO (hybrid action masking)")
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            verbose=verbose,
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=128,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.3,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 64], vf=[128, 64])
            ),
            tensorboard_log="./logs/nyaya_hybrid/",
        )
    else:
        print("⚠️  Using regular PPO (no action masking)")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=verbose,
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=128,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.3,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 64], vf=[128, 64])
            ),
            tensorboard_log="./logs/nyaya_ppo/",
        )

    algo_name = "MaskablePPO" if HAS_MASKABLE else "PPO"
    print("=" * 60)
    print(f"  TRAINING {algo_name} AGENT ON NYAYAENV (HYBRID)")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Action masking: {'ON' if HAS_MASKABLE else 'OFF'}")
    print(f"  Heuristic guidance: ON (in reward function)")
    print("=" * 60)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✅ Model saved to {save_path}")

    return model


def load_agent(path: str = "models/ppo_nyaya"):
    if HAS_MASKABLE:
        return MaskablePPO.load(path)
    else:
        from stable_baselines3 import PPO
        return PPO.load(path)


class RLAgent:
    def __init__(self, model_path: str = "models/ppo_nyaya"):
        if HAS_MASKABLE:
            self.model = MaskablePPO.load(model_path)
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        self.is_maskable = HAS_MASKABLE

    def act(self, observation, action_masks=None):
        if self.is_maskable and action_masks is not None:
            action, _ = self.model.predict(
                observation, deterministic=True,
                action_masks=action_masks,
            )
        else:
            action, _ = self.model.predict(
                observation, deterministic=True
            )
        return action

    def __repr__(self):
        algo = "MaskablePPO" if self.is_maskable else "PPO"
        return f"RLAgent({algo})"
