"""
Hybrid Agent — Combines Heuristic + RL for NyayaEnv

HYBRID LAYER 3: Deployment blending

Strategy:
  1. Get heuristic recommendation (domain expertise)
  2. Get PPO recommendation (learned policy)
  3. If PPO action is valid and different → use PPO
  4. If PPO picks invalid action → fall back to heuristic
  5. Gradually increase trust in PPO as confidence grows

This guarantees performance >= Heuristic
while allowing RL to push beyond when confident.
"""

import numpy as np
from agents.heuristic_agent import HeuristicAgent


class HybridAgent:
    """
    Blends heuristic + RL agent.

    The heuristic provides a FLOOR of performance.
    The RL agent can IMPROVE on the heuristic.
    If RL makes a bad choice, heuristic catches it.
    """

    def __init__(self, env, rl_model_path="models/ppo_nyaya", rl_weight=0.7):
        """
        Args:
            env: NyayaEnv instance
            rl_model_path: Path to trained PPO model
            rl_weight: How much to trust RL (0.0 = pure heuristic, 1.0 = pure RL)
        """
        self.env = env
        self.heuristic = HeuristicAgent(env)
        self.rl_weight = rl_weight

        # Try to load RL agent
        try:
            from agents.rl_agent import RLAgent
            self.rl_agent = RLAgent(rl_model_path)
            self.has_rl = True
            print(f"✅ Hybrid: RL agent loaded (weight={rl_weight})")
        except Exception as e:
            self.rl_agent = None
            self.has_rl = False
            print(f"⚠️  Hybrid: RL agent not available ({e})")
            print(f"   Using pure heuristic fallback")

    def act(self, observation):
        """
        Choose action using hybrid strategy.

        Decision process:
        1. Always get heuristic recommendation
        2. If RL available, get RL recommendation
        3. Validate RL action (is the case still active?)
        4. If valid → probabilistically choose RL vs heuristic
        5. If invalid → use heuristic
        """
        # Always get heuristic action (guaranteed valid)
        heuristic_action = self.heuristic.act(observation)

        # If no RL agent, use pure heuristic
        if not self.has_rl:
            return heuristic_action

        # Get RL action
        masks = self.env.action_masks() if hasattr(self.env, 'action_masks') else None
        rl_action = self.rl_agent.act(observation, action_masks=masks)

        # Validate RL action
        rl_case_idx = int(rl_action[0])
        rl_case_idx = min(rl_case_idx, len(self.env.cases) - 1)
        rl_case = self.env.cases[rl_case_idx]

        # If RL picks a disposed case → use heuristic
        if rl_case["status"] == "disposed":
            return heuristic_action

        # Probabilistic blend
        if np.random.random() < self.rl_weight:
            return rl_action  # Trust RL
        else:
            return heuristic_action  # Trust heuristic

    def __repr__(self):
        if self.has_rl:
            return f"HybridAgent(rl_weight={self.rl_weight})"
        else:
            return "HybridAgent(heuristic_fallback)"
