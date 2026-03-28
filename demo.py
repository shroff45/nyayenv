from nyaya_env import NyayaEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def run_demo():
    print()
    print("=" * 60)
    print("  NYAYAENV v2.0 — HYBRID APPROACH")
    print("  Heuristic + RL + Action Masking")
    print("  Part of NyayaSaathiAI")
    print("=" * 60)

    env = NyayaEnv(num_cases=15, num_judges=3, max_steps=100, render_mode="human")
    obs, info = env.reset(seed=42)

    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action mask: {env.action_masks().shape} ({env.action_masks().sum()} valid)")
    print(f"Cases: {info['active_cases']}")
    env.render()

    env_silent = NyayaEnv(num_cases=15, num_judges=3, max_steps=100)

    def evaluate(agent_class, label, steps=100):
        obs, _ = env_silent.reset(seed=42)
        agent = agent_class(env_silent)
        total = 0
        for _ in range(steps):
            obs, r, term, trunc, info = env_silent.step(agent.act(obs))
            total += r
            if term or trunc:
                break
        print(f"\n{label}:")
        print(f"  Reward:      {total:>8.1f}")
        print(f"  Disposed:    {info['total_disposed']:>8d} / 15")
        print(f"  Violations:  {info['total_violations']:>8d}")
        print(f"  Compliance:  {info['compliance_score']:>8.1%}")
        return total

    print("\n\n📊 AGENT COMPARISON")
    print("-" * 40)
    r1 = evaluate(RandomAgent, "Random Agent")
    r2 = evaluate(HeuristicAgent, "Heuristic Agent")

    # Try hybrid
    try:
        from agents.hybrid_agent import HybridAgent
        obs, _ = env_silent.reset(seed=42)
        agent = HybridAgent(env_silent)
        total = 0
        for _ in range(100):
            obs, r, term, trunc, info = env_silent.step(agent.act(obs))
            total += r
            if term or trunc:
                break
        print(f"\nHybrid Agent:")
        print(f"  Reward:      {total:>8.1f}")
        print(f"  Disposed:    {info['total_disposed']:>8d} / 15")
        print(f"  Violations:  {info['total_violations']:>8d}")
        print(f"  Compliance:  {info['compliance_score']:>8.1%}")
    except Exception:
        print("\n⚠️  Train RL agent first for hybrid: python train.py")

    print("\n🚀 NEXT:")
    print("  pip install sb3-contrib")
    print("  python train.py --timesteps 300000 --eval")


if __name__ == "__main__":
    run_demo()
