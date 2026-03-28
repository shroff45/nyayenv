import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Dict

from nyaya_env import NyayaEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def evaluate_agent(agent, env, num_episodes: int = 50) -> Dict:
    results = {
        "episode_rewards": [],
        "total_disposed": [],
        "total_violations": [],
        "avg_case_age": [],
        "disposal_rate": [],
        "compliance_score": [],
        "evidence_integrity": [],
    }

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        results["episode_rewards"].append(episode_reward)
        results["total_disposed"].append(info["total_disposed"])
        results["total_violations"].append(info["total_violations"])
        results["avg_case_age"].append(info["avg_case_age"])
        results["disposal_rate"].append(info["disposal_rate"])
        results["compliance_score"].append(info["compliance_score"])
        results["evidence_integrity"].append(info["evidence_integrity"])

    return results


def run_benchmark(
    num_episodes: int = 50,
    include_rl: bool = False,
    include_hybrid: bool = False,
    rl_model_path: str = "models/ppo_nyaya",
    num_cases: int = 20,
    num_judges: int = 3,
):
    print("=" * 60)
    print("  NYAYAENV BENCHMARK (HYBRID)")
    print(f"  {num_episodes} episodes | {num_cases} cases | {num_judges} judges")
    print("=" * 60)

    env = NyayaEnv(
        num_cases=num_cases, num_judges=num_judges, max_steps=200
    )

    agents = {
        "Random": RandomAgent(env),
        "Heuristic": HeuristicAgent(env),
    }

    if include_rl:
        try:
            from agents.rl_agent import RLAgent
            agents["RL (Masked PPO)"] = RLAgent(rl_model_path)
        except Exception as e:
            print(f"⚠️  RL agent not available: {e}")

    if include_hybrid:
        try:
            from agents.hybrid_agent import HybridAgent
            agents["Hybrid (0.7 RL)"] = HybridAgent(env, rl_model_path, rl_weight=0.7)
        except Exception as e:
            print(f"⚠️  Hybrid agent not available: {e}")

    all_results = {}
    for name, agent in agents.items():
        print(f"\n📊 Evaluating {name}...")
        results = evaluate_agent(agent, env, num_episodes)
        all_results[name] = results

        print(f"   Avg Reward:      {np.mean(results['episode_rewards']):>8.1f}")
        print(f"   Avg Disposed:    {np.mean(results['total_disposed']):>8.1f} / {num_cases}")
        print(f"   Avg Violations:  {np.mean(results['total_violations']):>8.1f}")
        print(f"   Avg Case Age:    {np.mean(results['avg_case_age']):>8.0f} days")
        print(f"   Compliance:      {np.mean(results['compliance_score']):>8.1%}")

    _plot_comparison(all_results, num_cases)
    _print_summary(all_results)
    return all_results


def _plot_comparison(results: Dict, num_cases: int):
    os.makedirs("outputs", exist_ok=True)
    agents = list(results.keys())
    metrics = {
        "Average Episode Reward": "episode_rewards",
        f"Cases Disposed (/{num_cases})": "total_disposed",
        "BNSS Violations": "total_violations",
        "Avg Case Age (days)": "avg_case_age",
        "Compliance Score": "compliance_score",
        "Evidence Integrity": "evidence_integrity",
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "NyayaEnv HYBRID Agent Comparison\n"
        "Heuristic + RL + Action Masking",
        fontsize=14, fontweight="bold",
    )

    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"]

    for idx, (metric_name, metric_key) in enumerate(metrics.items()):
        ax = axes[idx // 3][idx % 3]
        means = [np.mean(results[a][metric_key]) for a in agents]
        stds = [np.std(results[a][metric_key]) for a in agents]
        bars = ax.bar(
            agents, means, yerr=stds,
            color=colors[:len(agents)],
            capsize=5, edgecolor="black", linewidth=0.5,
        )
        ax.set_title(metric_name, fontsize=10, fontweight="bold")
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(), f"{mean:.1f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig("outputs/agent_comparison.png", dpi=150, bbox_inches="tight")
    print("\n📈 Chart saved to outputs/agent_comparison.png")


def _print_summary(results: Dict):
    agents = list(results.keys())
    width = 16 * len(agents) + 22
    print("\n" + "=" * width)
    print(f"{'METRIC':<22}", end="")
    for a in agents:
        print(f"{a:>16}", end="")
    print("\n" + "-" * width)
    for label, key, fmt in [
        ("Avg Reward", "episode_rewards", ".1f"),
        ("Cases Disposed", "total_disposed", ".1f"),
        ("BNSS Violations", "total_violations", ".1f"),
        ("Avg Case Age", "avg_case_age", ".0f"),
        ("Compliance", "compliance_score", ".1%"),
        ("Evidence", "evidence_integrity", ".1%"),
    ]:
        print(f"{label:<22}", end="")
        for a in agents:
            val = np.mean(results[a][key])
            print(f"{val:>16{fmt}}", end="")
        print()
    print("=" * width)


if __name__ == "__main__":
    run_benchmark(num_episodes=50, include_hybrid=True)
