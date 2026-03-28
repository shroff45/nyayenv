import argparse
from nyaya_env import NyayaEnv
from agents.rl_agent import train_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    env = NyayaEnv(
        num_cases=args.cases,
        num_judges=args.judges,
        max_steps=200,
    )

    print(f"\nEnvironment: {args.cases} cases, {args.judges} judges")
    print(f"Observation: {env.observation_space.shape}")
    print(f"Actions: {env.action_space}")
    print(f"Action mask size: {len(env.action_masks())}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Approach: HYBRID (masking + heuristic guidance)\n")

    model = train_agent(env, total_timesteps=args.timesteps)

    if args.eval:
        from evaluation.benchmark import run_benchmark
        run_benchmark(
            num_episodes=args.eval_episodes,
            include_rl=True,
            include_hybrid=True,
            rl_model_path="models/ppo_nyaya",
            num_cases=args.cases,
            num_judges=args.judges,
        )


if __name__ == "__main__":
    main()
