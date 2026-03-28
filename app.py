import gradio as gr
import numpy as np
from nyaya_env import NyayaEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def run_episode(agent_type, num_cases, num_judges, max_steps, seed):
    env = NyayaEnv(
        num_cases=int(num_cases),
        num_judges=int(num_judges),
        max_steps=int(max_steps),
    )
    obs, info = env.reset(seed=int(seed))

    if agent_type == "Random":
        agent = RandomAgent(env)
    elif agent_type == "Heuristic (FIFO+Urgency)":
        agent = HeuristicAgent(env)
    elif agent_type == "PPO (Trained)":
        try:
            from agents.rl_agent import RLAgent
            agent = RLAgent("models/ppo_nyaya")
        except Exception:
            agent = HeuristicAgent(env)
            agent_type = "Heuristic (PPO not available)"
    else:
        agent = RandomAgent(env)

    total_reward = 0
    step_log = []
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        step_log.append(
            f"Day {info['step']:>3} | "
            f"Disposed: {info['total_disposed']:>2} | "
            f"Violations: {info['total_violations']:>2} | "
            f"Reward: {reward:>+6.1f} | "
            f"Compliance: {info['compliance_score']:.0%}"
        )

    render_output = env.render()

    summary = f"""
EPISODE COMPLETE — {agent_type}
{'='*45}
Total Steps:      {info['step']:>5}
Total Reward:     {total_reward:>8.1f}
Cases Disposed:   {info['total_disposed']:>5} / {int(num_cases)}
BNSS Violations:  {info['total_violations']:>5}
Compliance:       {info['compliance_score']:>5.0%}
Evidence Score:   {info['evidence_integrity']:>5.0%}
Avg Case Age:     {info['avg_case_age']:>5.0f} days
"""

    log_text = "\n".join(step_log[-20:])
    return summary, render_output, log_text


with gr.Blocks(
    title="NyayaEnv — Judicial RL Environment",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # ⚖️ NyayaEnv — India's First Judicial RL Environment
        ### Part of NyayaSaathiAI

        Simulates an Indian District Court where AI agents learn to:
        - 📋 Schedule cases optimally
        - 👨⚖️ Assign judges by specialization
        - 🔐 Verify evidence integrity (SHA-256 / BSA Section 63)
        - ⏰ Comply with BNSS 2023 deadlines

        **India has 5 crore pending cases. This trains AI to help.**
        """
    )

    with gr.Row():
        with gr.Column():
            agent_type = gr.Dropdown(
                choices=[
                    "Random",
                    "Heuristic (FIFO+Urgency)",
                    "PPO (Trained)",
                ],
                value="Heuristic (FIFO+Urgency)",
                label="Agent Type",
            )
            num_cases = gr.Slider(5, 50, value=20, step=5, label="Cases")
            num_judges = gr.Slider(1, 10, value=3, step=1, label="Judges")
            max_steps = gr.Slider(50, 500, value=200, step=50, label="Max Days")
            seed = gr.Slider(0, 100, value=42, step=1, label="Seed")
            run_btn = gr.Button("▶️ Run Episode", variant="primary")

        with gr.Column():
            summary_output = gr.Textbox(label="📊 Summary", lines=12)
            render_output = gr.Textbox(label="🏛️ Court State", lines=15)

    log_output = gr.Textbox(label="📜 Step Log (last 20)", lines=10)

    run_btn.click(
        fn=run_episode,
        inputs=[agent_type, num_cases, num_judges, max_steps, seed],
        outputs=[summary_output, render_output, log_output],
    )

if __name__ == "__main__":
    demo.launch()
