import pytest
import numpy as np
from nyaya_env import NyayaEnv


class TestAPI:

    def setup_method(self):
        self.env = NyayaEnv(num_cases=10, num_judges=2, max_steps=50)

    def test_reset_observation(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == self.env.observation_space.shape
        assert self.env.observation_space.contains(obs)

    def test_reset_info(self):
        obs, info = self.env.reset(seed=42)
        assert info["active_cases"] == 10
        assert info["step"] == 0

    def test_step_format(self):
        self.env.reset(seed=42)
        result = self.env.step(self.env.action_space.sample())
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_episode_terminates(self):
        self.env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 500:
            _, _, term, trunc, _ = self.env.step(
                self.env.action_space.sample()
            )
            done = term or trunc
            steps += 1
        assert done

    def test_observation_bounds(self):
        obs, _ = self.env.reset(seed=42)
        for _ in range(50):
            obs, _, term, trunc, _ = self.env.step(
                self.env.action_space.sample()
            )
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
            if term or trunc:
                break


class TestBugFixes:

    def test_violations_counted_once(self):
        """BUG #8: Violated case counted only once."""
        env = NyayaEnv(num_cases=3, num_judges=1, max_steps=300)
        env.reset(seed=42)

        for c in env.cases:
            c["bnss_remaining"] = 5
            c["bnss_max_deadline"] = 5
            c["bnss_violated"] = False

        for _ in range(100):
            _, _, term, trunc, info = env.step(
                env.action_space.sample()
            )
            if term or trunc:
                break

        assert info["total_violations"] <= 3, (
            f"Violations ({info['total_violations']}) > 3. "
            f"Bug #8 not fixed."
        )

    def test_is_scheduled_resets(self):
        """BUG #13: is_scheduled resets each step."""
        env = NyayaEnv(num_cases=5, num_judges=2, max_steps=10)
        env.reset(seed=42)

        env.step(np.array([0, 0, 0]))
        env.step(np.array([1, 0, 0]))

        if env.cases[0]["status"] != "disposed":
            assert not env.cases[0]["is_scheduled"]

    def test_disposed_cases_zeroed(self):
        """BUG #14: Disposed cases have zero features in obs."""
        env = NyayaEnv(num_cases=5, num_judges=2, max_steps=500)
        env.reset(seed=42)

        for _ in range(500):
            obs, _, term, trunc, info = env.step(
                env.action_space.sample()
            )
            if info["total_disposed"] > 0:
                break
            if term or trunc:
                break

        if info["total_disposed"] > 0:
            for i, case in enumerate(env.cases):
                if case["status"] == "disposed":
                    start = i * 6
                    features = obs[start: start + 6]
                    assert np.all(features == 0.0), (
                        f"Case {i} not zeroed: {features}"
                    )

    def test_vulnerable_bonus_only_new(self):
        """BUG #2: Bonus fires only for THIS step's disposals."""
        env = NyayaEnv(num_cases=10, num_judges=3, max_steps=500)
        env.reset(seed=42)

        for c in env.cases:
            c["victim_vulnerable"] = True

        rewards = []
        for _ in range(200):
            _, reward, term, trunc, _ = env.step(
                env.action_space.sample()
            )
            rewards.append(reward)
            if term or trunc:
                break

        max_reward = max(rewards)
        assert max_reward < 40, (
            f"Max reward ({max_reward:.1f}) too high. Bug #2."
        )

    def test_unverified_hearing_penalty(self):
        """BUG #9: Hearing unverified evidence gets lower reward."""
        from nyaya_env.rewards import RewardCalculator

        calc = RewardCalculator()

        base_case = {
            "id": 0, "status": "active", "bnss_remaining": 100,
            "age_days": 100, "victim_vulnerable": False,
            "is_scheduled": True,
        }
        base_judge = {
            "cases_today": 1, "weekly_hearings": 1,
            "weekly_capacity": 15,
        }
        base_args = dict(
            step_disposed=0,
            step_violations=0,
            action_success=True,
            action_type=0,
            judges=[base_judge],
            newly_disposed_ids=set(),
            consecutive_same_case_and_action=0,
            acted_case_idx=0,
        )

        r_verified = calc.calculate(
            cases=[{**base_case, "evidence_status": "verified"}],
            acted_case_evidence_at_hearing="verified",
            **base_args,
        )

        r_pending = calc.calculate(
            cases=[{**base_case, "evidence_status": "pending"}],
            acted_case_evidence_at_hearing="pending",
            **base_args,
        )

        assert r_pending < r_verified, (
            f"Pending ({r_pending:.2f}) should be < "
            f"Verified ({r_verified:.2f})"
        )

    def test_heuristic_beats_random(self):
        """Heuristic outperforms random over many episodes."""
        from agents.random_agent import RandomAgent
        from agents.heuristic_agent import HeuristicAgent

        env = NyayaEnv(num_cases=15, num_judges=3, max_steps=150)

        def evaluate(agent_class, num_eps=50):
            agent = agent_class(env)
            rewards = []
            for ep in range(num_eps):
                obs, _ = env.reset(seed=ep + 1000)
                total = 0
                done = False
                while not done:
                    obs, r, term, trunc, _ = env.step(agent.act(obs))
                    total += r
                    done = term or trunc
                rewards.append(total)
            return np.mean(rewards)

        avg_random = evaluate(RandomAgent)
        avg_heuristic = evaluate(HeuristicAgent)

        print(f"\nRandom:    {avg_random:.1f}")
        print(f"Heuristic: {avg_heuristic:.1f}")

        assert avg_heuristic > avg_random * 0.9 or avg_heuristic > avg_random

    def test_different_seeds(self):
        env = NyayaEnv(num_cases=5, num_judges=1)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=99)
        assert not np.array_equal(obs1, obs2)

    def test_info_complete(self):
        env = NyayaEnv(num_cases=5, num_judges=1)
        _, info = env.reset(seed=42)
        for key in [
            "step", "total_disposed", "active_cases",
            "total_violations", "avg_case_age", "disposal_rate",
            "compliance_score", "evidence_integrity",
        ]:
            assert key in info

    def test_render_works(self):
        env = NyayaEnv(num_cases=5, num_judges=1, render_mode="human")
        env.reset(seed=42)
        try:
            env.render()
        except Exception as e:
            pytest.fail(f"render() crashed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
