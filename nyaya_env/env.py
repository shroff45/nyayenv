import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple

from nyaya_env.case_generator import CaseGenerator
from nyaya_env.bnss_compliance import BNSSComplianceEngine
from nyaya_env.evidence_engine import EvidenceIntegrityEngine
from nyaya_env.rewards import RewardCalculator


class NyayaEnv(gym.Env):

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_cases: int = 20,
        num_judges: int = 3,
        num_courtrooms: int = 2,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_cases = num_cases
        self.num_judges = num_judges
        self.num_courtrooms = num_courtrooms
        self.max_steps = max_steps
        self.render_mode = render_mode

        obs_size = (num_cases * 6) + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete(
            [num_cases, num_judges, 3]
        )

        self.case_generator = CaseGenerator()
        self.bnss_engine = BNSSComplianceEngine()
        self.evidence_engine = EvidenceIntegrityEngine()
        self.reward_calculator = RewardCalculator()

        self.cases = []
        self.judges = []
        self.current_step = 0
        self.total_disposed = 0
        self.total_violations = 0
        self.episode_rewards = []

        self._newly_disposed_this_step: set = set()
        self._last_acted_case_id = -1
        self._last_acted_action_type = -1
        self._consecutive_same = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.cases = self.case_generator.generate_cases(
            self.num_cases, self.np_random
        )
        self.judges = self._initialize_judges()
        self.evidence_engine.reset()

        self.current_step = 0
        self.total_disposed = 0
        self.total_violations = 0
        self.episode_rewards = []

        self._newly_disposed_this_step = set()
        self._last_acted_case_id = -1
        self._last_acted_action_type = -1
        self._consecutive_same = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def action_masks(self) -> np.ndarray:
        case_mask = np.array([
            c["status"] != "disposed" for c in self.cases
        ], dtype=bool)

        if not case_mask.any():
            case_mask = np.ones(self.num_cases, dtype=bool)

        judge_mask = np.ones(self.num_judges, dtype=bool)

        for j, judge in enumerate(self.judges):
            if judge.get("weekly_hearings", 0) >= judge.get("weekly_capacity", 15):
                judge_mask[j] = False

        if not judge_mask.any():
            judge_mask = np.ones(self.num_judges, dtype=bool)

        has_unverified = any(
            c["evidence_status"] in ["pending", "collected"]
            and c["status"] != "disposed"
            for c in self.cases
        )
        has_fast_trackable = any(
            not c.get("fast_tracked", False)
            and c["status"] != "disposed"
            and (
                (c["case_type"] == "criminal" and c["severity"] in ["serious", "heinous"])
                or c["age_days"] > 1095
                or c.get("victim_vulnerable", False)
                or c["bnss_remaining"] < 30
                or (c["case_type"] == "family" and c["severity"] in ["serious", "heinous"])
            )
            for c in self.cases
        )

        action_type_mask = np.array([
            True,              
            has_unverified,    
            has_fast_trackable, 
        ], dtype=bool)

        if not action_type_mask.any():
            action_type_mask = np.ones(3, dtype=bool)

        return np.concatenate([case_mask, judge_mask, action_type_mask])

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        self.current_step += 1

        if self.current_step % 7 == 1:
            for j in self.judges:
                j["weekly_hearings"] = 0

        for j in self.judges:
            j["cases_today"] = 0

        for c in self.cases:
            c["is_scheduled"] = False

        self._newly_disposed_this_step = set()

        case_idx = int(action[0])
        judge_idx = int(action[1])
        action_type = int(action[2])

        case_idx = min(case_idx, len(self.cases) - 1)
        judge_idx = min(judge_idx, len(self.judges) - 1)

        case = self.cases[case_idx]
        judge = self.judges[judge_idx]

        if (
            case_idx == self._last_acted_case_id
            and action_type == self._last_acted_action_type
        ):
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0

        self._last_acted_case_id = case_idx
        self._last_acted_action_type = action_type

        evidence_at_hearing = case["evidence_status"]

        step_disposed = 0
        action_success = False

        if action_type == 0:
            action_success, disposed = self._schedule_hearing(case, judge)
            step_disposed += disposed
            if disposed > 0:
                self._newly_disposed_this_step.add(case["id"])

        elif action_type == 1:
            action_success = self.evidence_engine.verify(
                case, self.np_random
            )
            evidence_at_hearing = None

        elif action_type == 2:
            action_success = self._fast_track_case(case)
            evidence_at_hearing = None

        self._advance_time()

        step_violations = self.bnss_engine.check_violations(self.cases)
        self.total_violations += step_violations

        heuristic_bonus = self._calculate_heuristic_bonus(
            case_idx, judge_idx, action_type
        )

        reward = self.reward_calculator.calculate(
            cases=self.cases,
            step_disposed=step_disposed,
            step_violations=step_violations,
            action_success=action_success,
            action_type=action_type,
            judges=self.judges,
            newly_disposed_ids=self._newly_disposed_this_step,
            consecutive_same_case_and_action=self._consecutive_same,
            acted_case_idx=case_idx,
            acted_case_evidence_at_hearing=evidence_at_hearing,
        )

        reward += heuristic_bonus

        self.episode_rewards.append(reward)

        active_cases = sum(
            1 for c in self.cases if c["status"] != "disposed"
        )
        terminated = active_cases == 0
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()
        info["step_disposed"] = step_disposed
        info["step_violations"] = step_violations
        info["action_success"] = action_success
        info["action_type"] = action_type
        info["heuristic_bonus"] = heuristic_bonus

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _calculate_heuristic_bonus(
        self, case_idx: int, judge_idx: int, action_type: int
    ) -> float:
        bonus = 0.0
        case = self.cases[case_idx]
        judge = self.judges[judge_idx]

        if case["status"] == "disposed":
            return 0.0

        if case["bnss_remaining"] < 30:
            bonus += 1.0  
        elif case["bnss_remaining"] < 60:
            bonus += 0.5

        if judge["specialization"] == case["case_type"]:
            bonus += 0.5

        if action_type == 1 and case["evidence_status"] in ["pending", "collected"]:
            bonus += 0.5  

        if action_type == 0 and case["evidence_status"] == "verified":
            bonus += 0.5  

        if action_type == 0 and case["evidence_status"] == "pending":
            bonus -= 0.3

        if action_type == 2 and case.get("victim_vulnerable", False):
            bonus += 0.5

        if case.get("victim_vulnerable", False):
            bonus += 0.3

        return bonus

    def _schedule_hearing(
        self, case: Dict, judge: Dict
    ) -> Tuple[bool, int]:
        if case["status"] == "disposed":
            return False, 0

        if judge["weekly_hearings"] >= judge["weekly_capacity"]:
            return False, 0

        judge["weekly_hearings"] += 1
        judge["cases_today"] += 1
        judge["total_cases_handled"] += 1
        case["hearings"] += 1
        case["is_scheduled"] = True

        disposal_prob = 0.08

        if case["evidence_status"] == "verified":
            disposal_prob += 0.15
        elif case["evidence_status"] == "tampered":
            disposal_prob += 0.05
        elif case["evidence_status"] == "collected":
            disposal_prob += 0.05

        if judge["specialization"] == case["case_type"]:
            disposal_prob += 0.12
        else:
            disposal_prob += 0.03

        if case["age_days"] > 365:
            disposal_prob += 0.05
        if case["age_days"] > 730:
            disposal_prob += 0.05
        if case["age_days"] > 1460:
            disposal_prob += 0.05

        if case["bnss_remaining"] < 30:
            disposal_prob += 0.15
        elif case["bnss_remaining"] < 60:
            disposal_prob += 0.08

        if case.get("fast_tracked", False):
            disposal_prob += 0.10

        severity_mod = {
            "petty": 0.05, "moderate": 0.0,
            "serious": -0.03, "heinous": -0.05,
        }
        disposal_prob += severity_mod.get(case["severity"], 0.0)

        if case["hearings"] > 5:
            disposal_prob += 0.05
        if case["hearings"] > 10:
            disposal_prob += 0.05

        disposal_prob = max(0.02, min(disposal_prob, 0.70))

        if self.np_random.random() < disposal_prob:
            case["status"] = "disposed"
            self.total_disposed += 1
            return True, 1

        return True, 0

    def _fast_track_case(self, case: Dict) -> bool:
        if case["status"] == "disposed":
            return False
        if case.get("fast_tracked", False):
            return False

        eligible = (
            (case["case_type"] == "criminal"
             and case["severity"] in ["serious", "heinous"])
            or case["age_days"] > 1095
            or case.get("victim_vulnerable", False)
            or case["bnss_remaining"] < 30
            or (case["case_type"] == "family"
                and case["severity"] in ["serious", "heinous"])
        )

        if eligible:
            case["fast_tracked"] = True
            return True
        return False

    def _advance_time(self):
        for case in self.cases:
            if case["status"] != "disposed":
                case["age_days"] += 1
                case["bnss_remaining"] -= 1

    def _initialize_judges(self) -> list:
        specializations = ["criminal", "civil", "family", "commercial"]
        judges = []
        for i in range(self.num_judges):
            judges.append({
                "id": i,
                "specialization": specializations[i % len(specializations)],
                "cases_today": 0,
                "total_cases_handled": 0,
                "efficiency_score": float(self.np_random.uniform(0.6, 0.95)),
                "weekly_hearings": 0,
                "weekly_capacity": 15,
            })
        return judges

    def _get_observation(self) -> np.ndarray:
        obs = []
        type_map = {"criminal": 0.25, "civil": 0.50, "family": 0.75, "commercial": 1.00}
        severity_map = {"petty": 0.25, "moderate": 0.50, "serious": 0.75, "heinous": 1.00}
        evidence_map = {"tampered": 0.00, "pending": 0.33, "collected": 0.66, "verified": 1.00}
        max_age = 3650.0
        max_bnss = 545.0

        for case in self.cases:
            if case["status"] == "disposed":
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                obs.extend([
                    type_map.get(case["case_type"], 0.5),
                    severity_map.get(case["severity"], 0.5),
                    min(case["age_days"] / max_age, 1.0),
                    evidence_map.get(case["evidence_status"], 0.33),
                    max(min(case["bnss_remaining"] / max_bnss, 1.0), 0.0),
                    1.0 if case.get("is_scheduled", False) else 0.0,
                ])

        active = [c for c in self.cases if c["status"] != "disposed"]
        pending_ratio = len(active) / max(self.num_cases, 1)
        avg_age = 0.0
        if active:
            avg_age = min(np.mean([c["age_days"] for c in active]) / max_age, 1.0)
        compliance = self.bnss_engine.get_compliance_score(self.cases)
        obs.extend([pending_ratio, avg_age, compliance])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        active = [c for c in self.cases if c["status"] != "disposed"]
        return {
            "step": self.current_step,
            "total_disposed": self.total_disposed,
            "active_cases": len(active),
            "total_violations": self.total_violations,
            "avg_case_age": float(np.mean([c["age_days"] for c in active])) if active else 0.0,
            "disposal_rate": self.total_disposed / max(self.current_step, 1),
            "compliance_score": self.bnss_engine.get_compliance_score(self.cases),
            "evidence_integrity": self.evidence_engine.get_integrity_score(self.cases),
            "evidence_stats": self.evidence_engine.get_statistics(),
            "episode_reward_sum": sum(self.episode_rewards),
        }

    def render(self):
        info = self._get_info()
        active = [c for c in self.cases if c["status"] != "disposed"]
        output = f"""
╔══════════════════════════════════════════════════════════╗
║         NYAYA DISTRICT COURT — Day {self.current_step:>4}                  ║
║         NyayaEnv v2.0 HYBRID (NyayaSaathiAI)            ║
╠══════════════════════════════════════════════════════════╣
║  Active Cases:      {info['active_cases']:>3} / {self.num_cases:<3}                         ║
║  Disposed:          {info['total_disposed']:>3}                                 ║
║  BNSS Violations:   {info['total_violations']:>3}                                 ║
║  Avg Case Age:      {info['avg_case_age']:>7.0f} days                          ║
║  Compliance:        {info['compliance_score']:>6.1%}                            ║
║  Evidence:          {info['evidence_integrity']:>6.1%}                            ║
║  Total Reward:      {info['episode_reward_sum']:>8.1f}                          ║
╠══════════════════════════════════════════════════════════╣"""
        if active:
            urgent = sorted(active, key=lambda c: c["bnss_remaining"])[:5]
            output += "\n║  TOP 5 URGENT:                                           ║"
            for c in urgent:
                st = self.bnss_engine.get_status_label(c)
                ev = c["evidence_status"][:3].upper()
                ft = "⚡" if c.get("fast_tracked") else " "
                vl = "👶" if c.get("victim_vulnerable") else " "
                output += (
                    f"\n║  {st} #{c['id']:>2} "
                    f"{c['case_type']:>8} {c['severity']:>8} "
                    f"Age:{c['age_days']:>4}d BNSS:{c['bnss_remaining']:>4}d "
                    f"{ev} {ft}{vl} ║"
                )
        output += "\n╚══════════════════════════════════════════════════════════╝"
        print(output)
        return output

    def close(self):
        pass
