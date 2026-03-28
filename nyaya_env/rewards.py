import numpy as np
from typing import List, Dict, Set, Optional


class RewardCalculator:

    DISPOSAL_REWARD = 20.0
    COMPLIANCE_BONUS = 3.0
    EVIDENCE_VERIFIED_BONUS = 2.0
    VULNERABLE_DISPOSAL_BONUS = 5.0

    VIOLATION_PENALTY = -5.0
    INVALID_ACTION_PENALTY = -0.5
    UNVERIFIED_HEARING_PENALTY = -2.0
    AGE_PENALTY_SCALE = -0.2

    SPAM_PENALTY_PER_REPEAT = -1.0
    SPAM_PENALTY_MAX = -5.0

    OVERLOAD_PENALTY = -2.0

    def calculate(
        self,
        cases: List[Dict],
        step_disposed: int,
        step_violations: int,
        action_success: bool,
        action_type: int,
        judges: List[Dict],
        newly_disposed_ids: Set[int],
        consecutive_same_case_and_action: int,
        acted_case_idx: int,
        acted_case_evidence_at_hearing: Optional[str] = None,
    ) -> float:

        reward = 0.0

        reward += self.DISPOSAL_REWARD * step_disposed

        reward += self.VIOLATION_PENALTY * step_violations

        active = [c for c in cases if c["status"] != "disposed"]
        if active:
            compliant = sum(1 for c in active if c["bnss_remaining"] > 0)
            reward += self.COMPLIANCE_BONUS * (compliant / len(active))

        if active:
            verified = sum(
                1 for c in active if c["evidence_status"] == "verified"
            )
            reward += self.EVIDENCE_VERIFIED_BONUS * (verified / len(active))

        for judge in judges:
            weekly = judge.get("weekly_hearings", 0)
            capacity = judge.get("weekly_capacity", 15)
            if weekly > capacity:
                reward += self.OVERLOAD_PENALTY * (weekly - capacity)

        if active:
            avg_age_years = np.mean([c["age_days"] for c in active]) / 365.0
            reward += self.AGE_PENALTY_SCALE * avg_age_years

        if not action_success:
            reward += self.INVALID_ACTION_PENALTY

        for c in cases:
            if c["id"] in newly_disposed_ids and c.get("victim_vulnerable", False):
                reward += self.VULNERABLE_DISPOSAL_BONUS

        if (
            action_type == 0
            and action_success
            and acted_case_evidence_at_hearing is not None
            and acted_case_evidence_at_hearing in ["pending", "collected"]
        ):
            reward += self.UNVERIFIED_HEARING_PENALTY

        # Spam penalty with HARD CAP
        if consecutive_same_case_and_action > 2:
            spam_level = consecutive_same_case_and_action - 2
            raw_spam = self.SPAM_PENALTY_PER_REPEAT * spam_level
            reward += max(raw_spam, self.SPAM_PENALTY_MAX)

        return reward
