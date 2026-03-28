import numpy as np


class HeuristicAgent:

    def __init__(self, env):
        self.env = env

    def act(self, observation):
        cases = self.env.cases
        judges = self.env.judges

        best_case_idx = 0
        best_urgency = -1.0

        for i, case in enumerate(cases):
            if case["status"] == "disposed":
                continue

            urgency = 1.0 / max(case["bnss_remaining"], 1)

            if case.get("victim_vulnerable", False):
                urgency *= 1.5
            if case["severity"] == "heinous":
                urgency *= 1.3
            if case["severity"] == "serious":
                urgency *= 1.1

            if urgency > best_urgency:
                best_urgency = urgency
                best_case_idx = i

        case = cases[best_case_idx]

        best_judge_idx = 0
        best_judge_score = -999.0

        for j, judge in enumerate(judges):
            score = 0.0
            if judge["specialization"] == case["case_type"]:
                score += 10.0
            score -= judge["cases_today"] * 2.0
            score += judge["efficiency_score"] * 5.0

            if score > best_judge_score:
                best_judge_score = score
                best_judge_idx = j

        if case["evidence_status"] in ["pending", "collected"]:
            action_type = 1
        elif (
            case["severity"] in ["serious", "heinous"]
            and case["case_type"] in ["criminal", "family"]
            and not case.get("fast_tracked", False)
        ):
            action_type = 2
        elif (
            case["age_days"] > 1095
            and not case.get("fast_tracked", False)
        ):
            action_type = 2
        else:
            action_type = 0

        return np.array([best_case_idx, best_judge_idx, action_type])

    def __repr__(self):
        return "HeuristicAgent(FIFO+Urgency)"
