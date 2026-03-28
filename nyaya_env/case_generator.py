import numpy as np
from typing import List, Dict


class CaseGenerator:

    CASE_TYPES = ["criminal", "civil", "family", "commercial"]
    CASE_TYPE_PROBS = [0.35, 0.40, 0.15, 0.10]

    SEVERITIES = ["petty", "moderate", "serious", "heinous"]
    SEVERITY_PROBS = [0.40, 0.30, 0.20, 0.10]

    EVIDENCE_STATUSES = ["pending", "collected", "verified"]
    EVIDENCE_PROBS = [0.40, 0.40, 0.20]

    BNSS_DEADLINES = {
        ("criminal", "petty"): 90,
        ("criminal", "moderate"): 90,
        ("criminal", "serious"): 180,
        ("criminal", "heinous"): 180,
        ("civil", "petty"): 365,
        ("civil", "moderate"): 365,
        ("civil", "serious"): 545,
        ("civil", "heinous"): 545,
        ("family", "petty"): 180,
        ("family", "moderate"): 270,
        ("family", "serious"): 365,
        ("family", "heinous"): 365,
        ("commercial", "petty"): 270,
        ("commercial", "moderate"): 365,
        ("commercial", "serious"): 545,
        ("commercial", "heinous"): 545,
    }

    def __init__(self, seed=None):
        self.seed = seed

    def generate_cases(
        self, num_cases: int, rng: np.random.Generator
    ) -> List[Dict]:
        cases = []
        for i in range(num_cases):
            case_type = rng.choice(self.CASE_TYPES, p=self.CASE_TYPE_PROBS)
            severity = rng.choice(self.SEVERITIES, p=self.SEVERITY_PROBS)
            evidence = rng.choice(
                self.EVIDENCE_STATUSES, p=self.EVIDENCE_PROBS
            )

            age_days = int(rng.exponential(scale=400))
            age_days = min(age_days, 3650)

            max_deadline = self.BNSS_DEADLINES.get(
                (case_type, severity), 365
            )
            bnss_remaining = max(max_deadline - age_days, 0)

            victim_vulnerable = bool(rng.random() < 0.15)
            witness_count = max(1, min(int(rng.exponential(scale=3)), 15))
            hearings = min(
                int(rng.exponential(scale=max(age_days / 120, 1))), 50
            )

            case = {
                "id": i,
                "case_type": case_type,
                "severity": severity,
                "age_days": age_days,
                "evidence_status": evidence,
                "evidence_hash": None,
                "bnss_remaining": bnss_remaining,
                "bnss_max_deadline": max_deadline,
                "is_scheduled": False,
                "fast_tracked": False,
                "status": "active",
                "hearings": hearings,
                "victim_vulnerable": victim_vulnerable,
                "witness_count": witness_count,
                "bnss_violated": False,
            }
            cases.append(case)

        return cases
