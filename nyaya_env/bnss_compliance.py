from typing import List, Dict


class BNSSComplianceEngine:

    CRITICAL_THRESHOLD = 15
    WARNING_THRESHOLD = 30
    CAUTION_THRESHOLD = 60

    def check_violations(self, cases: List[Dict]) -> int:
        """
        Count only NEW violations. Once a case is flagged
        as violated, it is never counted again.
        """
        new_violations = 0
        for case in cases:
            if case["status"] == "disposed":
                continue
            if case["bnss_remaining"] <= 0 and not case.get(
                "bnss_violated", False
            ):
                case["bnss_violated"] = True
                new_violations += 1
        return new_violations

    def get_compliance_score(self, cases: List[Dict]) -> float:
        active = [c for c in cases if c["status"] != "disposed"]
        if not active:
            return 1.0
        compliant = sum(1 for c in active if c["bnss_remaining"] > 0)
        return compliant / len(active)

    def get_urgency_score(self, case: Dict) -> float:
        if case["status"] == "disposed":
            return 0.0
        remaining = case["bnss_remaining"]
        max_deadline = case.get("bnss_max_deadline", 365)

        if remaining <= 0:
            return 1.0
        elif remaining <= self.CRITICAL_THRESHOLD:
            return 0.95
        elif remaining <= self.WARNING_THRESHOLD:
            return 0.80
        elif remaining <= self.CAUTION_THRESHOLD:
            return 0.60
        else:
            time_used = 1.0 - (remaining / max(max_deadline, 1))
            return max(time_used * 0.4, 0.10)

    def get_status_label(self, case: Dict) -> str:
        remaining = case["bnss_remaining"]
        if remaining <= 0:
            return "🔴 VIOLATED"
        elif remaining <= self.CRITICAL_THRESHOLD:
            return "🔴 CRITICAL"
        elif remaining <= self.WARNING_THRESHOLD:
            return "🟡 WARNING"
        elif remaining <= self.CAUTION_THRESHOLD:
            return "🟡 CAUTION"
        else:
            return "🟢 COMPLIANT"
