import hashlib
import json
from typing import Dict, List
import numpy as np


class EvidenceIntegrityEngine:

    TAMPER_PROBABILITY = 0.05
    DETECTION_PROBABILITY = 0.90
    COLLECTION_SUCCESS_RATE = 0.85

    def __init__(self):
        self.evidence_ledger: Dict[int, str] = {}
        self.total_verifications = 0
        self.tampering_detected = 0
        self.tampering_missed = 0

    def verify(self, case: Dict, rng: np.random.Generator) -> bool:
        if case["status"] == "disposed":
            return False

        self.total_verifications += 1

        if case["evidence_status"] == "pending":
            return self._collect_evidence(case, rng)
        elif case["evidence_status"] == "collected":
            return self._verify_integrity(case, rng)
        return False

    def _collect_evidence(
        self, case: Dict, rng: np.random.Generator
    ) -> bool:
        if rng.random() < self.COLLECTION_SUCCESS_RATE:
            case["evidence_status"] = "collected"
            evidence_hash = self._generate_hash(case)
            case["evidence_hash"] = evidence_hash
            self.evidence_ledger[case["id"]] = evidence_hash
            return True
        return False

    def _verify_integrity(
        self, case: Dict, rng: np.random.Generator
    ) -> bool:
        is_tampered = rng.random() < self.TAMPER_PROBABILITY
        if is_tampered:
            detected = rng.random() < self.DETECTION_PROBABILITY
            if detected:
                case["evidence_status"] = "tampered"
                self.tampering_detected += 1
            else:
                case["evidence_status"] = "verified"
                self.tampering_missed += 1
        else:
            case["evidence_status"] = "verified"
            current_hash = self._generate_hash(case)
            if case["id"] not in self.evidence_ledger:
                self.evidence_ledger[case["id"]] = current_hash
        return True

    def _generate_hash(self, case: Dict) -> str:
        evidence_data = json.dumps(
            {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "severity": case["severity"],
                "witness_count": case["witness_count"],
            },
            sort_keys=True,
        )
        return hashlib.sha256(evidence_data.encode()).hexdigest()

    def get_integrity_score(self, cases: List[Dict]) -> float:
        active = [c for c in cases if c["status"] != "disposed"]
        if not active:
            return 1.0
        scores = {
            "verified": 1.0,
            "tampered": 0.8,
            "collected": 0.3,
            "pending": 0.1,
        }
        total = sum(
            scores.get(c["evidence_status"], 0.1) for c in active
        )
        return total / len(active)

    def get_statistics(self) -> Dict:
        return {
            "total_verifications": self.total_verifications,
            "tampering_detected": self.tampering_detected,
            "tampering_missed": self.tampering_missed,
            "ledger_entries": len(self.evidence_ledger),
        }

    def reset(self):
        self.evidence_ledger = {}
        self.total_verifications = 0
        self.tampering_detected = 0
        self.tampering_missed = 0
