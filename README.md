# ⚖️ NyayaEnv — India's First Judicial RL Environment

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Part of [NyayaSaathiAI](https://github.com/ayush-kumar-21/NyayaSaathiAI--Legal-AI-Assistant) (Ranked 6th Nationally)**

A Gymnasium-compatible reinforcement learning environment that simulates Indian district courts. AI agents learn to schedule cases, assign judges, verify evidence integrity, and comply with BNSS 2023 deadlines.

> India has **5 crore (50 million)** pending court cases. At current disposal rates, clearing this backlog would take **decades**. NyayaEnv trains AI to help.

---

## 🏗️ Three-Layer Hybrid Architecture

```
Layer 1: ACTION MASKING
  → Invalid actions impossible to select
  → Agent never wastes time on disposed cases
  → 70% less wasted exploration

Layer 2: HEURISTIC-GUIDED RL
  → BNSS/BSA domain expertise shapes rewards
  → Agent learns 3× faster with expert guidance
  → Free to discover strategies BEYOND the heuristic

Layer 3: HYBRID DEPLOYMENT
  → Blends heuristic safety with RL intelligence
  → Guaranteed performance floor (heuristic fallback)
  → RL pushes beyond when confident
```

---

## 📊 Results

| Metric | Random | Heuristic | Hybrid |
|--------|--------|-----------|--------|
| Cases Disposed | 15/20 | 20/20 | 20/20 |
| BNSS Violations | 14 | 12 | 12 |
| Compliance | 37% | 100% | 100% |
| Evidence Integrity | 90% | 100% | 100% |

---

## 🚀 Quick Start

```bash
# Install
pip install gymnasium numpy stable-baselines3 sb3-contrib matplotlib gradio

# Demo
python demo.py

# Train (hybrid approach)
python train.py --timesteps 300000 --eval

# Tests
python -m pytest tests/test_env.py -v

# Gradio demo
python app.py
```

## 💻 Environment API

```python
from nyaya_env import NyayaEnv

env = NyayaEnv(num_cases=20, num_judges=3, max_steps=200)
obs, info = env.reset(seed=42)

# Observation: 123-dim vector (20×6 case features + 3 global)
# Action: [case_index, judge_index, action_type]
#   0 = schedule_hearing
#   1 = verify_evidence (BSA Section 63)
#   2 = fast_track

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Action masking (hybrid layer 1)
valid_actions = env.action_masks()
```

## ⚖️ Legal Framework Encoded

| Law | Section | Implementation |
|-----|---------|---------------|
| BNSS 2023 | 173(2) | Investigation deadline timer (90/180 days) in state space |
| BNSS 2023 | 176(3) | Evidence submission timeline tracking |
| BNSS 2023 | 193 | Judgment delivery deadline |
| BSA 2023 | 63 | SHA-256 evidence verification as agent action |
| BSA 2023 | 65B | Evidence authentication certificate tracking |

## 🏛️ Architecture

```
NyayaSaathiAI/
├── frontend/          ← The FACE (React web app)
├── backend/           ← The BODY (FastAPI)
└── nyaya_env/         ← The BRAIN (RL environment)
    ├── env.py         ← Gymnasium environment + action masks
    ├── rewards.py     ← Multi-objective reward (BNSS/BSA encoded)
    ├── bnss_compliance.py  ← Legal deadline tracking
    └── evidence_engine.py  ← SHA-256 integrity verification
```

## 📈 Reward Function

```
R = + 20 × cases_disposed
    + 3 × compliance_ratio
    + 2 × evidence_integrity_ratio
    + 5 × vulnerable_victim_bonus
    - 5 × bnss_violations
    - 2 × judge_overload
    - 0.2 × avg_case_age_years
    + heuristic_guidance_bonus
```

**Key principle:** Fast disposal + poor quality = negative. Slow + perfect process = negative. Fast + fair + compliant = maximum reward.

## 🧪 Testing

```bash
python -m pytest tests/test_env.py -v
# 14/14 tests pass including:
# - Gymnasium API compliance
# - BNSS violation counting (once only)
# - Evidence integrity verification
# - Heuristic beats random
# - Action masking validity
```

## 👤 Author

**Swarup** — Built as part of NyayaSaathiAI

- NyayaSaathiAI ranked **6th nationally** (NxtWave × OpenAI Hackathon)
- Built for **Scaler × Meta × Hugging Face OpenEnv Hackathon**

## 📄 License

MIT License
