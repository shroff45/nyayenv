# NyayaEnv

India's first reinforcement learning environment for judicial case management.

Part of [NyayaSaathiAI](https://github.com/ayush-kumar-21/NyayaSaathiAI--Legal-AI-Assistant)

## Quick Start

```python
from nyaya_env import NyayaEnv

env = NyayaEnv(num_cases=20, num_judges=3, max_steps=200)
obs, info = env.reset()

action = env.action_space.sample()  # [case_idx, judge_idx, action_type]
obs, reward, terminated, truncated, info = env.step(action)
```

## Actions
| Code | Action | Description |
|------|--------|-------------|
| 0 | Schedule Hearing | Conduct hearing, may dispose |
| 1 | Verify Evidence | BSA 63 compliance check |
| 2 | Fast Track | Priority processing |

## Legal Framework
| Law | Section | Implementation |
|-----|---------|---------------|
| BNSS 2023 | 173(2) | Timer in state + violation penalty |
| BSA 2023 | 63 | SHA-256 verification as action |

## Run
```bash
python demo.py
python train.py --timesteps 100000 --eval
python -m pytest tests/ -v
```
