---
title: Life Drift Cognitive Env
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Life Drift & Cognitive Load Environment

An OpenEnv environment that simulates **productivity coaching** — an AI agent must guide a simulated user toward their goals while managing cognitive energy and preventing burnout.

## Motivation

Everyone struggles with staying on track and avoiding burnout. This environment models the real tension between **pushing toward goals** and **managing energy** — a task that human coaches, therapists, and productivity systems handle daily. Training AI agents on this task has direct applications in:

- AI-powered productivity apps
- Digital wellness coaching
- Adaptive task schedulers
- Burnout prevention systems

## How It Works

The environment simulates a user with:
- **Goals** they want to achieve (fitness, study, side projects)
- **Energy** that depletes with work and recovers with rest
- **Drift** that increases when they go off-track
- **Fatigue** that accumulates and can cause burnout cascades

The agent acts as a life coach, making micro-interventions each step.

## Observation Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `goals` | `List[str]` | — | User's active goals |
| `recent_actions` | `List[str]` | — | Last 3 simulated user actions |
| `goal_alignment_score` | `float` | [0, 1] | How aligned actions are with goals |
| `energy_level` | `float` | [0, 1] | Current energy/motivation |
| `fatigue` | `float` | [0, 1] | Accumulated fatigue (burnout risk) |
| `focus_score` | `float` | [0, 1] | Current concentration level |
| `drift_score` | `float` | [0, 1] | How far user has drifted from goals |
| `time_of_day` | `str` | — | Simulated time period |
| `step_number` | `int` | — | Current step |
| `max_steps` | `int` | — | Episode length |

## Action Space

| Action | Description | Effect |
|--------|-------------|--------|
| `suggest_task` | Suggest working on a goal | Alignment ↑, Energy ↓, Drift ↓ |
| `insert_break` | Suggest a rest break | Energy ↑, Fatigue ↓, Drift ↑ slightly |
| `reschedule_task` | Reorganize schedule | Alignment ↑ moderate, Energy ↓ slight |
| `reduce_difficulty` | Simplify current task | Fatigue ↓, Alignment ↑ slight |
| `prioritize_goal` | Focus on one specific goal | Alignment ↑↑, Energy ↓↓ |
| `do_nothing` | No intervention | Drift ↑, Natural recovery |

Actions that target specific goals accept `target_goal` parameter.

## Reward Function

Dense reward signal computed each step:

```
reward = 0.4 * alignment_improvement
       - 0.3 * drift_increase
       + 0.1 * energy_balance_bonus
       - 0.2 * fatigue_increase
       + state_bonuses/penalties
```

- **Bonuses**: High alignment + low drift (+0.1), good energy + low fatigue (+0.05)
- **Penalties**: Critical fatigue > 0.9 (-0.15), high drift > 0.8 (-0.1)
- Burnout cascade: fatigue > 0.85 causes focus, alignment, and drift to degrade

## Tasks

### Task 1: Drift Correction (Easy)
- **Scenario**: User is wasting time — low alignment (0.2), high drift (0.8)
- **Goal**: Bring alignment > 0.7 and drift < 0.3 within 10 steps
- **Grader**: `0.6 * (alignment/0.7) + 0.4 * ((1-drift)/0.7)`

### Task 2: Energy Balance (Medium)
- **Scenario**: User is productive but exhausted — high alignment (0.8), high fatigue (0.8)
- **Goal**: Maintain alignment > 0.6 while reducing fatigue < 0.5 over 15 steps
- **Grader**: `0.4 * avg_alignment + 0.3 * (1-avg_fatigue) + 0.3 * (1-final_fatigue)`

### Task 3: Long-term Stability (Hard)
- **Scenario**: Mixed behavior — all metrics at 0.5, 4 goals to balance over 20 steps
- **Goal**: Optimize both low drift AND stable energy with no burnout spikes
- **Grader**: `0.3 * (1-avg_drift) + 0.25 * energy_stability + 0.25 * (1-max_fatigue) + 0.2 * avg_reward`

## Setup

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t life-drift-env .
docker run -p 8000:8000 life-drift-env
```

### Run Baseline Inference

```bash
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export ENV_URL=http://localhost:8000

python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode `{task_id, seed, episode_id}` |
| `/step` | POST | Take action `{episode_id, action}` |
| `/state/{episode_id}` | GET | Get current state metadata |
| `/grade` | POST | Get grader score `{episode_id}` |
| `/tasks` | GET | List available tasks |
| `/cleanup/{episode_id}` | POST | Clean up session |

## Expected Baseline Scores

With a simple rule-based heuristic strategy:
- **drift_correction** (easy): ~0.75–0.85
- **energy_balance** (medium): ~0.55–0.70
- **long_term_stability** (hard): ~0.45–0.60

With an LLM agent (gpt-4o-mini):
- Scores typically 5-15% higher than rule-based baseline

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | HuggingFace / API key | — |
| `OPENAI_API_KEY` | OpenAI API key (or HF_TOKEN) | — |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |
