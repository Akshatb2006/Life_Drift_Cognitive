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

# Life Drift — Cognitive Load Planning Benchmark

A sequential decision-making benchmark where an agent must guide a simulated user toward long-term goals under realistic cognitive constraints: limited energy, accumulating fatigue, stochastic behavioral drift, and a nonlinear burnout cascade.

## Benchmark Objective

The environment evaluates three distinct planning skills under human-like dynamics:

1. **Short-horizon correction** — recover alignment from a degraded state.
2. **Trade-off balancing** — sustain productivity while containing fatigue.
3. **Long-horizon planning under risk** — keep mixed metrics stable while avoiding an irreversible burnout region.

Formally, the agent solves a finite-horizon control problem over a 5-dimensional continuous cognitive state `(alignment, drift, energy, fatigue, focus) ∈ [0,1]⁵` acting on a discrete intervention set. Transitions are stochastic within per-action bounds and include a nonlinear cascade above `fatigue > 0.85` that makes late-episode recovery increasingly expensive — **rewarding foresight over reactive play**.

## Why the Task Resists Greedy Play

Most productivity-style environments reduce to "push on the graded axis." Life Drift is designed to resist that:

- **Every action trades off.** There is no free action. `insert_break` restores energy but drifts alignment. `prioritize_goal` boosts alignment but accelerates fatigue. `do_nothing` drifts and loses alignment. The agent must choose *which* axis to pay with.
- **Fatigue has a soft cliff.** Above `fatigue = 0.85`, focus and alignment begin to degrade *on their own* and drift accelerates — a cascade that punishes agents who optimize short-term reward at the cost of recovery margin.
- **Dynamics are stochastic.** Each action's magnitude is drawn from a bounded range, so policies that memorize a single trajectory fail to generalize across seeds.
- **Time-of-day modulates cost.** Late afternoon and evening compound fatigue and reduce focus, so the same action has different expected value depending on when in the episode it fires.

Together these make the environment a test of **planning under delayed, irreversible, and stochastic consequences** — not just reactive heuristics.

## Observation Space

The full agent view is a `LifeDriftObservation` with these fields:

| Field | Type | Range | Description |
|---|---|---|---|
| `goal_alignment_score` | float | [0, 1] | How aligned recent user actions are with stated goals |
| `drift_score` | float | [0, 1] | Distance from goal-directed behavior |
| `energy_level` | float | [0, 1] | Available cognitive energy |
| `fatigue` | float | [0, 1] | Accumulated fatigue (cascade threshold at 0.85) |
| `focus_score` | float | [0, 1] | Current concentration |
| `goals` | List[str] | — | Active user goals (per task) |
| `recent_actions` | List[str] | — | Last 3 simulated user behaviors |
| `time_of_day` | str | — | Simulated diurnal phase |
| `step_number`, `max_steps` | int | — | Episode progress |

## Action Space

| Action | Dominant effect | Cost |
|---|---|---|
| `suggest_task` | alignment ↑, drift ↓ | energy ↓, fatigue ↑ |
| `prioritize_goal` | alignment ↑↑, drift ↓↓ | energy ↓↓, fatigue ↑ |
| `insert_break` | energy ↑, fatigue ↓ | drift ↑, alignment ↓ slight |
| `reschedule_task` | alignment ↑ moderate, drift ↓ | energy ↓ slight |
| `reduce_difficulty` | fatigue ↓, alignment ↑ slight | minimal |
| `do_nothing` | passive energy recovery | drift ↑, alignment ↓ |

All actions accept an optional `target_goal`. Effect magnitudes are drawn stochastically within per-action bounds.

## Reward

Dense shaped reward, clipped to `[-1, 1]`:

```
reward = 0.4 · Δalignment
       − 0.3 · Δdrift
       − 0.2 · max(0, Δfatigue)          # only penalize fatigue increases
       + energy_window_bonus             # +0.1 if energy ∈ [0.4, 0.7] else -0.05
       + state_shaping                   # +/- bonuses for healthy / critical states
```

The reward signal is intentionally dense and shaped to be trainable. **Grading is computed separately** (see below) so agents cannot fully game the reward proxy.

## Tasks

All tasks use fixed initial conditions and accept a `seed` parameter for reproducibility.

### Task 1 — Drift Correction *(10 steps, easy)*
**Tests:** reactive short-term control.
**Initial:** `alignment=0.2, drift=0.8, energy=0.7, fatigue=0.3`.
**Grader:** `0.6 · min(alignment / 0.7, 1) + 0.4 · min((1 − drift) / 0.7, 1)`

### Task 2 — Energy Balance *(15 steps, medium)*
**Tests:** sustained trade-off management between two competing axes.
**Initial:** `alignment=0.8, fatigue=0.8, energy=0.3` — already productive, already burning out.
**Grader:** `0.4 · avg_alignment + 0.3 · (1 − avg_fatigue) + 0.3 · (1 − final_fatigue)`

### Task 3 — Long-term Stability *(20 steps, hard)*
**Tests:** long-horizon planning under the burnout cascade with a 4-goal portfolio.
**Initial:** all metrics at `0.5` (high-variance starting point).
**Grader:** weighted combination of inverse average drift, energy-variance stability, worst-case fatigue, and mean step reward.

All graders clamp to the open interval `(0.01, 0.99)` so every run produces a meaningful bounded scalar.

## Running

### Docker
```bash
docker build -t life-drift-env .
docker run -p 8000:8000 life-drift-env
```

### Local
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Baseline LLM agent
```bash
export HF_TOKEN=your_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start episode. Body: `{task_id, seed, episode_id}` |
| `/step` | POST | Take action. Body: `{episode_id, action}` |
| `/state/{episode_id}` | GET | Current state metadata (includes score) |
| `/grade` | POST | Get final grader score. Body: `{episode_id}` |
| `/tasks` | GET | List available tasks |
| `/cleanup/{episode_id}` | POST | Clean up session |
| `/health` | GET | Health check |

## Environment Variables

| Var | Default |
|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` / `API_KEY` | — |
| `IMAGE_NAME` | — (Docker image tag for containerized runs) |
