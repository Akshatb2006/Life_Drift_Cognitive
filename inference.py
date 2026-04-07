"""Inference script for the Life Drift & Cognitive Load environment.

Uses an OpenAI-compatible LLM to act as a productivity coaching agent.
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# --- Configuration (mandatory env vars) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "life_drift_cognitive_env"
TASKS = ["drift_correction", "energy_balance", "long_term_stability"]
TEMPERATURE = 0.3
MAX_TOKENS = 150

VALID_ACTIONS = [
    "suggest_task",
    "insert_break",
    "reschedule_task",
    "reduce_difficulty",
    "prioritize_goal",
    "do_nothing",
]

SYSTEM_PROMPT = textwrap.dedent("""\
You are a productivity coaching AI agent. You observe a user's cognitive state and take actions to optimize their goal alignment while managing energy and preventing burnout.

Available actions (respond with JSON only):
- suggest_task: Suggest the user work on a specific goal. Requires target_goal.
- insert_break: Suggest a break to recover energy and reduce fatigue.
- reschedule_task: Reorganize the user's schedule for better efficiency.
- reduce_difficulty: Simplify the current task to reduce fatigue.
- prioritize_goal: Focus the user's attention on a specific goal. Requires target_goal.
- do_nothing: Take no action and let the user continue.

Strategy guidelines:
- If drift_score > 0.5, the user is off-track — suggest tasks or prioritize goals.
- If fatigue > 0.6, the user is burning out — insert breaks or reduce difficulty.
- If energy_level < 0.4, the user needs recovery — insert breaks.
- If goal_alignment_score < 0.5, focus on alignment — suggest tasks or prioritize goals.
- Balance between pushing for goals and preventing burnout.

Respond with ONLY a JSON object:
{"action_type": "<action>", "target_goal": "<goal_or_null>"}
""")


# --- Mandatory stdout logging ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# --- Helpers ---

def format_observation(obs: dict) -> str:
    return textwrap.dedent(f"""\
Current State (Step {obs['step_number']}/{obs['max_steps']}):
- Task: {obs['task_description']}
- Goals: {', '.join(obs['goals'])}
- Goal Alignment: {obs['goal_alignment_score']:.2f}
- Drift Score: {obs['drift_score']:.2f}
- Energy Level: {obs['energy_level']:.2f}
- Fatigue: {obs['fatigue']:.2f}
- Focus Score: {obs['focus_score']:.2f}
- Time of Day: {obs['time_of_day']}
- Recent Actions: {', '.join(obs['recent_actions'])}
- Reward: {obs.get('reward', 0):.4f}

What action should be taken? Respond with JSON only.""")


def parse_llm_response(response_text: str, goals: list) -> dict:
    try:
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        action = json.loads(text)
        action_type = action.get("action_type", "do_nothing")
        if action_type not in VALID_ACTIONS:
            action_type = "do_nothing"

        target_goal = action.get("target_goal")
        if target_goal and target_goal not in goals:
            target_goal = goals[0] if goals else None

        return {"action_type": action_type, "target_goal": target_goal}

    except (json.JSONDecodeError, KeyError, IndexError):
        text_lower = response_text.lower()
        if "break" in text_lower:
            return {"action_type": "insert_break", "target_goal": None}
        elif "suggest" in text_lower or "task" in text_lower:
            return {"action_type": "suggest_task", "target_goal": goals[0] if goals else None}
        elif "prioritize" in text_lower:
            return {"action_type": "prioritize_goal", "target_goal": goals[0] if goals else None}
        elif "reschedule" in text_lower:
            return {"action_type": "reschedule_task", "target_goal": None}
        elif "reduce" in text_lower or "difficulty" in text_lower:
            return {"action_type": "reduce_difficulty", "target_goal": None}
        else:
            return {"action_type": "do_nothing", "target_goal": None}


def get_llm_action(client: OpenAI, obs: dict) -> dict:
    user_msg = format_observation(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        llm_text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        llm_text = '{"action_type": "do_nothing"}'

    return parse_llm_response(llm_text, obs["goals"])


def run_task(client: OpenAI, task_id: str, seed: int = 42) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        episode_id = obs["metadata"]["episode_id"]

        while not obs["done"]:
            action = get_llm_action(client, obs)
            action_str = action["action_type"]

            # Step environment
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"episode_id": episode_id, "action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()

            reward = obs.get("reward", 0.0) or 0.0
            done = obs.get("done", False)
            error = obs.get("last_action_error")
            steps_taken += 1
            rewards.append(reward)

            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get final grade
        grade_resp = requests.post(
            f"{ENV_URL}/grade",
            json={"episode_id": episode_id},
            timeout=30,
        )
        grade_resp.raise_for_status()
        grade = grade_resp.json()
        score = max(0.0, min(1.0, grade["score"]))
        success = score > 0.0

        # Cleanup
        try:
            requests.post(f"{ENV_URL}/cleanup/{episode_id}", timeout=10)
        except Exception:
            pass

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "steps": steps_taken, "rewards": rewards}


def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.", flush=True)
        sys.exit(1)

    # Verify environment is running
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        health.raise_for_status()
    except Exception:
        print(f"ERROR: Cannot connect to environment at {ENV_URL}", flush=True)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    results = []
    for task_id in TASKS:
        result = run_task(client, task_id, seed=42)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\nAverage Score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
