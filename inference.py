"""Baseline inference script for the Life Drift & Cognitive Load environment.

Uses an OpenAI-compatible LLM to act as a productivity coaching agent.
Reads API credentials from environment variables.
"""

import json
import os
import sys
import time
import requests
from openai import OpenAI

# --- Configuration ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

VALID_ACTIONS = [
    "suggest_task",
    "insert_break",
    "reschedule_task",
    "reduce_difficulty",
    "prioritize_goal",
    "do_nothing",
]

SYSTEM_PROMPT = """You are a productivity coaching AI agent. You observe a user's cognitive state and take actions to optimize their goal alignment while managing energy and preventing burnout.

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
"""


def format_observation(obs: dict) -> str:
    """Format observation as a readable prompt for the LLM."""
    return f"""Current State (Step {obs['step_number']}/{obs['max_steps']}):
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

What action should be taken? Respond with JSON only."""


def parse_llm_response(response_text: str, goals: list) -> dict:
    """Parse LLM response into a valid action."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()
        # Handle markdown code blocks
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
        # Fallback: try to detect action from text
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


def run_task(client: OpenAI, task_id: str, seed: int = 42) -> dict:
    """Run a single task and return results."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()
    episode_id = obs["metadata"]["episode_id"]

    print(f"Episode ID: {episode_id}")
    print(f"Task: {obs['task_description']}")
    print(f"Initial state: align={obs['goal_alignment_score']:.2f}, drift={obs['drift_score']:.2f}, "
          f"energy={obs['energy_level']:.2f}, fatigue={obs['fatigue']:.2f}")

    total_reward = 0.0
    step = 0

    while not obs["done"]:
        # Get LLM action
        user_msg = format_observation(obs)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            llm_text = response.choices[0].message.content
        except Exception as e:
            print(f"  LLM error at step {step}: {e}")
            llm_text = '{"action_type": "do_nothing"}'

        action = parse_llm_response(llm_text, obs["goals"])
        target = action['target_goal']
        suffix = f" ({target})" if target else ""
        print(f"  Step {step}: action={action['action_type']}{suffix}")

        # Step environment
        step_resp = requests.post(
            f"{ENV_URL}/step",
            json={"episode_id": episode_id, "action": action},
        )
        step_resp.raise_for_status()
        obs = step_resp.json()

        total_reward += obs.get("reward", 0)
        step += 1

        print(f"         align={obs['goal_alignment_score']:.2f}, drift={obs['drift_score']:.2f}, "
              f"energy={obs['energy_level']:.2f}, fatigue={obs['fatigue']:.2f}, "
              f"reward={obs.get('reward', 0):.4f}")

    # Get final grade
    grade_resp = requests.post(
        f"{ENV_URL}/grade",
        json={"episode_id": episode_id},
    )
    grade_resp.raise_for_status()
    grade = grade_resp.json()

    # Cleanup
    requests.post(f"{ENV_URL}/cleanup/{episode_id}")

    result = {
        "task_id": task_id,
        "score": grade["score"],
        "total_reward": round(total_reward, 4),
        "steps": step,
    }

    print(f"\nResult: score={grade['score']:.4f}, total_reward={total_reward:.4f}, steps={step}")
    return result


def main():
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY or HF_TOKEN environment variable.")
        sys.exit(1)

    print(f"API Base: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}")

    # Verify environment is running
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=5)
        health.raise_for_status()
        print("Environment health check: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to environment at {ENV_URL}")
        print("Start the environment first: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

    tasks = ["drift_correction", "energy_balance", "long_term_stability"]
    results = []

    for task_id in tasks:
        result = run_task(client, task_id, seed=42)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Score':>8} {'Reward':>10} {'Steps':>6}")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*6}")
    for r in results:
        print(f"{r['task_id']:<25} {r['score']:>8.4f} {r['total_reward']:>10.4f} {r['steps']:>6}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.4f}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "average_score": avg_score}, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
