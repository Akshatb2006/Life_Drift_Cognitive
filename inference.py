"""
Inference Script for Life Drift & Cognitive Load Environment
===================================
Uses an OpenAI-compatible LLM to act as a productivity coaching agent.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import LifeDriftEnv
from models import LifeDriftAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("LIFE_DRIFT_TASK", "drift_correction")
BENCHMARK = os.getenv("LIFE_DRIFT_BENCHMARK", "life_drift_cognitive_env")
MAX_STEPS = 20
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
You are a productivity coaching AI agent. You observe a user's cognitive state \
and take actions to optimize their goal alignment while managing energy and \
preventing burnout.

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


def format_observation(obs) -> str:
    return textwrap.dedent(f"""\
Current State (Step {obs.step_number}/{obs.max_steps}):
- Task: {obs.task_description}
- Goals: {', '.join(obs.goals)}
- Goal Alignment: {obs.goal_alignment_score:.2f}
- Drift Score: {obs.drift_score:.2f}
- Energy Level: {obs.energy_level:.2f}
- Fatigue: {obs.fatigue:.2f}
- Focus Score: {obs.focus_score:.2f}
- Time of Day: {obs.time_of_day}
- Recent Actions: {', '.join(obs.recent_actions)}

What action should be taken? Respond with JSON only.""")


def parse_llm_response(response_text: str, goals: list) -> LifeDriftAction:
    try:
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        action_type = parsed.get("action_type", "do_nothing")
        if action_type not in VALID_ACTIONS:
            action_type = "do_nothing"

        target_goal = parsed.get("target_goal")
        if target_goal and target_goal not in goals:
            target_goal = goals[0] if goals else None

        return LifeDriftAction(action_type=action_type, target_goal=target_goal)

    except (json.JSONDecodeError, KeyError, IndexError):
        text_lower = response_text.lower()
        if "break" in text_lower:
            return LifeDriftAction(action_type="insert_break", target_goal=None)
        elif "suggest" in text_lower or "task" in text_lower:
            return LifeDriftAction(action_type="suggest_task", target_goal=goals[0] if goals else None)
        elif "prioritize" in text_lower:
            return LifeDriftAction(action_type="prioritize_goal", target_goal=goals[0] if goals else None)
        elif "reschedule" in text_lower:
            return LifeDriftAction(action_type="reschedule_task", target_goal=None)
        elif "reduce" in text_lower or "difficulty" in text_lower:
            return LifeDriftAction(action_type="reduce_difficulty", target_goal=None)
        else:
            return LifeDriftAction(action_type="do_nothing", target_goal=None)


def get_model_action(client: OpenAI, obs, history: List[str]) -> LifeDriftAction:
    user_prompt = format_observation(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            text = '{"action_type": "do_nothing"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        text = '{"action_type": "do_nothing"}'

    return parse_llm_response(text, obs.goals)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await LifeDriftEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=TASK_NAME)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, obs, history)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f}")

            if done:
                break

        # Get score from metadata if available, otherwise compute from rewards
        score = obs.metadata.get("score", 0.0) if obs.metadata else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
