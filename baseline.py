"""
Baseline measurement — does this env actually separate agent quality?

Runs three policies against all three tasks across multiple seeds and reports
mean/std of grader scores per (policy, task). Interpretation:

    GOOD env  : Random << Rule-based < LLM       (clear gaps)
    WEAK env  : Random ~ Rule-based ~ LLM        (no discrimination)
    BROKEN env: all policies score ~uniformly high or low

The env is imported and driven in-process, so this runs in seconds with no
network or Docker. The --llm flag adds the inference.py LLM agent, which
requires HF_TOKEN / API_KEY env vars and is slow.

Usage:
    python baseline.py                 # random + rule-based, 10 seeds
    python baseline.py --seeds 20      # more seeds
    python baseline.py --llm           # include LLM policy
"""

import argparse
import os
import random
import statistics
from typing import Callable, Dict, List

from models import LifeDriftAction, LifeDriftObservation
from server.environment import LifeDriftEnvironment

TASK_NAMES = ["drift_correction", "energy_balance", "long_term_stability"]

ACTIONS = [
    "suggest_task",
    "insert_break",
    "reschedule_task",
    "reduce_difficulty",
    "prioritize_goal",
    "do_nothing",
]


# --- Policies ---

def random_policy(obs: LifeDriftObservation, rng: random.Random) -> LifeDriftAction:
    action_type = rng.choice(ACTIONS)
    target = rng.choice(obs.goals) if obs.goals else None
    return LifeDriftAction(action_type=action_type, target_goal=target)


def rule_based_policy(obs: LifeDriftObservation, rng: random.Random) -> LifeDriftAction:
    """Deterministic heuristic — a reasonable engineer's first-pass strategy."""
    goal = obs.goals[0] if obs.goals else None

    # Emergency: critical burnout must be treated before anything else
    if obs.fatigue > 0.75:
        return LifeDriftAction(action_type="insert_break", target_goal=None)
    if obs.energy_level < 0.3:
        return LifeDriftAction(action_type="insert_break", target_goal=None)
    # Severe misalignment: use the heaviest alignment tool
    if obs.goal_alignment_score < 0.5 and obs.drift_score > 0.4:
        return LifeDriftAction(action_type="prioritize_goal", target_goal=goal)
    # Moderate drift: push goals
    if obs.drift_score > 0.4:
        return LifeDriftAction(action_type="suggest_task", target_goal=goal)
    # Healthy but tired: simplify to sustain
    if obs.fatigue > 0.55:
        return LifeDriftAction(action_type="reduce_difficulty", target_goal=None)
    # Default: keep pushing the goal
    return LifeDriftAction(action_type="suggest_task", target_goal=goal)


# --- Runner ---

def run_episode(policy_fn: Callable, task_id: str, seed: int) -> float:
    env = LifeDriftEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    # Separate RNG for the policy so it doesn't consume entropy from env's stream
    policy_rng = random.Random(seed * 2654435761 & 0xFFFFFFFF)
    while not obs.done:
        action = policy_fn(obs, policy_rng)
        obs = env.step(action)
    return env.grade()


def summarize(scores: List[float]) -> str:
    if not scores:
        return "n/a"
    m = statistics.mean(scores)
    s = statistics.stdev(scores) if len(scores) > 1 else 0.0
    lo, hi = min(scores), max(scores)
    return f"mean={m:.3f}  std={s:.3f}  range=[{lo:.3f}, {hi:.3f}]"


def separation_gap(a: List[float], b: List[float]) -> float:
    return statistics.mean(b) - statistics.mean(a)


def verdict_label(gap: float) -> str:
    if gap > 0.10:
        return "GOOD"
    if gap > 0.03:
        return "WEAK"
    return "NO SIGNAL"


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--llm", action="store_true", help="Include LLM policy (slow, needs API key)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    policies: Dict[str, Callable] = {
        "Random    ": random_policy,
        "Rule-based": rule_based_policy,
    }

    if args.llm:
        from openai import OpenAI

        from inference import get_model_action

        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY"),
        )

        def llm_policy(obs: LifeDriftObservation, rng: random.Random) -> LifeDriftAction:
            return get_model_action(client, obs, [])

        policies["LLM       "] = llm_policy

    results: Dict[str, Dict[str, List[float]]] = {}

    for task in TASK_NAMES:
        print(f"\nTask: {task}")
        print("-" * 70)
        results[task] = {}
        for name, fn in policies.items():
            try:
                scores = [run_episode(fn, task, seed) for seed in seeds]
                results[task][name.strip()] = scores
                print(f"  {name}  {summarize(scores)}")
            except Exception as exc:
                # Per-policy failure shouldn't erase the other policies' data.
                # Common cause: LLM API failure, rate limits, bad keys.
                print(f"  {name}  FAILED: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("Separation analysis")
    print("=" * 70)
    print("Gap = mean(better policy) − mean(worse policy)")
    print("Verdict: GOOD (> 0.10)  WEAK (> 0.03)  NO SIGNAL (≤ 0.03)")
    print()
    for task in TASK_NAMES:
        r = results[task]
        print(f"  {task}")
        if "Random" in r and "Rule-based" in r:
            gap_rb_random = separation_gap(r["Random"], r["Rule-based"])
            print(f"    Rule-based − Random : {gap_rb_random:+.3f}  [{verdict_label(gap_rb_random)}]")
        if "LLM" in r and "Rule-based" in r:
            gap_llm_rb = separation_gap(r["Rule-based"], r["LLM"])
            print(f"    LLM        − Rule  : {gap_llm_rb:+.3f}  [{verdict_label(gap_llm_rb)}]")
        if "LLM" in r and "Random" in r:
            gap_llm_random = separation_gap(r["Random"], r["LLM"])
            print(f"    LLM        − Random: {gap_llm_random:+.3f}  [{verdict_label(gap_llm_random)}]")


if __name__ == "__main__":
    main()
