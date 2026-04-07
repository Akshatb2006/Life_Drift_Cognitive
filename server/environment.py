"""Core environment logic for Life Drift & Cognitive Load."""

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from models import LifeDriftAction, LifeDriftObservation, LifeDriftState


# --- Task Definitions ---

TASKS = {
    "drift_correction": {
        "id": "drift_correction",
        "description": "The user is wasting time and drifting from their goals. Bring alignment above 0.7 and drift below 0.3 within 10 steps.",
        "difficulty": "easy",
        "max_steps": 10,
        "initial_state": {
            "goal_alignment_score": 0.2,
            "energy_level": 0.7,
            "fatigue": 0.3,
            "focus_score": 0.3,
            "drift_score": 0.8,
            "recent_actions": ["scrolled social media", "watched random videos", "browsed news"],
            "goals": ["fitness", "study", "side_project"],
        },
    },
    "energy_balance": {
        "id": "energy_balance",
        "description": "The user is productive but burning out. Maintain alignment above 0.6 while keeping fatigue below 0.5 over 15 steps.",
        "difficulty": "medium",
        "max_steps": 15,
        "initial_state": {
            "goal_alignment_score": 0.8,
            "energy_level": 0.3,
            "fatigue": 0.8,
            "focus_score": 0.6,
            "drift_score": 0.2,
            "recent_actions": ["coded for 4 hours", "skipped lunch", "worked through break"],
            "goals": ["fitness", "study", "side_project"],
        },
    },
    "long_term_stability": {
        "id": "long_term_stability",
        "description": "The user has mixed behavior patterns. Optimize for both low drift and stable energy over 20 steps with no burnout spikes.",
        "difficulty": "hard",
        "max_steps": 20,
        "initial_state": {
            "goal_alignment_score": 0.5,
            "energy_level": 0.5,
            "fatigue": 0.5,
            "focus_score": 0.5,
            "drift_score": 0.5,
            "recent_actions": ["worked a bit", "took a long break", "started task then stopped"],
            "goals": ["fitness", "study", "side_project", "networking"],
        },
    },
}

# Simulated user action responses for each agent action type
USER_RESPONSES = {
    "suggest_task": [
        "worked on {goal} for 30 min",
        "started {goal} task but got distracted",
        "completed a {goal} session",
        "partially followed {goal} suggestion",
    ],
    "insert_break": [
        "took a 10-minute walk",
        "did a short meditation",
        "rested for 15 minutes",
        "had a snack break",
    ],
    "reschedule_task": [
        "reorganized schedule",
        "moved hard tasks to morning",
        "batched similar tasks together",
    ],
    "reduce_difficulty": [
        "switched to easier subtask",
        "broke task into smaller pieces",
        "simplified the approach",
    ],
    "prioritize_goal": [
        "focused attention on {goal}",
        "set {goal} as top priority",
        "eliminated distractions for {goal}",
    ],
    "do_nothing": [
        "continued current activity",
        "kept doing what they were doing",
        "no change in behavior",
    ],
}

TIME_PERIODS = ["morning", "late_morning", "afternoon", "late_afternoon", "evening"]


class LifeDriftEnvironment(Environment[LifeDriftAction, LifeDriftObservation, LifeDriftState]):
    """Simulates a user's productivity and cognitive state."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._session: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LifeDriftObservation:
        """Reset environment to initial state for a given task."""
        if seed is not None:
            random.seed(seed)

        task_id = kwargs.get("task_id", "drift_correction")
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")

        task = TASKS[task_id]
        init = task["initial_state"]
        eid = episode_id or str(uuid.uuid4())

        self._session = {
            "episode_id": eid,
            "task_id": task_id,
            "step_count": 0,
            "max_steps": task["max_steps"],
            "goal_alignment_score": init["goal_alignment_score"],
            "energy_level": init["energy_level"],
            "fatigue": init["fatigue"],
            "focus_score": init["focus_score"],
            "drift_score": init["drift_score"],
            "goals": list(init["goals"]),
            "recent_actions": list(init["recent_actions"]),
            "time_index": 0,
            # Trajectory tracking for grading
            "alignment_history": [init["goal_alignment_score"]],
            "fatigue_history": [init["fatigue"]],
            "drift_history": [init["drift_score"]],
            "energy_history": [init["energy_level"]],
            "reward_history": [],
            "done": False,
        }

        return self._make_observation(reward=0.0, done=False)

    def step(
        self,
        action: LifeDriftAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LifeDriftObservation:
        """Execute one step in the environment."""
        session = self._session
        if session is None:
            raise ValueError("Environment not reset. Call reset() first.")

        if session["done"]:
            return self._make_observation(reward=0.0, done=True)

        # Save previous state for reward computation
        prev_alignment = session["goal_alignment_score"]
        prev_drift = session["drift_score"]
        prev_fatigue = session["fatigue"]

        # Apply action effects
        self._apply_action(action)

        # Advance time
        session["step_count"] += 1
        session["time_index"] = min(
            session["step_count"] * len(TIME_PERIODS) // session["max_steps"],
            len(TIME_PERIODS) - 1,
        )

        # Add natural drift/fatigue accumulation
        self._apply_natural_dynamics()

        # Clamp all values
        self._clamp_values()

        # Record history
        session["alignment_history"].append(session["goal_alignment_score"])
        session["fatigue_history"].append(session["fatigue"])
        session["drift_history"].append(session["drift_score"])
        session["energy_history"].append(session["energy_level"])

        # Compute reward
        reward = self._compute_reward(prev_alignment, prev_drift, prev_fatigue)
        session["reward_history"].append(reward)

        # Check episode end
        done = session["step_count"] >= session["max_steps"]
        session["done"] = done

        return self._make_observation(reward=reward, done=done)

    @property
    def state(self) -> LifeDriftState:
        """Return current state metadata."""
        if self._session is None:
            return LifeDriftState()
        session = self._session
        score = self._grade()
        return LifeDriftState(
            episode_id=session["episode_id"],
            step_count=session["step_count"],
            task_id=session["task_id"],
            score=score,
        )

    def grade(self) -> float:
        """Compute final grader score for the episode."""
        return self._grade()

    def _grade(self) -> float:
        """Internal grading logic per task."""
        session = self._session
        if session is None:
            return 0.0

        task_id = session["task_id"]

        if task_id == "drift_correction":
            final_align = session["goal_alignment_score"]
            final_drift = session["drift_score"]
            align_score = min(final_align / 0.7, 1.0)
            drift_score = min((1.0 - final_drift) / 0.7, 1.0)
            return round(0.6 * align_score + 0.4 * drift_score, 4)

        elif task_id == "energy_balance":
            avg_align = sum(session["alignment_history"]) / len(session["alignment_history"])
            final_fatigue = session["fatigue"]
            avg_fatigue = sum(session["fatigue_history"]) / len(session["fatigue_history"])
            align_score = min(avg_align / 0.6, 1.0)
            fatigue_score = max(0, 1.0 - avg_fatigue)
            final_fatigue_score = max(0, 1.0 - final_fatigue)
            return round(
                0.4 * align_score + 0.3 * fatigue_score + 0.3 * final_fatigue_score,
                4,
            )

        elif task_id == "long_term_stability":
            avg_drift = sum(session["drift_history"]) / len(session["drift_history"])
            avg_energy = sum(session["energy_history"]) / len(session["energy_history"])
            max_fatigue = max(session["fatigue_history"])
            avg_reward = (
                sum(session["reward_history"]) / len(session["reward_history"])
                if session["reward_history"]
                else 0
            )
            energy_var = sum(
                (e - avg_energy) ** 2 for e in session["energy_history"]
            ) / len(session["energy_history"])
            stability_score = max(0, 1.0 - energy_var * 4)

            drift_component = max(0, 1.0 - avg_drift)
            burnout_penalty = max(0, 1.0 - max_fatigue)
            reward_component = max(0, min(1.0, (avg_reward + 1) / 2))

            return round(
                0.3 * drift_component
                + 0.25 * stability_score
                + 0.25 * burnout_penalty
                + 0.2 * reward_component,
                4,
            )

        return 0.0

    def _apply_action(self, action: LifeDriftAction):
        """Apply the agent's action to the environment state."""
        session = self._session
        action_type = action.action_type
        target = action.target_goal or (session["goals"][0] if session["goals"] else "general")

        if target not in session["goals"]:
            target = session["goals"][0] if session["goals"] else "general"

        templates = USER_RESPONSES.get(action_type, USER_RESPONSES["do_nothing"])
        response = random.choice(templates).format(goal=target)
        session["recent_actions"] = session["recent_actions"][-2:] + [response]

        if action_type == "suggest_task":
            effectiveness = random.uniform(0.08, 0.18)
            session["goal_alignment_score"] += effectiveness
            session["drift_score"] -= effectiveness * 0.9
            session["energy_level"] -= random.uniform(0.05, 0.12)
            session["fatigue"] += random.uniform(0.05, 0.12)
            session["focus_score"] += random.uniform(0.02, 0.08)

        elif action_type == "insert_break":
            recovery = random.uniform(0.1, 0.2)
            session["energy_level"] += recovery
            session["fatigue"] -= recovery * 1.1
            session["drift_score"] += random.uniform(0.02, 0.06)
            session["focus_score"] += random.uniform(0.05, 0.1)
            session["goal_alignment_score"] -= random.uniform(0.01, 0.04)

        elif action_type == "reschedule_task":
            session["goal_alignment_score"] += random.uniform(0.05, 0.1)
            session["drift_score"] -= random.uniform(0.04, 0.08)
            session["energy_level"] -= random.uniform(0.02, 0.05)
            session["focus_score"] += random.uniform(0.03, 0.07)

        elif action_type == "reduce_difficulty":
            session["fatigue"] -= random.uniform(0.03, 0.07)
            session["energy_level"] += random.uniform(0.02, 0.05)
            session["goal_alignment_score"] += random.uniform(0.03, 0.08)
            session["drift_score"] -= random.uniform(0.02, 0.05)

        elif action_type == "prioritize_goal":
            session["goal_alignment_score"] += random.uniform(0.1, 0.2)
            session["drift_score"] -= random.uniform(0.08, 0.15)
            session["energy_level"] -= random.uniform(0.08, 0.15)
            session["fatigue"] += random.uniform(0.06, 0.1)
            session["focus_score"] += random.uniform(0.05, 0.1)

        elif action_type == "do_nothing":
            session["drift_score"] += random.uniform(0.05, 0.1)
            session["goal_alignment_score"] -= random.uniform(0.03, 0.07)
            session["energy_level"] += random.uniform(0.01, 0.03)
            session["fatigue"] -= random.uniform(0.01, 0.03)
            session["focus_score"] -= random.uniform(0.03, 0.06)

    def _apply_natural_dynamics(self):
        """Apply natural time-based dynamics."""
        session = self._session
        session["fatigue"] += random.uniform(0.01, 0.03)
        session["drift_score"] += random.uniform(0.01, 0.02)
        session["energy_level"] -= random.uniform(0.01, 0.03)

        time_period = TIME_PERIODS[session["time_index"]]
        if time_period == "morning":
            session["focus_score"] += 0.02
        elif time_period in ("late_afternoon", "evening"):
            session["fatigue"] += 0.02
            session["focus_score"] -= 0.02

        if session["fatigue"] > 0.85:
            session["focus_score"] -= 0.05
            session["goal_alignment_score"] -= 0.03
            session["drift_score"] += 0.03

    def _clamp_values(self):
        """Clamp all values to [0, 1]."""
        session = self._session
        for key in [
            "goal_alignment_score",
            "energy_level",
            "fatigue",
            "focus_score",
            "drift_score",
        ]:
            session[key] = max(0.0, min(1.0, session[key]))

    def _compute_reward(
        self,
        prev_alignment: float,
        prev_drift: float,
        prev_fatigue: float,
    ) -> float:
        """Compute step reward with dense signal."""
        session = self._session
        alignment_delta = session["goal_alignment_score"] - prev_alignment
        drift_delta = session["drift_score"] - prev_drift
        fatigue_delta = session["fatigue"] - prev_fatigue

        r_alignment = 0.4 * alignment_delta
        r_drift = -0.3 * drift_delta

        energy = session["energy_level"]
        if 0.4 <= energy <= 0.7:
            r_energy = 0.1
        else:
            r_energy = -0.05

        r_fatigue = -0.2 * max(0, fatigue_delta)

        state_bonus = 0.0
        if session["goal_alignment_score"] > 0.7 and session["drift_score"] < 0.3:
            state_bonus = 0.1
        if session["fatigue"] < 0.4 and session["energy_level"] > 0.5:
            state_bonus += 0.05

        if session["fatigue"] > 0.9:
            state_bonus -= 0.15
        if session["drift_score"] > 0.8:
            state_bonus -= 0.1

        reward = r_alignment + r_drift + r_energy + r_fatigue + state_bonus
        return round(max(-1.0, min(1.0, reward)), 4)

    def _make_observation(self, reward: float, done: bool) -> LifeDriftObservation:
        """Create observation from session state."""
        session = self._session
        task = TASKS[session["task_id"]]
        time_period = TIME_PERIODS[session["time_index"]]

        # Include grade in metadata when episode is done
        meta = {"episode_id": session["episode_id"]}
        if done:
            meta["score"] = self._grade()

        return LifeDriftObservation(
            done=done,
            reward=reward,
            metadata=meta,
            goals=session["goals"],
            recent_actions=session["recent_actions"],
            goal_alignment_score=round(session["goal_alignment_score"], 4),
            energy_level=round(session["energy_level"], 4),
            fatigue=round(session["fatigue"], 4),
            focus_score=round(session["focus_score"], 4),
            drift_score=round(session["drift_score"], 4),
            time_of_day=time_period,
            step_number=session["step_count"],
            max_steps=session["max_steps"],
            task_id=session["task_id"],
            task_description=task["description"],
        )
