"""Life Drift & Cognitive Load Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import LifeDriftAction, LifeDriftObservation, LifeDriftState


class LifeDriftEnv(EnvClient[LifeDriftAction, LifeDriftObservation, LifeDriftState]):
    """Client for the Life Drift & Cognitive Load Environment."""

    def _step_payload(self, action: LifeDriftAction) -> Dict:
        return {
            "action_type": action.action_type,
            "target_goal": action.target_goal,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LifeDriftObservation]:
        obs_data = payload.get("observation", {})
        observation = LifeDriftObservation(
            goals=obs_data.get("goals", []),
            recent_actions=obs_data.get("recent_actions", []),
            goal_alignment_score=obs_data.get("goal_alignment_score", 0.0),
            energy_level=obs_data.get("energy_level", 0.0),
            fatigue=obs_data.get("fatigue", 0.0),
            focus_score=obs_data.get("focus_score", 0.0),
            drift_score=obs_data.get("drift_score", 0.0),
            time_of_day=obs_data.get("time_of_day", "morning"),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 20),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> LifeDriftState:
        return LifeDriftState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            score=payload.get("score", 0.0),
        )
