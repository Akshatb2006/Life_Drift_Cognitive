"""Pydantic models for the Life Drift & Cognitive Load environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class LifeDriftAction(Action):
    """Action the agent can take to guide the user."""

    action_type: str = Field(
        ...,
        description="One of: suggest_task, insert_break, reschedule_task, reduce_difficulty, prioritize_goal, do_nothing",
    )
    target_goal: Optional[str] = Field(
        default=None,
        description="Which goal to target (required for suggest_task, prioritize_goal)",
    )


class LifeDriftObservation(Observation):
    """What the agent observes about the user's state."""

    # Environment-specific fields
    goals: List[str] = Field(default_factory=list, description="User's active goals")
    recent_actions: List[str] = Field(
        default_factory=list, description="Recent user actions"
    )
    goal_alignment_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How aligned recent actions are with goals"
    )
    energy_level: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current energy/motivation level"
    )
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0, description="Accumulated fatigue")
    focus_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current focus/concentration level"
    )
    drift_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How far the user has drifted from goals"
    )
    time_of_day: str = Field(default="morning", description="Current simulated time period")
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=20, description="Maximum steps in the episode")
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Description of the current task")


class LifeDriftState(State):
    """Current environment state metadata."""

    task_id: str = Field(default="")
    score: float = Field(default=0.0, description="Current grader score")
