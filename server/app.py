"""FastAPI server for the Life Drift & Cognitive Load environment."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import LifeDriftAction, LifeDriftObservation, LifeDriftState
from server.environment import LifeDriftEnvironment, TASKS

app = FastAPI(
    title="Life Drift & Cognitive Load Environment",
    description="An OpenEnv environment that tracks goal alignment and cognitive load to help AI agents learn productivity coaching.",
    version="0.1.0",
)

env = LifeDriftEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "drift_correction"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    episode_id: str
    action: LifeDriftAction


class GradeRequest(BaseModel):
    episode_id: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def root():
    return {
        "name": "life_drift_cognitive_env",
        "description": "Life Drift & Cognitive Load - AI Productivity Coach Environment",
        "tasks": list(TASKS.keys()),
    }


@app.post("/reset", response_model=LifeDriftObservation)
def reset(request: ResetRequest):
    try:
        obs = env.reset(
            task_id=request.task_id,
            seed=request.seed,
            episode_id=request.episode_id,
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=LifeDriftObservation)
def step(request: StepRequest):
    try:
        obs = env.step(request.episode_id, request.action)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state/{episode_id}", response_model=LifeDriftState)
def state(episode_id: str):
    try:
        return env.state(episode_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade")
def grade(request: GradeRequest):
    try:
        score = env.grade(request.episode_id)
        return {"score": score, "episode_id": request.episode_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "id": task["id"],
            "description": task["description"],
            "difficulty": task["difficulty"],
            "max_steps": task["max_steps"],
        }
        for task_id, task in TASKS.items()
    }


@app.post("/cleanup/{episode_id}")
def cleanup(episode_id: str):
    env.cleanup_session(episode_id)
    return {"status": "cleaned up", "episode_id": episode_id}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
