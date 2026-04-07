"""FastAPI server for the Life Drift & Cognitive Load environment."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

from models import LifeDriftAction, LifeDriftObservation
from server.environment import LifeDriftEnvironment

app = create_app(
    env=LifeDriftEnvironment,
    action_cls=LifeDriftAction,
    observation_cls=LifeDriftObservation,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
