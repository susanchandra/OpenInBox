"""
FastAPI application for OpenInbox.

Exposes exactly 5 endpoints:
  POST /reset   — start a new episode
  POST /step    — submit an action
  GET  /state   — get full environment state
  GET  /tasks   — list available tasks
  POST /grader  — score a completed episode

Sessions are stored in a module-level dict (one env per session_id).
This is fine for single-worker deployment on Hugging Face Spaces.
"""

import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment.env import OpenInboxEnv
from environment.models import Action
from environment.graders import GRADERS


DATA_DIR = Path(__file__).parent.parent / "environment" / "data"

# Load static config files once at startup — they don't change at runtime.
_TASKS: dict = json.loads((DATA_DIR / "tasks.json").read_text(encoding="utf-8"))
_THREADS: dict = json.loads((DATA_DIR / "threads.json").read_text(encoding="utf-8"))

app = FastAPI(
    title="OpenInbox",
    description="Stateful enterprise email agent environment for the OpenEnv spec.",
    version="1.0.0",
)

# In-memory session store. Each session_id maps to one OpenInboxEnv instance.
_sessions: dict[str, OpenInboxEnv] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str
    seed: int = 0


class StepRequest(BaseModel):
    session_id: str
    action: Action


class GraderRequest(BaseModel):
    task_id: str
    episode_log: list[dict]
    # thread_id can be passed explicitly or read from episode_log[0]["thread_id"]
    thread_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> OpenInboxEnv:
    """Look up a session or raise a 404."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset to start a new episode.",
        )
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(req: ResetRequest):
    """
    Start a new episode.

    Creates a fresh environment for the given task and seed, stores it in a
    new session, and returns the initial observation plus session metadata.

    The seed selects which thread is loaded: thread_ids[seed % len(thread_ids)].
    The same task_id + seed always produces the same episode.
    """
    if req.task_id not in _TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: '{req.task_id}'. Valid options: {list(_TASKS)}",
        )

    env = OpenInboxEnv()
    try:
        obs = env.reset(req.task_id, req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session_id = str(uuid4())
    _sessions[session_id] = env

    return {
        "session_id": session_id,
        "thread_id": env.thread_id,
        "task_id": req.task_id,
        "seed": req.seed,
        "max_steps": env.max_steps,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(req: StepRequest):
    """
    Submit one action and advance the environment.

    Returns the next observation, the scalar reward for this step,
    whether the episode has ended, and a breakdown of the reward components.

    Calling /step after the episode is done returns a 400 error.
    """
    env = _get_session(req.session_id)

    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        # env.step raises RuntimeError if the episode has already ended
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    """
    Return the full internal state of the environment for a session.

    Includes step count, ticket status, SLA timers, team queues,
    injection flags, and the full episode log so far.
    Useful for debugging and for building the grader request.
    """
    env = _get_session(session_id)
    return env.state()


@app.get("/tasks")
def tasks():
    """
    List all available tasks with their configuration.

    Returns the task IDs, thread IDs, episode horizons, and SLA parameters.
    Use the task_id values in POST /reset requests.
    """
    result = {}
    for task_id, cfg in _TASKS.items():
        result[task_id] = {
            "thread_ids": cfg["thread_ids"],
            "max_steps": cfg["max_steps"],
            "sla_start": cfg["sla_start"],
            "sla_decrement": cfg["sla_decrement"],
            "thread_count": len(cfg["thread_ids"]),
        }
    return result


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Score a completed episode.

    Requires task_id and episode_log. The thread_id is used to load
    the ground truth from the dataset. If thread_id is not passed in the
    request body, it is read from episode_log[0]["thread_id"] (which env.py
    always sets).

    Returns the overall score in [0.0, 1.0] and a per-component breakdown.
    """
    if req.task_id not in GRADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: '{req.task_id}'. Valid options: {list(GRADERS)}",
        )

    if not req.episode_log:
        raise HTTPException(status_code=400, detail="episode_log is empty.")

    # Resolve thread_id — from request body or from first log entry
    thread_id = req.thread_id
    if not thread_id:
        thread_id = req.episode_log[0].get("thread_id")
    if not thread_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "thread_id is required. Pass it in the request body or ensure "
                "episode_log entries contain a 'thread_id' field."
            ),
        )

    if thread_id not in _THREADS:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{thread_id}' not found in the dataset.",
        )

    ground_truth = _THREADS[thread_id]["ground_truth"]

    try:
        result = GRADERS[req.task_id](req.episode_log, ground_truth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "task_id": req.task_id,
        "thread_id": thread_id,
        "score": result["score"],
        "breakdown": result["breakdown"],
    }
