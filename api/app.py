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
from fastapi.responses import HTMLResponse
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


_LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OpenInbox</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f5f5f5;
      color: #1a1a1a;
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 48px 16px;
    }
    .card {
      background: #ffffff;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      max-width: 680px;
      width: 100%;
      padding: 40px 44px;
    }
    h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px; }
    .subtitle {
      color: #555;
      font-size: 0.95rem;
      margin-top: 4px;
      margin-bottom: 24px;
    }
    p { line-height: 1.65; color: #333; margin-bottom: 24px; }
    .actions { display: flex; gap: 12px; margin-bottom: 32px; flex-wrap: wrap; }
    .btn {
      display: inline-block;
      padding: 10px 22px;
      border-radius: 5px;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      transition: opacity 0.15s;
    }
    .btn:hover { opacity: 0.82; }
    .btn-primary { background: #1a1a1a; color: #ffffff; }
    .btn-secondary {
      background: transparent;
      color: #1a1a1a;
      border: 1px solid #c0c0c0;
    }
    h2 { font-size: 1rem; font-weight: 600; margin-bottom: 12px; color: #111; }
    ol { padding-left: 20px; }
    ol li { line-height: 1.8; color: #333; font-size: 0.92rem; }
    ol li code {
      background: #f0f0f0;
      border-radius: 3px;
      padding: 1px 6px;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
      font-size: 0.88rem;
    }
    .status {
      margin-top: 32px;
      padding: 10px 14px;
      background: #f0faf0;
      border: 1px solid #b6e0b6;
      border-radius: 5px;
      font-size: 0.88rem;
      color: #2a6e2a;
    }
    .footer {
      margin-top: 24px;
      font-size: 0.82rem;
      color: #888;
      border-top: 1px solid #eee;
      padding-top: 16px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>OpenInbox</h1>
    <p class="subtitle">Stateful enterprise email agent environment</p>

    <p>
      This is an OpenEnv-style reinforcement learning environment where an agent
      handles enterprise email tasks including triage, routing, field extraction,
      SLA deadline handling, escalation, and prompt-injection-aware multi-step
      thread processing. Three tasks are available: easy, medium, and hard.
      All scoring is deterministic and reproducible via a fixed seed.
    </p>

    <div class="actions">
      <a class="btn btn-primary" href="/docs">Open API Docs</a>
      <a class="btn btn-secondary" href="/tasks">View Tasks</a>
    </div>

    <h2>How to evaluate</h2>
    <ol>
      <li>Open <code>/docs</code> to access the interactive API browser.</li>
      <li>Call <code>GET /tasks</code> to see available task IDs and configuration.</li>
      <li>Call <code>POST /reset</code> with a <code>task_id</code> and a <code>seed</code> to start an episode.</li>
      <li>Call <code>POST /step</code> repeatedly with your agent actions until the episode ends.</li>
      <li>Call <code>POST /grader</code> with the episode log to receive a score in [0.0, 1.0].</li>
    </ol>

    <div class="status">
      API is running.
    </div>

    <div class="footer">
      This Space is API-first. The full interactive evaluation flow is available through /docs.
    </div>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Landing page shown on the Hugging Face App tab."""
    return _LANDING_HTML


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
