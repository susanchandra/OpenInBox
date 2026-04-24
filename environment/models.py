"""
Core data models for the OpenInbox environment.

All environment methods and API endpoints exchange these types.
Using Pydantic v2 for validation and serialization.

Action space (v2 — abstract 5-action):
  handle_self      — agent handles the email directly (no team delegation)
  delegate_fast    — route to a fast team (billing/tech); lower accuracy, cheaper
  delegate_thorough — route to a thorough team (legal/HR); slower, more accurate
  escalate         — escalate to senior management; costs 0.40 of budget
  wait             — do nothing this step; strategic when teams are degraded
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmailMessage(BaseModel):
    """A single email, either inbound from an external sender or a team response."""

    id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    timestamp: str  # ISO8601
    has_injection: bool  # set in the dataset, never computed at runtime
    step_index: int  # position of this email in the thread sequence


class Observation(BaseModel):
    """Everything the agent sees at the start of each step."""

    current_email: EmailMessage
    thread_history: list[EmailMessage]  # emails seen so far this episode
    open_tickets: int
    team_queues: dict[str, int]  # team name -> number of queued tickets
    sla_timers: dict[str, float]  # ticket_id -> hours remaining (empty for task_easy)
    step: int
    max_steps: int
    task_id: str
    flags: dict[str, bool]  # sla_at_risk, injection_in_current_email
    delegation_history: dict  # track recent outcomes and total uses


class Action(BaseModel):
    """The agent's decision for the current email."""

    classification: Literal["billing", "technical", "hr", "legal", "spam", "unknown"]
    priority: Literal["low", "medium", "high", "critical"]
    route_to: Literal[
        "handle_self",       # agent self-handles; maps internally to spam_filter path
        "delegate_fast",     # fast team (billing/tech); lower accuracy
        "delegate_thorough", # thorough team (legal/HR); slower, more accurate
        "escalate",          # escalate; costs 0.40 of budget
        "wait",              # do nothing; strategic when teams are degraded
    ]
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    escalate: bool = False   # kept for backward compat; auto-set True when route_to=="escalate"
    flag_injection: bool = False
    reply_draft: Optional[str] = None


class RewardBreakdown(BaseModel):
    """Individual reward components for one step. Total is clamped to [-1.0, 1.0]."""

    # --- Per-step reward components (these DO sum into total) ---
    sla_urgency: float = 0.0            # urgency bonus: high/critical when SLA at risk
    escalation_penalty: float = 0.0     # unnecessary escalation
    budget_penalty: float = 0.0         # cost of escalate action (-0.40)
    injection_reward: float = 0.0       # correct injection flag (task_hard)
    injection_penalty: float = 0.0      # missed injection (task_hard)
    false_positive_penalty: float = 0.0 # false injection flag (task_hard)
    repeat_penalty: float = 0.0         # identical action repeated
    sla_breach_penalty: float = 0.0     # terminal SLA breach
    drift_routing_penalty: float = 0.0  # routing to a degraded team
    wait_bonus: float = 0.0             # strategic wait when teams degraded
    # Cascade consequence fields — default 0.0, only non-zero in task_hard
    cascade_penalty: float = 0.0
    correction_bonus: float = 0.0
    # Strategic tradeoff fields — default 0.0, only non-zero in task_hard
    escalation_tradeoff_bonus: float = 0.0
    sla_risk_penalty: float = 0.0
    total: float = 0.0

