"""
Core data models for the OpenInbox environment.

All environment methods and API endpoints exchange these types.
Using Pydantic v2 for validation and serialization.
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


class Action(BaseModel):
    """The agent's decision for the current email."""

    classification: Literal["billing", "technical", "hr", "legal", "spam", "unknown"]
    priority: Literal["low", "medium", "high", "critical"]
    route_to: Literal[
        "billing_team", "tech_team", "hr_team", "legal_team", "spam_filter"
    ]
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    escalate: bool = False
    flag_injection: bool = False
    reply_draft: Optional[str] = None


class RewardBreakdown(BaseModel):
    """Individual reward components for one step. Total is clamped to [-1.0, 1.0]."""

    classification_reward: float = 0.0
    routing_reward: float = 0.0
    extraction_reward: float = 0.0
    priority_reward: float = 0.0
    sla_bonus: float = 0.0
    escalation_penalty: float = 0.0
    injection_reward: float = 0.0
    injection_penalty: float = 0.0
    false_positive_penalty: float = 0.0
    repeat_penalty: float = 0.0
    sla_breach_penalty: float = 0.0
    # Cascade consequence fields — default 0.0, only non-zero in task_hard
    cascade_penalty: float = 0.0
    correction_bonus: float = 0.0
    # Strategic tradeoff fields — default 0.0, only non-zero in task_hard
    escalation_tradeoff_bonus: float = 0.0
    sla_risk_penalty: float = 0.0
    total: float = 0.0

