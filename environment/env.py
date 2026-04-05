"""
OpenInboxEnv - the main environment class.

Implements reset(), step(), and state() as required by the OpenEnv spec.
All state transition logic lives here — there is no separate state machine module.

Thread structure used by this env (from threads.json):
  emails[]          - ordered list of inbound emails the agent processes
  followups{}       - pre-authored team responses keyed by routing decision
  ground_truth{}    - correct classification, routing, fields per step

For task_easy and task_medium, each thread has one email. The follow-up from
the routing decision becomes the next current_email.

For task_hard, threads have multiple emails. The env advances through them in
order by step count. Follow-up team responses are added to thread_history as
internal acknowledgments, but the next current_email comes from the thread's
emails array.

Cascade consequence chain (task_hard only):
  If the agent wrong-routes at step N and the thread has a
  "cascade_{N}_to_{N+2}" follow-up authored, the environment schedules an
  escalation email to appear at step N+2. That step is marked in
  _cascade_steps so reward.py receives cascade_step=True and can apply
  cascade_penalty / correction_bonus.
"""

import json
from pathlib import Path
from typing import Optional

from environment.injection import detect as detect_injection
from environment.models import Action, EmailMessage, Observation, RewardBreakdown
from environment import reward as reward_module


DATA_DIR = Path(__file__).parent / "data"


class OpenInboxEnv:
    """
    One instance per active session. Call reset() before stepping.
    """

    def __init__(self):
        # Load both config files once at construction. They're small and static.
        self._threads: dict = json.loads(
            (DATA_DIR / "threads.json").read_text(encoding="utf-8")
        )
        self._tasks: dict = json.loads(
            (DATA_DIR / "tasks.json").read_text(encoding="utf-8")
        )

        # All of these are set properly in reset(). Listed here for clarity.
        self.task_id: str = ""
        self.thread_id: str = ""
        self.thread: dict = {}
        self.step_count: int = 0
        self.max_steps: int = 0
        self.open_tickets: int = 0
        self.ticket_status: str = "open"
        self.team_queues: dict[str, int] = {}
        self.sla_timers: dict[str, float] = {}
        self.current_email: Optional[EmailMessage] = None
        self.thread_history: list[EmailMessage] = []
        self.prev_action: Optional[Action] = None
        self.episode_log: list[dict] = []
        self.done: bool = False
        self.env_injection_flags: dict = {}

        # Cascade consequence state — only used in task_hard.
        # _cascade_queue: step_index -> EmailMessage to inject at that step
        # _cascade_steps: set of step indices at which a cascade was triggered
        self._cascade_queue: dict[int, EmailMessage] = {}
        self._cascade_steps: set[int] = set()

        # Strategic tradeoff state — only used in task_hard
        # context_locked: agent chose to escalate; thread_history is hidden hereafter
        # tradeoff_risk: agent chose not to escalate at step 2; wrong routing = -0.40
        self.context_locked: bool = False
        self.tradeoff_risk: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str, seed: int = 0) -> Observation:
        """
        Start a new episode for the given task.

        Thread is selected deterministically: thread_ids[seed % len(thread_ids)].
        The same seed always produces the same episode.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(self._tasks)}")

        task_cfg = self._tasks[task_id]
        thread_ids = task_cfg["thread_ids"]
        self.thread_id = thread_ids[seed % len(thread_ids)]

        if self.thread_id not in self._threads:
            raise RuntimeError(
                f"Thread {self.thread_id!r} not found in threads.json. "
                "Run environment/data/merge_threads.py to rebuild threads.json."
            )

        self.task_id = task_id
        self.thread = self._threads[self.thread_id]
        self.max_steps = task_cfg["max_steps"]
        self.step_count = 0
        self.done = False
        self.prev_action = None
        self.episode_log = []
        self.env_injection_flags = {}

        # Reset cascade state
        self._cascade_queue = {}
        self._cascade_steps = set()

        # Reset tradeoff state
        self.context_locked = False
        self.tradeoff_risk = False

        # One ticket per episode, keyed by thread_id
        self.open_tickets = 1
        self.ticket_status = "open"

        self.team_queues = {
            "billing_team": 0,
            "tech_team": 0,
            "hr_team": 0,
            "legal_team": 0,
            "spam_filter": 0,
        }

        # SLA timer only applies to task_medium and task_hard
        sla_start = task_cfg.get("sla_start")
        self.sla_timers = {self.thread_id: sla_start} if sla_start is not None else {}

        # Start with the first email in the thread
        self.current_email = EmailMessage(**self.thread["emails"][0])
        self.thread_history = []

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Apply the agent's action and return the next state.

        Returns:
            observation: What the agent sees next.
            reward: Scalar reward for this step (float, in [-1.0, 1.0]).
            done: True if the episode has ended.
            info: Dict containing reward_breakdown and ticket_status.

        Execution order:
          1.  Guard — raise if already done
          2.  Tick SLA timers
          3.  Check SLA breach — early return if breached
          4.  Update ticket status and team queues
          4b. Schedule cascade if wrong routing on task_hard
          5.  Load pre-authored follow-up email
          6.  Determine next current_email (thread advance or followup)
          7.  Resolve ticket if follow-up is a confirmation
          8.  Run injection detector on current email (informational)
          9.  Compute reward (includes cascade signals)
          10. Append to episode log
          11. Advance current_email — honour cascade queue first
          12. Store prev_action
          13. Increment step_count
          14. Check remaining termination conditions
          15. Return
        """
        # 1. Guard
        if self.done:
            raise RuntimeError(
                "Episode is over. Call reset() to start a new one."
            )

        # Capture the step index before we change anything.
        step_idx = self.step_count

        # Determine whether this step was triggered by a cascade event
        is_cascade_step = step_idx in self._cascade_steps

        # Save the previous action for the repeat-penalty check
        prev = self.prev_action

        # 2. Tick SLA timers down by one step's worth
        sla_decrement = self._tasks[self.task_id].get("sla_decrement")
        if sla_decrement is not None and self.ticket_status not in (
            "resolved",
            "sla_breached",
        ):
            for tid in list(self.sla_timers):
                self.sla_timers[tid] = round(
                    self.sla_timers[tid] - sla_decrement, 2
                )

        # 3. Check for SLA breach — ends the episode immediately
        sla_breach = False
        for tid, remaining in self.sla_timers.items():
            if remaining <= 0.0 and self.ticket_status not in (
                "resolved",
                "sla_breached",
            ):
                self.ticket_status = "sla_breached"
                sla_breach = True
                self.done = True
                break

        if sla_breach:
            gt = self._resolve_step_ground_truth(step_idx)
            breakdown = reward_module.compute(
                action=action,
                prev_action=prev,
                ground_truth=gt,
                current_email_has_injection=self.current_email.has_injection,
                sla_at_risk=self._sla_at_risk(),
                task_id=self.task_id,
                sla_breach=True,
                cascade_step=is_cascade_step,
                corrected_cascade=False,
            )
            self._append_log(action, breakdown, step_idx)
            return (
                self._build_observation(),
                breakdown.total,
                True,
                self._build_info(breakdown),
            )

        # 4. Update ticket status and team queues based on the agent's decision
        if action.escalate:
            self.ticket_status = "escalated"
        else:
            self.ticket_status = "routed"

        if action.route_to in self.team_queues:
            self.team_queues[action.route_to] += 1

        # 5. Load the follow-up email for this routing decision
        followup_raw = self._load_followup(action, step_idx)

        # 6. Decide what the next current_email will be
        new_email = self._advance_email(followup_raw, step_idx)

        # 7. Resolve ticket if the team's response is a confirmation
        if followup_raw.get("is_confirmation", False):
            self.ticket_status = "resolved"
            self.open_tickets = 0
            if (
                action.route_to in self.team_queues
                and self.team_queues[action.route_to] > 0
            ):
                self.team_queues[action.route_to] -= 1

        # 8. Run injection detector on the email the agent just processed.
        # Result is stored only — it does not affect rewards.
        self.env_injection_flags = detect_injection(self.current_email.body)

        # 9. Compute reward for this step (uses the email the agent processed,
        # not the new one)
        gt = self._resolve_step_ground_truth(step_idx)

        # 4b. Cascade scheduling — task_hard only.
        # If the agent routed wrong at step N and the thread has authored a
        # cascade_{N}_to_{N+2} follow-up, schedule the escalation email for
        # step N+2. This happens after normal followup lookup so it does not
        # affect the current step's email chain.
        if self.task_id == "task_hard" and gt.get("route_to"):
            if action.route_to != gt["route_to"]:
                cascade_step_target = step_idx + 2
                cascade_key = f"cascade_{step_idx}_to_{cascade_step_target}"
                followups = self.thread.get("followups", {})
                if (
                    cascade_key in followups
                    and cascade_step_target not in self._cascade_queue
                ):
                    self._cascade_queue[cascade_step_target] = EmailMessage(
                        **followups[cascade_key]
                    )
                    self._cascade_steps.add(cascade_step_target)

        # Cascade correction signal — was this a cascade step, and did the agent
        # route correctly on it?
        corrected_cascade = False
        if is_cascade_step and gt.get("route_to"):
            corrected_cascade = (action.route_to == gt["route_to"])

        breakdown = reward_module.compute(
            action=action,
            prev_action=prev,
            ground_truth=gt,
            current_email_has_injection=self.current_email.has_injection,
            sla_at_risk=self._sla_at_risk(),
            task_id=self.task_id,
            sla_breach=False,
            cascade_step=is_cascade_step,
            corrected_cascade=corrected_cascade,
        )

        # Strategic tradeoff logic (task_hard only, fires from step 2 onward)
        _TRADEOFF_STEP = 2
        if self.task_id == "task_hard":
            if not self.context_locked and action.escalate and step_idx >= _TRADEOFF_STEP:
                # OPTION A: agent chose to escalate — immediate bonus, locks context
                self.context_locked = True
                breakdown.escalation_tradeoff_bonus = 0.30
                breakdown.total = round(
                    max(-1.0, min(1.0, breakdown.total + 0.30)), 4
                )
            elif (
                not self.context_locked
                and not self.tradeoff_risk
                and not action.escalate
                and step_idx == _TRADEOFF_STEP
            ):
                # OPTION B: agent chose not to escalate at the tradeoff step
                self.tradeoff_risk = True

            # Apply SLA-risk penalty for wrong routing under tradeoff_risk
            if self.tradeoff_risk and step_idx > _TRADEOFF_STEP and gt.get("route_to"):
                if action.route_to != gt["route_to"]:
                    breakdown.sla_risk_penalty = -0.40
                    breakdown.total = round(
                        max(-1.0, min(1.0, breakdown.total - 0.40)), 4
                    )

        # 10. Record this step
        self._append_log(action, breakdown, step_idx)

        # 11. Advance to the new email — honour cascade queue first
        next_step_idx = step_idx + 1
        if self.task_id == "task_hard" and next_step_idx in self._cascade_queue:
            # Cascade email replaces the normal next email.
            # Append the normal next email to history so context is preserved.
            self.thread_history.append(new_email)
            self.current_email = self._cascade_queue.pop(next_step_idx)
        else:
            self.current_email = new_email

        # 12. Store action for next step's repeat check
        self.prev_action = action

        # 13. Increment step count
        self.step_count += 1

        # 14. Check remaining termination conditions
        if not self.done:
            if self.step_count >= self.max_steps:
                self.done = True
            elif self.open_tickets == 0:
                self.done = True

        # 15. Return
        return (
            self._build_observation(),
            breakdown.total,
            self.done,
            self._build_info(breakdown),
        )

    def state(self) -> dict:
        """
        Return a full snapshot of the environment's internal state.

        Useful for the /state API endpoint and for debugging.
        """
        return {
            "task_id": self.task_id,
            "thread_id": self.thread_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "open_tickets": self.open_tickets,
            "ticket_status": self.ticket_status,
            "team_queues": dict(self.team_queues),
            "sla_timers": dict(self.sla_timers),
            "sla_at_risk": self._sla_at_risk(),
            "done": self.done,
            "env_injection_flags": self.env_injection_flags,
            "episode_log": self.episode_log,
            "cascade_steps_triggered": sorted(self._cascade_steps),
            "context_locked": self.context_locked,
            "tradeoff_risk": self.tradeoff_risk,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        # When context is locked the agent can no longer see thread history.
        # This is the consequence of choosing Option A (escalate for safety).
        visible_history = [] if self.context_locked else list(self.thread_history)
        return Observation(
            current_email=self.current_email,
            thread_history=visible_history,
            open_tickets=self.open_tickets,
            team_queues=dict(self.team_queues),
            sla_timers=dict(self.sla_timers),
            step=self.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id,
            flags={
                "sla_at_risk": self._sla_at_risk(),
                # has_injection reflects the NEW current_email (already swapped)
                "injection_in_current_email": self.current_email.has_injection,
                "context_locked": self.context_locked,
                "tradeoff_risk": self.tradeoff_risk,
            },
        )

    def _sla_at_risk(self) -> bool:
        """True when any open SLA timer is at or below the risk threshold."""
        threshold = self._tasks[self.task_id].get("sla_at_risk_threshold")
        if threshold is None or not self.sla_timers:
            return False
        return any(v <= threshold for v in self.sla_timers.values())

    def _load_followup(self, action: Action, step_idx: int) -> dict:
        """
        Look up the pre-authored follow-up for this routing decision.

        For task_hard, follow-ups are indexed by step number and team name:
          "step2_legal_team" -> the legal team's response at step 2
          "step2_default"    -> fallback if the specific team key is missing
        For task_easy and task_medium, the key is just the team name:
          "billing_team", "tech_team", etc.
        Falls back to "unknown" for any unrecognized route.
        """
        followups = self.thread["followups"]

        if self.task_id == "task_hard":
            key = f"step{step_idx}_{action.route_to}"
            if key not in followups:
                key = f"step{step_idx}_default"
            if key not in followups:
                key = "unknown"
        else:
            key = action.route_to
            if key not in followups:
                key = "unknown"

        if key not in followups:
            raise RuntimeError(
                f"Follow-up key {key!r} not found in thread {self.thread_id!r}. "
                "The 'unknown' fallback must always be present in the dataset."
            )

        return followups[key]

    def _advance_email(self, followup_raw: dict, step_idx: int) -> EmailMessage:
        """
        Update thread_history and return the next email the agent will see.

        For task_hard with remaining emails in the thread:
          - Add the current email to history
          - Add the team's follow-up to history as an internal note
          - Return the next sender email from the thread's emails array

        For task_easy, task_medium, or task_hard after all thread emails
        are exhausted:
          - Add the current email to history
          - The team's follow-up becomes the next current_email

        Note: cascade injection is handled in step() after this call returns,
        by checking _cascade_queue at step N+1.
        """
        # Current email is now part of history
        self.thread_history.append(self.current_email)

        thread_emails = self.thread["emails"]
        next_step_index = step_idx + 1

        if self.task_id == "task_hard" and next_step_index < len(thread_emails):
            # Multi-email thread: add team response to history, advance sender thread
            self.thread_history.append(EmailMessage(**followup_raw))
            return EmailMessage(**thread_emails[next_step_index])
        else:
            # Single-email thread or exhausted thread: follow-up becomes next email
            return EmailMessage(**followup_raw)

    def _resolve_step_ground_truth(self, step_idx: int) -> dict:
        """
        Return a flat ground-truth dict for the given step index.

        task_hard stores lists (one value per thread step), so we extract
        the element at step_idx. task_easy and task_medium store scalars
        directly and are returned as-is.
        """
        gt = self.thread["ground_truth"]

        if self.task_id == "task_hard":
            # Clamp to last index if the agent actions beyond the authored steps
            idx = min(step_idx, len(gt["classifications"]) - 1)
            return {
                "classification": gt["classifications"][idx],
                "route_to": gt["routes"][idx],
                "extracted_fields": gt.get("extracted_fields", {}),
                # Priority is not scored in task_hard
                "priority": "",
                "requires_escalation": gt["requires_escalation"],
            }

        return gt

    def _append_log(
        self, action: Action, breakdown: RewardBreakdown, step_idx: int
    ) -> None:
        """Add a record of this step to the episode log."""
        self.episode_log.append(
            {
                "step": step_idx,
                "thread_id": self.thread_id,
                "action": action.model_dump(),
                "reward_breakdown": breakdown.model_dump(),
                "cascade_triggered": step_idx in self._cascade_steps,
            }
        )

    def _build_info(self, breakdown: RewardBreakdown) -> dict:
        return {
            "reward_breakdown": breakdown.model_dump(),
            "ticket_status": self.ticket_status,
        }
