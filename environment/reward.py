"""
Per-step reward computation for OpenInbox.

Called once per step from env.py. All weights are fixed — no randomness.
The total is clamped to [-1.0, 1.0].

=== LOCKED PER-STEP COMPONENTS (Phase 1D — 4 signals only) ===
  budget_penalty        -0.40   escalate action cost (always applied when escalating)
  sla_urgency           +0.10   high/critical priority when SLA is at risk
  cascade_penalty       -0.20   wrong routing on a cascade-triggered step (task_hard only)
  repeat_penalty        -0.10   identical action as previous step

=== LOGGED ONLY — NOT SUMMED INTO total (computed for grading/debug) ===
  escalation_penalty    -0.15   escalated when not warranted
  injection_reward      +0.20   flagged injection correctly  (task_hard only)
  injection_penalty     -0.20   missed injection             (task_hard only)
  false_positive_penalty -0.05  flagged injection when none  (task_hard only)
  sla_breach_penalty    -0.30   terminal step when SLA breach occurs
  drift_routing_penalty -0.10   routing to a reliability-degraded team
  wait_bonus            +0.15   strategic wait when ≥1 team is degraded and SLA not at risk
  correction_bonus      +0.15   correct routing on a cascade-triggered step (task_hard only)

=== BANNED FROM PER-STEP TOTAL (removed in Phase 1D) ===
  classification_reward  +0.25  logged for grader debugging, NOT in total
  routing_reward         +0.20  logged for grader debugging, NOT in total
  extraction_reward      +0.20  logged for grader debugging, NOT in total
  priority_reward        +0.15  logged for grader debugging, NOT in total

Note: priority is not scored in task_hard (ground_truth has empty priority string).

=== INTERNAL ACTION→TEAM MAPPING ===
  handle_self      → spam_filter   (exact GT match: spam_filter)
  delegate_fast    → billing_team / tech_team  (GT match: either)
  delegate_thorough → legal_team / hr_team     (GT match: either)
  escalate         → requires_escalation==True
  wait             → never a correct route (no routing credit)
"""

import re
from typing import Optional

from environment.models import Action, RewardBreakdown


# Priority levels in order from lowest to highest.
_PRIORITY_ORDER = ["low", "medium", "high", "critical"]

# Abstract-action → set of internal GT team names that count as correct routing
_FAST_TEAMS = {"billing_team", "tech_team"}
_THOROUGH_TEAMS = {"legal_team", "hr_team", "compliance_team"}
_SELF_TEAMS = {"spam_filter"}


def compute(
    action: Action,
    prev_action: Optional[Action],
    ground_truth: dict,
    current_email_has_injection: bool,
    sla_at_risk: bool,
    task_id: str,
    sla_breach: bool = False,
    cascade_step: bool = False,
    corrected_cascade: bool = False,
    any_team_degraded: bool = False,
) -> RewardBreakdown:
    """
    Compute the reward for one step.

    ground_truth is already resolved to a per-step dict by env.py,
    so this function never touches list-type ground truth directly.

    any_team_degraded: True when ≥1 team's reliability has drifted below threshold.
                       Used to determine wait_bonus eligibility.
    """
    r = RewardBreakdown()

    gt_classification = ground_truth.get("classification", "")
    gt_route = ground_truth.get("route_to", "")
    gt_fields = ground_truth.get("extracted_fields", {})
    gt_priority = ground_truth.get("priority", "")
    gt_requires_escalation = ground_truth.get("requires_escalation", False)

    # Normalise escalate: route_to=="escalate" is treated as escalating regardless of bool
    is_escalating = (action.route_to == "escalate") or action.escalate

    # -----------------------------------------------------------------------
    # PER-STEP components (these ARE added to total)
    # -----------------------------------------------------------------------

    # Budget penalty — escalate costs 0.40 regardless of correctness
    if is_escalating:
        r.budget_penalty = -0.40

    # SLA urgency bonus — agent correctly identified that this needs urgent attention
    if sla_at_risk and action.priority in {"high", "critical"}:
        r.sla_urgency = 0.10

    # Unnecessary escalation (stacks with budget_penalty for a double disincentive)
    if is_escalating and not gt_requires_escalation:
        r.escalation_penalty = -0.15

    # Injection signals — task_hard only
    if task_id == "task_hard":
        if current_email_has_injection:
            if action.flag_injection:
                r.injection_reward = 0.20
            else:
                r.injection_penalty = -0.20
        else:
            if action.flag_injection:
                r.false_positive_penalty = -0.05

    # Repeat action — penalise the agent for not changing anything
    if prev_action is not None and _actions_identical(action, prev_action):
        r.repeat_penalty = -0.10

    # Terminal SLA breach penalty (only added by env on the breach step)
    if sla_breach:
        r.sla_breach_penalty = -0.30

    # Cascade consequence signals — task_hard only
    if task_id == "task_hard" and cascade_step:
        if corrected_cascade:
            r.correction_bonus = 0.15
        else:
            r.cascade_penalty = -0.20

    # Reliability drift penalty — routing to a team whose reliability has degraded
    # Only applies to actual delegation/routing actions, not wait or escalate
    if any_team_degraded and action.route_to in {"delegate_fast", "delegate_thorough", "handle_self"}:
        r.drift_routing_penalty = -0.10

    # Wait bonus — strategic wait when ≥1 team degraded and time permits
    if action.route_to == "wait" and any_team_degraded and not sla_at_risk:
        r.wait_bonus = 0.15

    # -----------------------------------------------------------------------
    # Sum ONLY the four permitted per-step components (Phase 1D locked spec).
    # All other signals above are computed and stored for logging/grading only.
    # -----------------------------------------------------------------------
    per_step_components = [
        r.budget_penalty,
        r.sla_urgency,
        r.cascade_penalty,
        r.repeat_penalty,
    ]
    r.total = round(max(-1.0, min(1.0, sum(per_step_components))), 4)
    return r


def terminal_outcome_reward(
    action: Action,
    ground_truth: dict,
    ticket_status: str,
    budget_remaining: float,
) -> float:
    """
    Fires ONLY when a task is fully resolved (end of episode).
    """
    total = 0.0

    # Base resolution outcome
    if ticket_status == "resolved":
        total += 1.0
        total += 0.30  # SLA met at resolution
    else:
        total -= 1.0

    # Classification correctness
    gt_classification = ground_truth.get("classification", "")
    if gt_classification and action.classification == gt_classification:
        total += 0.20

    # Extraction correctness
    gt_fields = ground_truth.get("extracted_fields", {})
    if gt_fields:
        total += round(token_f1(action.extracted_fields, gt_fields) * 0.15, 4)

    # Priority correctness
    gt_priority = ground_truth.get("priority", "")
    if gt_priority and action.priority == gt_priority:
        total += 0.10

    # Budget conservation
    if budget_remaining > 0.20:
        total += 0.20

    return round(total, 4)


def token_f1(predicted: dict[str, str], ground: dict[str, str]) -> float:
    """
    Token-level F1 score between predicted and ground-truth field dicts.

    For each field in ground truth, computes F1 between the predicted value's
    tokens and the ground-truth value's tokens. Returns the mean across all fields.
    Fields the agent left empty count as zero recall.
    """
    if not ground:
        # No fields to extract — agent correctly extracted nothing → full score.
        return 1.0 if not predicted else 0.0

    scores = []
    for field, gt_value in ground.items():
        gt_tokens = _tokenize(gt_value)
        pred_tokens = _tokenize(predicted.get(field, ""))

        if not gt_tokens and not pred_tokens:
            scores.append(1.0)
            continue
        if not gt_tokens or not pred_tokens:
            scores.append(0.0)
            continue

        common = set(pred_tokens) & set(gt_tokens)
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)

        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters."""
    return re.findall(r"\w+", text.lower())


def _priority_score(predicted: str, ground: str) -> float:
    """
    +0.15 for exact match, +0.05 if off by exactly one priority level, else 0.
    Adjacent levels: low<medium<high<critical.
    """
    if predicted == ground:
        return 0.15
    try:
        pi = _PRIORITY_ORDER.index(predicted)
        gi = _PRIORITY_ORDER.index(ground)
        if abs(pi - gi) == 1:
            return 0.05
    except ValueError:
        pass
    return 0.0


def _actions_identical(a: Action, b: Action) -> bool:
    """True if every meaningful field in both actions is the same."""
    return (
        a.classification == b.classification
        and a.priority == b.priority
        and a.route_to == b.route_to
        and a.extracted_fields == b.extracted_fields
        and a.escalate == b.escalate
        and a.flag_injection == b.flag_injection
    )
