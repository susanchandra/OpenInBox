"""
Grader for task_medium (SLA-sensitive routing).

Scores the agent's action at step 0. The key difference from task_easy
is that the email carries an SLA deadline, so the agent gets scored on
whether it recognised the urgency, not just whether it got priority exact.

Priority scoring gives partial credit for being one level off, which
rewards the agent for understanding the situation is urgent even if it
doesn't pick the exact right label.

Weight table (weights sum to 1.00):
  classification  0.25  exact match
  routing         0.20  exact match
  extraction      0.20  token F1
  priority        0.20  full credit for exact, half credit if one level off
  sla_urgency     0.15  did the agent pick high or critical at all?

Note: priority and sla_urgency can both contribute (max combined = 0.35
when the agent gets priority exactly right on a high/critical email).
They measure different things — correctness vs urgency awareness.
"""

from environment.graders.base import exact_match, token_f1

_PRIORITY_ORDER = ["low", "medium", "high", "critical"]


def grade(episode_log: list[dict], ground_truth: dict) -> dict:
    """
    Score a task_medium episode.

    Args:
        episode_log: List of step dicts from env.state()["episode_log"].
        ground_truth: The thread's ground_truth dict from threads.json.

    Returns:
        {"score": float in [0.0, 1.0], "breakdown": dict of weighted components}
    """
    if not episode_log:
        return {"score": 0.0, "breakdown": {"error": "episode_log is empty"}}

    required_gt = {"classification", "priority", "route_to", "extracted_fields"}
    missing = required_gt - ground_truth.keys()
    if missing:
        return {"score": 0.0, "breakdown": {"error": f"ground_truth missing keys: {missing}"}}

    action = episode_log[0]["action"]

    required_action = {"classification", "priority", "route_to", "extracted_fields"}
    missing_action = required_action - action.keys()
    if missing_action:
        return {"score": 0.0, "breakdown": {"error": f"action missing keys: {missing_action}"}}

    # Classification and routing — exact match only
    c_score = exact_match(action["classification"], ground_truth["classification"])
    r_score = exact_match(action["route_to"], ground_truth["route_to"])

    # Field extraction — token F1
    e_score = token_f1(
        action.get("extracted_fields", {}),
        ground_truth.get("extracted_fields", {}),
    )

    # Priority — full credit for exact, half credit for adjacent level
    gt_priority = ground_truth["priority"]
    ag_priority = action["priority"]
    if ag_priority == gt_priority:
        priority_partial = 1.0
    else:
        try:
            gi = _PRIORITY_ORDER.index(gt_priority)
            ai = _PRIORITY_ORDER.index(ag_priority)
            priority_partial = 0.5 if abs(gi - ai) == 1 else 0.0
        except ValueError:
            priority_partial = 0.0

    # SLA urgency awareness — did the agent at least recognise this needs urgent attention?
    sla_urgency = 1.0 if ag_priority in {"high", "critical"} else 0.0

    # Weighted sum — 0.25 + 0.20 + 0.20 + 0.20 + 0.15 = 1.00
    score = (
        0.25 * c_score +
        0.20 * r_score +
        0.20 * e_score +
        0.20 * priority_partial +
        0.15 * sla_urgency
    )

    return {
        "score": round(score, 4),
        "breakdown": {
            "classification":  round(0.25 * c_score, 4),
            "routing":         round(0.20 * r_score, 4),
            "extraction":      round(0.20 * e_score, 4),
            "priority_partial": round(0.20 * priority_partial, 4),
            "sla_urgency":     round(0.15 * sla_urgency, 4),
        },
    }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # thread_medium_001: legal, high, legal_team, with contract fields
    gt = {
        "classification": "legal",
        "priority": "high",
        "route_to": "legal_team",
        "extracted_fields": {
            "contract_id": "SCT-2024-0047",
            "expiry_date": "2024-03-16",
            "vendor_name": "VendorPeak Solutions",
        },
        "requires_escalation": False,
    }

    def _action(classification, priority, route_to, extracted_fields=None):
        return {
            "classification": classification,
            "priority": priority,
            "route_to": route_to,
            "extracted_fields": extracted_fields or {},
            "escalate": False,
            "flag_injection": False,
            "reply_draft": None,
        }

    # Perfect
    perfect_log = [{"step": 0, "action": _action(
        "legal", "high", "legal_team",
        {"contract_id": "SCT-2024-0047", "expiry_date": "2024-03-16",
         "vendor_name": "VendorPeak Solutions"},
    )}]
    r = grade(perfect_log, gt)
    print(f"Perfect score: {r['score']}  (expected 1.0)")
    print(f"  breakdown: {r['breakdown']}")

    # Zero
    zero_log = [{"step": 0, "action": _action("spam", "low", "spam_filter")}]
    r = grade(zero_log, gt)
    print(f"Zero score:    {r['score']}  (expected 0.0)")

    # Agent got priority wrong by one level (critical instead of high)
    # — should get partial priority credit and full SLA urgency credit
    close_log = [{"step": 0, "action": _action(
        "legal", "critical", "legal_team",
        {"contract_id": "SCT-2024-0047"},
    )}]
    r = grade(close_log, gt)
    print(f"Off-by-one priority: {r['score']}  (expected ~0.815)")
    print(f"  breakdown: {r['breakdown']}")

    # Correct class and route but missed urgency entirely
    low_pri_log = [{"step": 0, "action": _action(
        "legal", "low", "legal_team",
        {"contract_id": "SCT-2024-0047", "expiry_date": "2024-03-16",
         "vendor_name": "VendorPeak Solutions"},
    )}]
    r = grade(low_pri_log, gt)
    print(f"Missed urgency: {r['score']}  (expected 0.65, no priority or SLA credit)")
    print(f"  breakdown: {r['breakdown']}")
