"""
Grader for task_easy (single email triage).

Scores the agent's action at step 0 only. The email is clear and
unambiguous, so this is a baseline check of whether the agent can
classify, prioritise, route, and extract fields correctly.

Weight table (weights sum to 1.00):
  classification  0.30  exact match required
  priority        0.20  exact match required
  routing         0.20  exact match required
  extraction      0.30  token F1 across all ground-truth fields
"""

from environment.graders.base import exact_match, token_f1


def grade(episode_log: list[dict], ground_truth: dict) -> dict:
    """
    Score a task_easy episode.

    Args:
        episode_log: List of step dicts from env.state()["episode_log"].
                     Must have at least one entry.
        ground_truth: The thread's ground_truth dict from threads.json.

    Returns:
        {"score": float in [0.0, 1.0], "breakdown": dict of weighted components}
    """
    # Validate inputs
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

    # Score each component
    c_score = exact_match(action["classification"], ground_truth["classification"])
    p_score = exact_match(action["priority"], ground_truth["priority"])
    r_score = exact_match(action["route_to"], ground_truth["route_to"])
    e_score = token_f1(
        action.get("extracted_fields", {}),
        ground_truth.get("extracted_fields", {}),
    )

    # Weighted sum — 0.30 + 0.20 + 0.20 + 0.30 = 1.00
    score = (
        0.30 * c_score +
        0.20 * p_score +
        0.20 * r_score +
        0.30 * e_score
    )

    return {
        "score": round(score, 4),
        "breakdown": {
            "classification": round(0.30 * c_score, 4),
            "priority":       round(0.20 * p_score, 4),
            "routing":        round(0.20 * r_score, 4),
            "extraction":     round(0.30 * e_score, 4),
        },
    }


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to verify scores on known inputs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gt = {
        "classification": "billing",
        "priority": "medium",
        "route_to": "billing_team",
        "extracted_fields": {
            "invoice_number": "4821",
            "amount": "3200",
            "submitted_date": "2024-03-10",
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

    # Perfect score — all fields correct
    perfect_log = [{"step": 0, "action": _action(
        "billing", "medium", "billing_team",
        {"invoice_number": "4821", "amount": "3200", "submitted_date": "2024-03-10"},
    )}]
    result = grade(perfect_log, gt)
    print(f"Perfect score: {result['score']}  (expected 1.0)")
    print(f"  breakdown: {result['breakdown']}")

    # Wrong on everything
    zero_log = [{"step": 0, "action": _action("spam", "low", "spam_filter")}]
    result = grade(zero_log, gt)
    print(f"Zero score:    {result['score']}  (expected 0.0)")
    print(f"  breakdown: {result['breakdown']}")

    # Partial — correct classification and route, wrong priority, partial extraction
    partial_log = [{"step": 0, "action": _action(
        "billing", "low", "billing_team",
        {"invoice_number": "4821"},  # missing amount and submitted_date
    )}]
    result = grade(partial_log, gt)
    print(f"Partial score: {result['score']}  (expected ~0.70)")
    print(f"  breakdown: {result['breakdown']}")
