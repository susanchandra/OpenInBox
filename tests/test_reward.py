"""
Unit tests for the reward module.

Tests cover: each reward component in isolation, total clamping,
token_f1, priority scoring, and the repeat/injection gates.

Run with: pytest tests/test_reward.py -v
"""

import pytest
from environment.models import Action, RewardBreakdown
from environment.reward import compute, token_f1, _priority_score, _actions_identical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(**kwargs) -> Action:
    defaults = {
        "classification": "billing",
        "priority": "medium",
        "route_to": "delegate_fast",
        "extracted_fields": {},
        "escalate": False,
        "flag_injection": False,
        "reply_draft": None,
    }
    defaults.update(kwargs)
    return Action(**defaults)


def _gt(**kwargs) -> dict:
    defaults = {
        "classification": "billing",
        "priority": "medium",
        "route_to": "billing_team",   # GT still uses legacy team names
        "extracted_fields": {},
        "requires_escalation": False,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:

    def test_exact_match(self):
        assert token_f1({"a": "hello world"}, {"a": "hello world"}) == pytest.approx(1.0)

    def test_no_match(self):
        assert token_f1({"a": "foo"}, {"a": "bar"}) == pytest.approx(0.0)

    def test_partial_match(self):
        score = token_f1({"a": "hello world"}, {"a": "hello there"})
        assert 0.0 < score < 1.0

    def test_empty_ground_truth_returns_zero(self):
        assert token_f1({"a": "something"}, {}) == 0.0

    def test_empty_prediction_returns_zero(self):
        assert token_f1({}, {"a": "something"}) == pytest.approx(0.0)

    def test_multiple_fields_averaged(self):
        pred = {"invoice_number": "4821", "amount": "wrong"}
        gt = {"invoice_number": "4821", "amount": "3200"}
        score = token_f1(pred, gt)
        # First field: perfect (1.0), second: no match (0.0) -> avg 0.5
        assert score == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert token_f1({"a": "HELLO"}, {"a": "hello"}) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

class TestPriorityScore:

    def test_exact_match(self):
        assert _priority_score("medium", "medium") == pytest.approx(0.15)

    def test_adjacent_gives_partial(self):
        assert _priority_score("high", "critical") == pytest.approx(0.05)
        assert _priority_score("medium", "low") == pytest.approx(0.05)

    def test_two_levels_off_gives_zero(self):
        assert _priority_score("low", "high") == 0.0
        assert _priority_score("low", "critical") == 0.0


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

class TestRewardComponents:

    def test_sla_urgency_triggers_correctly(self):
        r = compute(_action(priority="high"), None, _gt(), True, True, "task_easy")
        assert r.sla_urgency == 0.10

    def test_sla_urgency_not_given_for_low_priority(self):
        r = compute(_action(priority="low"), None, _gt(), True, True, "task_easy")
        assert r.sla_urgency == 0.0

    def test_sla_urgency_not_given_when_not_at_risk(self):
        r = compute(_action(priority="critical"), None, _gt(), False, False, "task_easy")
        assert r.sla_urgency == 0.0

    def test_unnecessary_escalation_penalty(self):
        # escalate via route_to: triggers both budget_penalty and escalation_penalty
        r = compute(_action(route_to="escalate"), None,
                    _gt(requires_escalation=False), False, False, "task_easy")
        assert r.escalation_penalty == -0.15
        assert r.budget_penalty == -0.40

    def test_warranted_escalation_no_extra_penalty(self):
        # Warranted: budget_penalty still applies but escalation_penalty does NOT
        r = compute(_action(route_to="escalate"), None,
                    _gt(requires_escalation=True), False, False, "task_easy")
        assert r.escalation_penalty == 0.0
        assert r.budget_penalty == -0.40

    def test_budget_penalty_applies_on_escalate_bool(self):
        # escalate=True (bool) also triggers budget_penalty
        r = compute(_action(route_to="delegate_fast", escalate=True), None,
                    _gt(requires_escalation=False), False, False, "task_easy")
        assert r.budget_penalty == -0.40

    def test_repeat_action_penalty(self):
        action = _action()
        r = compute(action, action, _gt(), False, False, "task_easy")
        assert r.repeat_penalty == -0.10

    def test_different_action_no_repeat_penalty(self):
        a1 = _action(priority="low")
        a2 = _action(priority="high")
        r = compute(a1, a2, _gt(), False, False, "task_easy")
        assert r.repeat_penalty == 0.0

    def test_sla_breach_penalty(self):
        r = compute(_action(), None, _gt(), False, False, "task_easy", sla_breach=True)
        assert r.sla_breach_penalty == -0.30

    def test_no_sla_breach_no_penalty(self):
        r = compute(_action(), None, _gt(), False, False, "task_easy", sla_breach=False)
        assert r.sla_breach_penalty == 0.0

    def test_drift_routing_penalty_when_team_degraded(self):
        # Routing to delegate_fast when any_team_degraded=True → penalty
        r = compute(_action(route_to="delegate_fast"), None, _gt(),
                    False, False, "task_easy", any_team_degraded=True)
        assert r.drift_routing_penalty == -0.10

    def test_no_drift_penalty_when_not_degraded(self):
        r = compute(_action(route_to="delegate_fast"), None, _gt(),
                    False, False, "task_easy", any_team_degraded=False)
        assert r.drift_routing_penalty == 0.0

    def test_wait_bonus_when_degraded_and_sla_safe(self):
        # wait + any_team_degraded=True + sla_at_risk=False → bonus
        r = compute(_action(route_to="wait"), None, _gt(),
                    False, False, "task_easy", any_team_degraded=True)
        assert r.wait_bonus == 0.15

    def test_no_wait_bonus_when_sla_at_risk(self):
        # wait is NOT strategic when SLA is ticking down
        r = compute(_action(route_to="wait"), None, _gt(),
                    True, True, "task_easy", any_team_degraded=True)
        assert r.wait_bonus == 0.0

    def test_no_wait_bonus_when_no_degradation(self):
        r = compute(_action(route_to="wait"), None, _gt(),
                    False, False, "task_easy", any_team_degraded=False)
        assert r.wait_bonus == 0.0


# ---------------------------------------------------------------------------
# Injection components — task_hard only
# ---------------------------------------------------------------------------

class TestInjectionReward:

    def test_injection_reward_on_correct_flag_hard(self):
        r = compute(_action(flag_injection=True), None, _gt(),
                    True, False, "task_hard")
        assert r.injection_reward == 0.20
        assert r.injection_penalty == 0.0

    def test_injection_penalty_on_missed_flag_hard(self):
        r = compute(_action(flag_injection=False), None, _gt(),
                    True, False, "task_hard")
        assert r.injection_penalty == -0.20
        assert r.injection_reward == 0.0

    def test_false_positive_penalty_hard(self):
        r = compute(_action(flag_injection=True), None, _gt(),
                    False, False, "task_hard")
        assert r.false_positive_penalty == -0.05

    def test_no_injection_no_flag_no_penalty_hard(self):
        r = compute(_action(flag_injection=False), None, _gt(),
                    False, False, "task_hard")
        assert r.injection_reward == 0.0
        assert r.injection_penalty == 0.0
        assert r.false_positive_penalty == 0.0

    def test_injection_signals_not_applied_to_easy_task(self):
        # In task_easy, flag_injection has no reward effect
        r = compute(_action(flag_injection=True), None, _gt(),
                    True, False, "task_easy")
        assert r.injection_reward == 0.0
        assert r.injection_penalty == 0.0
        assert r.false_positive_penalty == 0.0


# ---------------------------------------------------------------------------
# Total clamping
# ---------------------------------------------------------------------------

class TestTotalClamping:

    def test_total_clamped_at_positive_one(self):
        # Phase 1D: only 4 signals sum into total.
        # Max positive from allowed signals: sla_urgency (+0.10).
        # injection_reward and correction_bonus are logged-only now.
        action = _action(
            classification="billing",
            priority="high",
            route_to="delegate_fast",
            flag_injection=True,
        )
        gt = _gt(
            classification="billing",
            priority="high",
            route_to="billing_team",
            extracted_fields={},
        )
        r = compute(action, None, gt, True, True, "task_hard",
                    cascade_step=True, corrected_cascade=True)
        assert r.total <= 1.0

    def test_total_clamped_at_negative_one(self):
        # Max out negative: budget_penalty + escalation_penalty + injection_penalty + sla_breach
        action = _action(
            classification="spam",
            route_to="escalate",
            flag_injection=True,
        )
        r = compute(action, action, _gt(requires_escalation=False),
                    False, False, "task_hard", sla_breach=True)
        assert r.total >= -1.0

    def test_total_equals_sum_of_per_step_components(self):
        # Phase 1D: Verify the total is exactly the sum of the 4 locked per-step
        # components only. Other signals (escalation_penalty, injection_reward, etc.)
        # are logged in the breakdown but NOT summed into total.
        action = _action(classification="billing", priority="medium",
                         route_to="delegate_fast")
        r = compute(action, None, _gt(), False, False, "task_easy")
        manual_sum = (
            r.budget_penalty + r.sla_urgency +
            r.cascade_penalty + r.repeat_penalty
        )
        assert r.total == pytest.approx(max(-1.0, min(1.0, manual_sum)), abs=1e-4)


# ---------------------------------------------------------------------------
# actions_identical
# ---------------------------------------------------------------------------

class TestActionsIdentical:
    def test_identical_actions(self):
        a = _action()
        assert _actions_identical(a, a)

    def test_different_classification(self):
        a1 = _action(classification="billing")
        a2 = _action(classification="legal")
        assert not _actions_identical(a1, a2)

    def test_different_priority(self):
        a1 = _action(priority="low")
        a2 = _action(priority="high")
        assert not _actions_identical(a1, a2)

class TestTerminalOutcomeReward:
    def test_resolved_correctly_gives_positive_reward(self):
        from environment.reward import terminal_outcome_reward
        gt = {
            "classification": "billing",
            "priority": "medium",
            "extracted_fields": {"amount": "100"}
        }
        action = _action(classification="billing", priority="medium", route_to="delegate_fast", extracted_fields={"amount": "100"})
        # +1.0 base, +0.30 SLA, +0.20 class, +0.15 extract, +0.10 priority, +0.20 budget = 1.95
        reward = terminal_outcome_reward(action, gt, "resolved", 0.50)
        assert reward == pytest.approx(1.95, abs=1e-3)

    def test_unresolved_gives_negative_base(self):
        from environment.reward import terminal_outcome_reward
        gt = {"classification": "billing"}
        action = _action(classification="spam", priority="low", route_to="wait")
        # -1.0 base, 0 class, 0 priority, 0 extract, +0.20 budget = -0.80
        reward = terminal_outcome_reward(action, gt, "sla_breached", 0.50)
        assert reward == pytest.approx(-0.80, abs=1e-3)

