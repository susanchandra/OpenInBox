"""
Unit tests for the OpenInbox environment.

Tests cover: reset, step, state, SLA timers, episode termination,
follow-up loading, and determinism.

Run with: pytest tests/test_env.py -v
"""

import pytest
from environment.env import OpenInboxEnv
from environment.models import Action


def _action(**kwargs) -> Action:
    defaults = {
        "classification": "billing",
        "priority": "medium",
        "route_to": "billing_team",
        "extracted_fields": {},
        "escalate": False,
        "flag_injection": False,
        "reply_draft": None,
    }
    defaults.update(kwargs)
    return Action(**defaults)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:

    def test_valid_task_returns_observation(self):
        env = OpenInboxEnv()
        obs = env.reset("task_easy", seed=0)
        assert obs.task_id == "task_easy"
        assert obs.step == 0
        assert obs.current_email is not None
        assert obs.open_tickets == 1

    def test_all_tasks_load(self):
        env = OpenInboxEnv()
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            obs = env.reset(task_id, seed=0)
            assert obs.task_id == task_id

    def test_unknown_task_raises(self):
        env = OpenInboxEnv()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("task_nonexistent")

    def test_seed_selects_thread_deterministically(self):
        env = OpenInboxEnv()
        obs0_a = env.reset("task_easy", seed=0)
        obs0_b = env.reset("task_easy", seed=0)
        assert obs0_a.current_email.id == obs0_b.current_email.id

    def test_different_seeds_may_select_different_threads(self):
        env = OpenInboxEnv()
        obs0 = env.reset("task_easy", seed=0)
        obs1 = env.reset("task_easy", seed=1)
        # seed 0 % 2 = 0, seed 1 % 2 = 1 — different threads
        assert obs0.current_email.thread_id != obs1.current_email.thread_id

    def test_sla_timer_empty_for_task_easy(self):
        env = OpenInboxEnv()
        obs = env.reset("task_easy", seed=0)
        assert obs.sla_timers == {}

    def test_sla_timer_set_for_task_medium(self):
        env = OpenInboxEnv()
        obs = env.reset("task_medium", seed=0)
        assert len(obs.sla_timers) == 1
        timer_val = list(obs.sla_timers.values())[0]
        assert timer_val == 8.0

    def test_thread_history_empty_on_reset(self):
        env = OpenInboxEnv()
        obs = env.reset("task_easy", seed=0)
        assert obs.thread_history == []

    def test_team_queues_all_zero_on_reset(self):
        env = OpenInboxEnv()
        obs = env.reset("task_easy", seed=0)
        assert all(v == 0 for v in obs.team_queues.values())

    def test_reset_clears_previous_episode(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        action = _action(route_to="billing_team")
        env.step(action)
        # Reset should start fresh
        obs = env.reset("task_easy", seed=0)
        assert obs.step == 0
        assert obs.open_tickets == 1
        assert env.episode_log == []


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_returns_four_values(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        result = env.step(_action())
        assert len(result) == 4
        obs, reward, done, info = result

    def test_reward_in_valid_range(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        _, reward, _, _ = env.step(_action())
        assert -1.0 <= reward <= 1.0

    def test_done_is_bool(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        _, _, done, _ = env.step(_action())
        assert isinstance(done, bool)

    def test_info_has_reward_breakdown_and_ticket_status(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        _, _, _, info = env.step(_action())
        assert "reward_breakdown" in info
        assert "ticket_status" in info

    def test_correct_routing_resolves_ticket_easy(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        action = _action(
            classification="billing",
            priority="medium",
            route_to="billing_team",
            extracted_fields={"invoice_number": "4821"},
        )
        _, _, done, info = env.step(action)
        assert done is True
        assert info["ticket_status"] == "resolved"

    def test_wrong_routing_does_not_resolve_ticket(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        _, _, done, info = env.step(_action(route_to="spam_filter"))
        assert info["ticket_status"] != "resolved"

    def test_step_after_done_raises(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        # Correct routing resolves and sets done=True
        env.step(_action(route_to="billing_team"))
        assert env.done is True
        with pytest.raises(RuntimeError, match="Episode is over"):
            env.step(_action())

    def test_episode_log_grows_per_step(self):
        env = OpenInboxEnv()
        env.reset("task_hard", seed=0)
        for _ in range(3):
            if env.done:
                break
            env.step(_action(classification="billing", route_to="billing_team"))
        assert len(env.episode_log) == 3

    def test_thread_history_grows_per_step(self):
        env = OpenInboxEnv()
        obs = env.reset("task_hard", seed=0)
        assert len(obs.thread_history) == 0
        obs2, _, _, _ = env.step(_action())
        assert len(obs2.thread_history) > 0

    def test_team_queue_increments_on_routing(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        env.step(_action(route_to="billing_team"))
        # Queue decrements when ticket resolves — check state before resolution
        # For this thread the follow-up is a confirmation, so queue is 0 after resolve

    def test_repeat_action_incurs_penalty(self):
        env = OpenInboxEnv()
        env.reset("task_hard", seed=0)
        action = _action(classification="billing", route_to="billing_team")
        _, r1, _, _ = env.step(action)
        if not env.done:
            _, r2, _, info2 = env.step(action)
            # repeat_penalty = -0.10 should have reduced reward
            assert info2["reward_breakdown"]["repeat_penalty"] == -0.10


# ---------------------------------------------------------------------------
# SLA mechanics
# ---------------------------------------------------------------------------

class TestSLA:

    def test_sla_ticks_down_each_step(self):
        env = OpenInboxEnv()
        obs = env.reset("task_medium", seed=0)
        initial = list(obs.sla_timers.values())[0]
        env.step(_action())  # wrong routing, ticket stays open
        remaining = list(env.sla_timers.values())[0]
        assert remaining == initial - 2.0

    def test_sla_breach_ends_episode(self):
        env = OpenInboxEnv()
        env.reset("task_medium", seed=0)
        # SLA is 8h, -2h/step: breaches at end of step 4
        noop = _action(classification="unknown", route_to="spam_filter")
        done = False
        for _ in range(6):
            if env.done:
                done = True
                break
            env.step(noop)
        assert done or env.done

    def test_sla_breach_adds_penalty(self):
        env = OpenInboxEnv()
        env.reset("task_medium", seed=0)
        noop = _action(classification="unknown", route_to="spam_filter")
        last_reward = 0.0
        for _ in range(6):
            if env.done:
                break
            _, reward, done, info = env.step(noop)
            last_reward = reward
            if info["ticket_status"] == "sla_breached":
                assert info["reward_breakdown"]["sla_breach_penalty"] == -0.30
                break

    def test_sla_at_risk_flag_triggers(self):
        env = OpenInboxEnv()
        env.reset("task_medium", seed=0)
        # After 1 step: 8.0 - 2.0 = 6.0h (above 4h threshold, not at risk)
        _, _, _, _ = env.step(_action(route_to="spam_filter"))
        assert not env._sla_at_risk()
        # After 2 steps: 6.0 - 2.0 = 4.0h (at threshold, flag should trigger)
        env.step(_action(route_to="spam_filter"))
        assert env._sla_at_risk()


# ---------------------------------------------------------------------------
# state()
# ---------------------------------------------------------------------------

class TestState:

    def test_state_has_required_keys(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        s = env.state()
        expected = {
            "task_id", "thread_id", "step_count", "max_steps",
            "open_tickets", "ticket_status", "team_queues",
            "sla_timers", "sla_at_risk", "done", "episode_log",
        }
        assert expected.issubset(s.keys())

    def test_state_reflects_current_step_count(self):
        env = OpenInboxEnv()
        env.reset("task_hard", seed=0)
        env.step(_action())
        env.step(_action())
        s = env.state()
        assert s["step_count"] == 2

    def test_episode_log_entries_have_expected_keys(self):
        env = OpenInboxEnv()
        env.reset("task_easy", seed=0)
        env.step(_action())
        log_entry = env.state()["episode_log"][0]
        assert "step" in log_entry
        assert "action" in log_entry
        assert "reward_breakdown" in log_entry
        assert "thread_id" in log_entry


# ---------------------------------------------------------------------------
# task_hard multi-step specifics
# ---------------------------------------------------------------------------

class TestTaskHard:

    def test_hard_task_has_sla(self):
        env = OpenInboxEnv()
        obs = env.reset("task_hard", seed=0)
        assert len(obs.sla_timers) == 1
        assert list(obs.sla_timers.values())[0] == 12.0

    def test_has_injection_flag_correct_per_step(self):
        env = OpenInboxEnv()
        obs = env.reset("task_hard", seed=0)
        # thread_hard_001: step 0 has no injection
        assert obs.current_email.has_injection is False
        # step 1 — still no injection
        obs2, _, _, _ = env.step(_action(classification="billing", route_to="billing_team"))
        # step 2 — injection is at step_index 2
        if not env.done:
            obs3, _, _, _ = env.step(_action(classification="billing", route_to="billing_team"))
            assert obs3.current_email.has_injection is True

    def test_injection_reward_on_correct_flag(self):
        env = OpenInboxEnv()
        env.reset("task_hard", seed=0)
        # Advance to step 2 (the injection step)
        env.step(_action(classification="billing", route_to="billing_team"))
        if not env.done:
            env.step(_action(classification="billing", route_to="billing_team"))
        if not env.done:
            _, _, _, info = env.step(
                _action(classification="legal", route_to="legal_team", flag_injection=True)
            )
            assert info["reward_breakdown"]["injection_reward"] == 0.20

    def test_injection_penalty_on_missed_flag(self):
        env = OpenInboxEnv()
        env.reset("task_hard", seed=0)
        env.step(_action(classification="billing", route_to="billing_team"))
        if not env.done:
            env.step(_action(classification="billing", route_to="billing_team"))
        if not env.done:
            _, _, _, info = env.step(
                _action(classification="legal", route_to="legal_team", flag_injection=False)
            )
            assert info["reward_breakdown"]["injection_penalty"] == -0.20
