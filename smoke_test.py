"""
Quick smoke test for the three core environment files.

Runs one full episode for each task using a known-good action and checks:
  - reset() returns a valid Observation
  - step() returns (Observation, float, bool, dict)
  - state() returns the expected keys
  - reward total is in [-1.0, 1.0]
  - episode terminates cleanly

Run with: python smoke_test.py
"""

from environment.env import OpenInboxEnv
from environment.models import Action


def make_action(**kwargs) -> Action:
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


def run_task_easy():
    print("--- task_easy ---")
    env = OpenInboxEnv()
    obs = env.reset("task_easy", seed=0)
    print(f"  reset ok | email: {obs.current_email.subject!r}")
    print(f"  step=0 task_id={obs.task_id} max_steps={obs.max_steps}")

    action = make_action(
        classification="billing",
        priority="medium",
        route_to="billing_team",
        extracted_fields={"invoice_number": "4821", "amount": "3200", "submitted_date": "2024-03-10"},
    )
    obs2, reward, done, info = env.step(action)
    print(f"  step ok | reward={reward} done={done}")
    print(f"  ticket_status={info['ticket_status']}")
    print(f"  reward_breakdown={info['reward_breakdown']}")
    assert -1.0 <= reward <= 1.0, "reward out of range"
    print(f"  next email: {obs2.current_email.subject!r}")
    s = env.state()
    assert "episode_log" in s
    print(f"  state ok | step_count={s['step_count']}")
    print()


def run_task_medium():
    print("--- task_medium ---")
    env = OpenInboxEnv()
    obs = env.reset("task_medium", seed=0)
    print(f"  reset ok | email: {obs.current_email.subject!r}")
    print(f"  sla_timers={obs.sla_timers} flags={obs.flags}")

    # Correct action for thread_medium_001: legal, high priority
    action = make_action(
        classification="legal",
        priority="high",
        route_to="legal_team",
        extracted_fields={
            "contract_id": "SCT-2024-0047",
            "expiry_date": "2024-03-16",
            "vendor_name": "VendorPeak Solutions",
        },
    )
    obs2, reward, done, info = env.step(action)
    print(f"  step ok | reward={reward} done={done}")
    print(f"  ticket_status={info['ticket_status']}")
    assert -1.0 <= reward <= 1.0
    print()


def run_task_hard():
    print("--- task_hard ---")
    env = OpenInboxEnv()
    obs = env.reset("task_hard", seed=0)
    print(f"  reset ok | email: {obs.current_email.subject!r}")

    # Step through all 4 emails in thread_hard_001
    actions = [
        make_action(classification="billing", priority="medium", route_to="billing_team"),
        make_action(classification="billing", priority="medium", route_to="billing_team"),
        make_action(classification="legal", priority="high", route_to="legal_team",
                    flag_injection=True, escalate=True),
        make_action(classification="legal", priority="high", route_to="legal_team",
                    escalate=True),
    ]

    for i, act in enumerate(actions):
        if env.done:
            break
        obs, reward, done, info = env.step(act)
        print(f"  step {i} | reward={reward:.4f} done={done} "
              f"ticket={info['ticket_status']} "
              f"sla_timers={env.sla_timers}")
        assert -1.0 <= reward <= 1.0

    s = env.state()
    print(f"  final state: steps={s['step_count']} ticket={s['ticket_status']}")
    print(f"  episode_log entries: {len(s['episode_log'])}")
    print()


def run_sla_breach():
    print("--- task_medium SLA breach simulation ---")
    env = OpenInboxEnv()
    env.reset("task_medium", seed=0)
    # Do nothing useful for 5 steps — SLA at 8h, -2h/step → breach at step 4
    noop = make_action(classification="unknown", priority="low", route_to="spam_filter")
    for i in range(6):
        if env.done:
            print(f"  episode ended at step {i} (expected)")
            break
        obs, reward, done, info = env.step(noop)
        print(f"  step {i} | reward={reward:.4f} done={done} sla={env.sla_timers}")
    print()


if __name__ == "__main__":
    run_task_easy()
    run_task_medium()
    run_task_hard()
    run_sla_breach()
    print("All smoke tests passed.")
