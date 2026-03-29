"""
Debug script for task_hard.

Runs one full episode and prints what happened at each step.
Useful for verifying that persona drift, injection detection, and
escalation all work as expected before running the real grader.

Usage:
    python debug_hard.py          # uses thread_hard_001 (seed=0)
    python debug_hard.py --seed 1 # uses thread_hard_002
"""

import argparse
from environment.env import OpenInboxEnv
from environment.models import Action


def make_action(classification, priority, route_to, escalate=False,
                flag_injection=False, extracted_fields=None) -> Action:
    return Action(
        classification=classification,
        priority=priority,
        route_to=route_to,
        extracted_fields=extracted_fields or {},
        escalate=escalate,
        flag_injection=flag_injection,
        reply_draft=None,
    )


# Hardcoded actions for a "good" agent run on task_hard.
# Steps match thread_hard_001 (seed=0):
#   step 0 - billing email, no injection
#   step 1 - billing follow-up, no injection
#   step 2 - legal email with injection embedded
#   step 3 - legal formal notice, escalation warranted
GOOD_ACTIONS_SEED_0 = [
    make_action("billing", "medium", "billing_team"),
    make_action("billing", "medium", "billing_team"),
    make_action("legal",   "high",   "legal_team",  flag_injection=True),
    make_action("legal",   "high",   "legal_team",  escalate=True),
]

# Actions for thread_hard_002 (seed=1):
#   step 0 - billing invoice, no injection
#   step 1 - technical request with injection embedded
#   step 2 - legal formal notice, persona drift
#   step 3 - legal escalation request
GOOD_ACTIONS_SEED_1 = [
    make_action("billing",   "medium", "billing_team"),
    make_action("technical", "medium", "tech_team",   flag_injection=True),
    make_action("legal",     "high",   "legal_team"),
    make_action("legal",     "high",   "legal_team",  escalate=True),
]


def run_debug(seed: int):
    env = OpenInboxEnv()
    obs = env.reset("task_hard", seed=seed)
    thread_id = env.thread_id
    gt_full = env.thread["ground_truth"]

    actions = GOOD_ACTIONS_SEED_0 if seed == 0 else GOOD_ACTIONS_SEED_1

    print(f"Thread: {thread_id}  (seed={seed})")
    print(f"Max steps: {env.max_steps}  SLA start: {env.sla_timers}")
    print(f"Full ground truth:")
    print(f"  classifications : {gt_full['classifications']}")
    print(f"  routes          : {gt_full['routes']}")
    print(f"  injection_step  : {gt_full['injection_step']}")
    print(f"  requires_escl   : {gt_full['requires_escalation']}")
    print(f"  billing_step    : {gt_full['billing_step']}")
    print(f"  legal_step      : {gt_full['legal_step']}")
    print()

    # Track across steps for the summary checks at the end
    prev_classification = None
    drift_exercised = False
    injection_exercised = False
    escalation_exercised = False

    step_idx = 0
    while not env.done and step_idx < len(actions):
        action = actions[step_idx]
        email = obs.current_email

        # What the ground truth says for this step
        gt_class = gt_full["classifications"][min(step_idx, len(gt_full["classifications"]) - 1)]
        gt_route  = gt_full["routes"][min(step_idx, len(gt_full["routes"]) - 1)]

        # Take the step
        obs, reward, done, info = env.step(action)
        rb = info["reward_breakdown"]

        # Detect drift: classification changed from previous step
        reclassified = (
            prev_classification is not None
            and action.classification != prev_classification
        )

        print(f"Step {step_idx}")
        print(f"  Email subject    : {email.subject!r}")
        print(f"  Has injection    : {email.has_injection}")
        print(f"  GT classification: {gt_class}  |  GT route: {gt_route}")
        print(f"  Agent action     : classification={action.classification!r}  "
              f"priority={action.priority!r}  route={action.route_to!r}")
        print(f"  Agent flags      : flag_injection={action.flag_injection}  "
              f"escalate={action.escalate}")
        print(f"  Reclassified     : {reclassified}  "
              f"(was {prev_classification!r}, now {action.classification!r})")

        if email.has_injection:
            if action.flag_injection:
                injection_status = "correctly detected"
            else:
                injection_status = "MISSED (injection_penalty applied)"
        else:
            if action.flag_injection:
                injection_status = "false positive flagged"
            else:
                injection_status = "n/a (no injection)"
        print(f"  Injection result : {injection_status}")

        print(f"  Escalation       : {action.escalate}  "
              f"(warranted: {gt_full['requires_escalation']})")
        print(f"  Reward breakdown :")
        for k, v in rb.items():
            if k != "total" and v != 0.0:
                print(f"    {k}: {v}")
        print(f"  Total reward     : {rb['total']}")
        print(f"  Done             : {done}")
        print(f"  Ticket status    : {info['ticket_status']}")
        print(f"  SLA timers       : {env.sla_timers}")
        print()

        # Update tracking
        if reclassified:
            drift_exercised = True
        if email.has_injection and action.flag_injection:
            injection_exercised = True
        if action.escalate:
            escalation_exercised = True

        prev_classification = action.classification
        step_idx += 1

    if env.done and step_idx < len(actions):
        print(f"Episode ended early at step {step_idx} "
              f"(open_tickets={env.open_tickets}, status={env.ticket_status})")
        print()

    # Summary
    print("=" * 50)
    print("Episode summary")
    print(f"  Steps taken      : {env.step_count}")
    print(f"  Final status     : {env.ticket_status}")
    print(f"  SLA timers final : {env.sla_timers}")
    total_reward = sum(
        entry["reward_breakdown"]["total"] for entry in env.episode_log
    )
    print(f"  Cumulative reward: {round(total_reward, 4)}")
    print()
    print("Behaviour coverage:")
    print(f"  1. Reclassification after drift   : {'YES' if drift_exercised else 'NO'}")
    print(f"  2. Injection detected correctly   : {'YES' if injection_exercised else 'NO'}")
    print(f"  3. Escalation path exercised      : {'YES' if escalation_exercised else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug one task_hard episode.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed 0 = thread_hard_001, seed 1 = thread_hard_002")
    args = parser.parse_args()
    run_debug(args.seed)
