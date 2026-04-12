"""
generate_sample_episode.py
Runs one full task_hard episode using the rule-based agent and saves a
human-readable episode log to sample_episode.json.  Safe to run anytime.
Does NOT import inference.py or touch the API.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment.env import OpenInboxEnv
from environment.graders import GRADERS
from baseline.rule_agent import RuleAgent

TASK_ID  = "task_hard"
SEED     = 0
OUT_PATH = Path("sample_episode.json")

env   = OpenInboxEnv()
agent = RuleAgent()
obs   = env.reset(TASK_ID, SEED)

print(f"Thread : {env.thread_id}")
print(f"Task   : {TASK_ID}")
print()

episode = []
step    = 0

while not env.done:
    action = agent.act(obs)
    obs_next, reward, done, info = env.step(action)

    email  = obs.current_email
    bd     = info["reward_breakdown"]

    cascade_marker = "*** CASCADE EMAIL INJECTED ***" if step in env._cascade_steps else ""

    entry = {
        "step"            : step,
        "email_id"        : email.id,
        "from"            : email.sender,
        "subject"         : email.subject,
        "has_injection"   : email.has_injection,
        "cascade_step"    : step in env._cascade_steps,
        "action"          : {
            "classification" : action.classification,
            "priority"       : action.priority,
            "route_to"       : action.route_to,
            "escalate"       : action.escalate,
            "flag_injection" : action.flag_injection,
        },
        "reward"          : reward,
        "reward_breakdown": bd,
        "ticket_status"   : info["ticket_status"],
    }
    if cascade_marker:
        entry["NOTE"] = cascade_marker

    episode.append(entry)

    label = f"[CASCADE]" if cascade_marker else ""
    print(f"Step {step:2d} {label}")
    print(f"  From   : {email.sender}")
    print(f"  Subject: {email.subject}")
    print(f"  Action : {action.classification} -> {action.route_to}"
          f"  escalate={action.escalate}  flag_injection={action.flag_injection}")
    print(f"  Reward : {reward:+.4f}  "
          f"(cls={bd['classification_reward']:+.2f} rte={bd['routing_reward']:+.2f} "
          f"inj={bd.get('injection_reward',0)+bd.get('injection_penalty',0):+.2f} "
          f"esc_tradeoff={bd.get('escalation_tradeoff_bonus',0):+.2f} "
          f"cascade_pen={bd.get('cascade_penalty',0):+.2f} "
          f"corr_bonus={bd.get('correction_bonus',0):+.2f})")
    print()

    obs   = obs_next
    step += 1
    if done:
        break

# Final grading
state  = env.state()
ep_log = state["episode_log"]
threads_data = json.loads(Path("environment/data/threads.json").read_text(encoding="utf-8"))
gt     = threads_data[env.thread_id]["ground_truth"]
result = GRADERS[TASK_ID](ep_log, gt)

print("=" * 52)
print(f"FINAL GRADER SCORE : {result['score']:.4f}")
print(f"BREAKDOWN          : {result['breakdown']}")
print(f"STEPS TAKEN        : {state['step_count']}")
print(f"CASCADE STEPS      : {sorted(env._cascade_steps)}")
print("=" * 52)

# Save a rich JSON with metadata
output = {
    "meta": {
        "task_id"     : TASK_ID,
        "thread_id"   : env.thread_id,
        "seed"        : SEED,
        "agent"       : "rule-based keyword agent (baseline/rule_agent.py)",
        "final_score" : result["score"],
        "breakdown"   : result["breakdown"],
        "steps_taken" : state["step_count"],
        "cascade_steps_triggered": sorted(env._cascade_steps),
        "note": (
            "This episode log demonstrates that OpenInbox requires multi-step "
            "sequential reasoning. Wrong routing at step N triggers a CASCADE: "
            "a harder escalation email appears at step N+2, carrying additional "
            "penalty/recovery signals. A classifier cannot model this."
        ),
    },
    "episode": episode,
}
OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved → {OUT_PATH}")
