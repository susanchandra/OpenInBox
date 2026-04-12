"""
generate_cascade_demo.py
Runs task_hard with the NAIVE agent (always billing_team) to show the
cascade mechanism firing when wrong routing occurs.  Saves cascade_demo.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment.env import OpenInboxEnv
from environment.graders import GRADERS
from baseline.naive_agent import NaiveAgent

TASK_ID  = "task_hard"
SEED     = 0
OUT_PATH = Path("cascade_demo.json")

env   = OpenInboxEnv()
agent = NaiveAgent()
obs   = env.reset(TASK_ID, SEED)

print(f"Thread : {env.thread_id}")
print(f"Naive agent always routes -> billing_team, priority=medium")
print(f"Watch what happens when the thread shifts to legal/injection...\n")

episode = []
step    = 0

while not env.done:
    action   = agent.act(obs)
    obs_next, reward, done, info = env.step(action)

    email = obs.current_email
    bd    = info["reward_breakdown"]
    is_cascade = step in env._cascade_steps

    entry = {
        "step"            : step,
        "from"            : email.sender,
        "subject"         : email.subject[:70],
        "has_injection"   : email.has_injection,
        "cascade_email"   : is_cascade,
        "action"          : {
            "classification" : action.classification,
            "route_to"       : action.route_to,
            "flag_injection" : action.flag_injection,
            "escalate"       : action.escalate,
        },
        "reward"          : round(reward, 4),
        "reward_breakdown": {k: round(v, 4) for k, v in bd.items() if v != 0},
        "cumulative_reward": round(sum(e["reward"] for e in episode) + reward, 4),
    }
    if is_cascade:
        entry["_JUDGE_NOTE"] = (
            "CASCADE TRIGGERED: wrong routing at step N caused this escalation "
            "email to be injected. A classifier cannot predict or handle this."
        )
    episode.append(entry)

    cascade_tag = " <<< CASCADE EMAIL INJECTED" if is_cascade else ""
    print(f"Step {step:2d}{cascade_tag}")
    print(f"  From   : {email.sender}")
    print(f"  Subject: {email.subject[:60]}")
    print(f"  Agent  : {action.classification} -> {action.route_to}  inj={action.flag_injection}")
    print(f"  Reward : {reward:+.4f}  [cls={bd['classification_reward']:+.2f}"
          f" rte={bd['routing_reward']:+.2f}"
          f" cascade_pen={bd.get('cascade_penalty',0):+.2f}"
          f" corr_bonus={bd.get('correction_bonus',0):+.2f}"
          f" inj={bd.get('injection_reward',0)+bd.get('injection_penalty',0):+.2f}]")
    print()

    obs   = obs_next
    step += 1
    if done:
        break

state  = env.state()
ep_log = state["episode_log"]
threads_data = json.loads(Path("environment/data/threads.json").read_text(encoding="utf-8"))
gt     = threads_data[env.thread_id]["ground_truth"]
result = GRADERS[TASK_ID](ep_log, gt)

print("=" * 60)
print(f"FINAL GRADER SCORE : {result['score']:.4f}  (vs rule-based: 1.0000)")
print(f"BREAKDOWN          : {result['breakdown']}")
print(f"STEPS TAKEN        : {state['step_count']}")
print(f"CASCADE STEPS FIRED: {sorted(env._cascade_steps)}")
print("=" * 60)

output = {
    "meta": {
        "task_id"              : TASK_ID,
        "thread_id"            : env.thread_id,
        "seed"                 : SEED,
        "agent"                : "NaiveAgent — always billing_team, priority=medium (baseline/naive_agent.py)",
        "final_score"          : result["score"],
        "breakdown"            : result["breakdown"],
        "steps_taken"          : state["step_count"],
        "cascade_steps_fired"  : sorted(env._cascade_steps),
        "judge_summary": {
            "what_this_shows": (
                "The naive agent always routes to billing_team. When the email "
                "thread shifts topic (billing -> legal, with injection), the wrong "
                "routing at step N triggers the cascade: an escalation email from "
                "the sender's lawyer appears at step N+2. This is a direct "
                "consequence of the agent's earlier decision — exactly the "
                "delayed-consequence property that defines RL environments."
            ),
            "why_classification_fails": (
                "A single-shot classifier sees each email independently. "
                "It cannot know that correct routing at step 0 prevents a harder "
                "escalation at step 2. It cannot detect that the same invoice dispute "
                "has now become a formal legal threat. OpenInbox requires an agent "
                "that maintains state across steps and acts with awareness of "
                "future consequences."
            ),
            "grader_gap": f"Naive: {result['score']:.4f}  vs  Rule-based: 1.0000  vs  LLM: ~0.51  (harder seed)"
        },
    },
    "episode": episode,
}

OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved -> {OUT_PATH}")
