"""Live end-to-end test against the deployed HF Space."""
import httpx
import json

BASE = "https://susannnnn-openinbox.hf.space"

# 1 — /tasks
print("Testing /tasks ...")
r = httpx.get(f"{BASE}/tasks", timeout=30)
tasks = r.json()
print(f"  tasks returned: {list(tasks.keys())}")
for tid, cfg in tasks.items():
    n = cfg.get("thread_count", len(cfg.get("thread_ids", [])))
    print(f"    {tid}: {n} threads, max_steps={cfg['max_steps']}")

# 2 — /reset
print("\nTesting /reset (task_easy, seed=0) ...")
r = httpx.post(f"{BASE}/reset", json={"task_id": "task_easy", "seed": 0}, timeout=30)
data = r.json()
session_id = data["session_id"]
thread_id  = data["thread_id"]
print(f"  session_id : {session_id[:12]}...")
print(f"  thread_id  : {thread_id}")
subject = data["observation"]["current_email"]["subject"]
print(f"  email      : {subject}")

# 3 — /step with a correct action
action = {
    "classification":   "billing",
    "priority":         "medium",
    "route_to":         "billing_team",
    "extracted_fields": {"invoice_number": "4821", "amount": "3200"},
    "escalate":         False,
    "flag_injection":   False,
    "reply_draft":      None,
}
print("\nTesting /step ...")
r = httpx.post(f"{BASE}/step", json={"session_id": session_id, "action": action}, timeout=30)
sdata = r.json()
reward = sdata["reward"]
done   = sdata["done"]
print(f"  reward  : {reward}")
print(f"  done    : {done}")
print(f"  breakdown: {sdata['info']['reward_breakdown']}")

# 4 — /grader
ep_log = [{"step": 0, "thread_id": thread_id, "action": action, "reward": reward}]
print("\nTesting /grader ...")
r = httpx.post(
    f"{BASE}/grader",
    json={"task_id": "task_easy", "thread_id": thread_id, "episode_log": ep_log},
    timeout=30,
)
gdata = r.json()
score = gdata["score"]
print(f"  score    : {score}")
print(f"  breakdown: {gdata['breakdown']}")

print()
print("=" * 42)
print("LIVE SPACE: ALL ENDPOINTS PASSED")
print(f"  Space URL: https://susannnnn-openinbox.hf.space")
print(f"  Docs URL : https://susannnnn-openinbox.hf.space/docs")
print("=" * 42)
