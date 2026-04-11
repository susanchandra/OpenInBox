import httpx

base = "https://susannnnn-openinbox.hf.space"

# Case 1: no body (validator probe)
r1 = httpx.post(f"{base}/reset", timeout=45)
print(f"POST /reset (no body)    status={r1.status_code}  task={r1.json().get('task_id')}  thread={r1.json().get('thread_id')}")

# Case 2: with body task_easy
r2 = httpx.post(f"{base}/reset", json={"task_id": "task_easy", "seed": 0}, timeout=45)
print(f"POST /reset (task_easy)  status={r2.status_code}  thread={r2.json().get('thread_id')}")

# Case 3: with body task_hard
r3 = httpx.post(f"{base}/reset", json={"task_id": "task_hard", "seed": 0}, timeout=45)
print(f"POST /reset (task_hard)  status={r3.status_code}  thread={r3.json().get('thread_id')}")

# Case 4: /tasks still works
r4 = httpx.get(f"{base}/tasks", timeout=45)
print(f"GET  /tasks              status={r4.status_code}  keys={list(r4.json().keys())}")

print()
if all(r.status_code == 200 for r in [r1, r2, r3, r4]):
    print("All live checks passed. Validator compatibility fix confirmed.")
else:
    print("Some checks failed.")
