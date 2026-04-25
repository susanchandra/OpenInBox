"""
Local verification for the /reset optional body fix.
Tests both: no body and body-based calls.
"""
from fastapi.testclient import TestClient
from api.app import app

c = TestClient(app)

# Case 1: no body at all (validator probe)
r1 = c.post("/reset")
assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.text}"
d1 = r1.json()
assert "session_id" in d1
assert d1["task_id"] == "task_easy"
assert d1["seed"] == 0
print(f"POST /reset (no body)   status={r1.status_code}  task={d1['task_id']}  thread={d1['thread_id']}")

# Case 2: body with task_easy seed=0
r2 = c.post("/reset", json={"task_id": "task_easy", "seed": 0})
assert r2.status_code == 200
d2 = r2.json()
assert d2["task_id"] == "task_easy"
print(f"POST /reset (task_easy)  status={r2.status_code}  thread={d2['thread_id']}")

# Case 3: body with task_hard seed=0
r3 = c.post("/reset", json={"task_id": "task_hard", "seed": 0})
assert r3.status_code == 200
d3 = r3.json()
assert d3["task_id"] == "task_hard"
print(f"POST /reset (task_hard)  status={r3.status_code}  thread={d3['thread_id']}")

# Case 4: existing endpoints still work
r4 = c.get("/tasks")
assert r4.status_code == 200
print(f"GET  /tasks              status={r4.status_code}  keys={list(r4.json().keys())}")

print()
print("All checks passed. /reset optional body fix is working.")
