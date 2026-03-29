"""
End-to-end test for all 5 API endpoints.

Uses FastAPI's TestClient so no server process is needed.
Runs a complete episode for task_easy, then calls /grader against
the episode log from /state.

Run with:  python test_api.py
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def separator(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def test_tasks():
    separator("GET /tasks")
    r = client.get("/tasks")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "task_easy" in data
    assert "task_medium" in data
    assert "task_hard" in data
    for task_id, cfg in data.items():
        assert "max_steps" in cfg
        assert "thread_ids" in cfg
        print(f"  {task_id}: max_steps={cfg['max_steps']}, threads={cfg['thread_ids']}")
    print("  /tasks OK")


def test_reset_unknown_task():
    separator("POST /reset — unknown task_id")
    r = client.post("/reset", json={"task_id": "task_nonexistent"})
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"  Got 400 as expected: {r.json()['detail']}")


def test_step_unknown_session():
    separator("POST /step — unknown session")
    r = client.post("/step", json={
        "session_id": "does-not-exist",
        "action": {
            "classification": "billing", "priority": "medium",
            "route_to": "billing_team", "extracted_fields": {},
            "escalate": False, "flag_injection": False, "reply_draft": None
        }
    })
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print(f"  Got 404 as expected: {r.json()['detail']}")


def test_full_easy_episode():
    separator("Full task_easy episode (seed=0)")

    # 1. Reset
    r = client.post("/reset", json={"task_id": "task_easy", "seed": 0})
    assert r.status_code == 200, r.text
    data = r.json()
    session_id = data["session_id"]
    thread_id = data["thread_id"]
    obs = data["observation"]
    print(f"  session_id : {session_id}")
    print(f"  thread_id  : {thread_id}")
    print(f"  email      : {obs['current_email']['subject']!r}")
    print(f"  max_steps  : {data['max_steps']}")
    print(f"  sla_timers : {obs['sla_timers']} (empty for task_easy)")

    # 2. Step — correct action for thread_easy_001
    action = {
        "classification": "billing",
        "priority": "medium",
        "route_to": "billing_team",
        "extracted_fields": {
            "invoice_number": "4821",
            "amount": "3200",
            "submitted_date": "2024-03-10",
        },
        "escalate": False,
        "flag_injection": False,
        "reply_draft": None,
    }
    r = client.post("/step", json={"session_id": session_id, "action": action})
    assert r.status_code == 200, r.text
    step_data = r.json()
    print(f"\n  After step:")
    print(f"    reward   : {step_data['reward']}")
    print(f"    done     : {step_data['done']}")
    print(f"    ticket   : {step_data['info']['ticket_status']}")
    print(f"    breakdown: {step_data['info']['reward_breakdown']}")
    assert step_data["done"] is True, "Expected done=True after correct routing"
    assert step_data["reward"] > 0.5, "Expected reward > 0.5 for correct action"

    # 3. State
    r = client.get(f"/state?session_id={session_id}")
    assert r.status_code == 200, r.text
    state_data = r.json()
    print(f"\n  /state:")
    print(f"    step_count  : {state_data['step_count']}")
    print(f"    ticket      : {state_data['ticket_status']}")
    print(f"    log entries : {len(state_data['episode_log'])}")
    episode_log = state_data["episode_log"]

    # 4. Step after done — should return 400
    r = client.post("/step", json={"session_id": session_id, "action": action})
    assert r.status_code == 400, f"Expected 400 after done, got {r.status_code}"
    print(f"\n  Stepping after done returned 400 as expected.")

    # 5. Grader — pass episode_log from state, thread_id resolved automatically
    r = client.post("/grader", json={
        "task_id": "task_easy",
        "episode_log": episode_log,
    })
    assert r.status_code == 200, r.text
    grade_data = r.json()
    print(f"\n  /grader result:")
    print(f"    score     : {grade_data['score']}  (expect ~1.0 for perfect action)")
    print(f"    breakdown : {grade_data['breakdown']}")
    assert 0.0 <= grade_data["score"] <= 1.0

    return session_id, thread_id, episode_log


def test_full_medium_episode():
    separator("Full task_medium episode — SLA breach simulation")

    r = client.post("/reset", json={"task_id": "task_medium", "seed": 0})
    assert r.status_code == 200
    session_id = r.json()["session_id"]
    obs = r.json()["observation"]
    print(f"  thread   : {r.json()['thread_id']}")
    print(f"  SLA start: {obs['sla_timers']}")

    # Send a wrong action — SLA will tick every step
    wrong_action = {
        "classification": "unknown", "priority": "low",
        "route_to": "spam_filter", "extracted_fields": {},
        "escalate": False, "flag_injection": False, "reply_draft": None,
    }
    for i in range(6):
        r = client.post("/step", json={"session_id": session_id, "action": wrong_action})
        assert r.status_code == 200, r.text
        d = r.json()
        state_r = client.get(f"/state?session_id={session_id}")
        sla = state_r.json()["sla_timers"]
        print(f"  step {i}: reward={d['reward']:.3f} done={d['done']} sla={sla}")
        if d["done"]:
            print(f"  Episode ended at step {i} (SLA breach or timeout expected)")
            break


def test_hard_episode_grader():
    separator("task_hard episode — grader check")

    r = client.post("/reset", json={"task_id": "task_hard", "seed": 0})
    assert r.status_code == 200
    session_id = r.json()["session_id"]
    print(f"  thread: {r.json()['thread_id']}")

    actions = [
        {"classification":"billing",  "priority":"medium","route_to":"billing_team","extracted_fields":{},"escalate":False,"flag_injection":False,"reply_draft":None},
        {"classification":"billing",  "priority":"medium","route_to":"billing_team","extracted_fields":{},"escalate":False,"flag_injection":False,"reply_draft":None},
        {"classification":"legal",    "priority":"high",  "route_to":"legal_team",  "extracted_fields":{},"escalate":False,"flag_injection":True, "reply_draft":None},
        {"classification":"legal",    "priority":"high",  "route_to":"legal_team",  "extracted_fields":{},"escalate":True, "flag_injection":False,"reply_draft":None},
    ]
    for i, action in enumerate(actions):
        r = client.post("/step", json={"session_id": session_id, "action": action})
        assert r.status_code == 200
        d = r.json()
        print(f"  step {i}: reward={d['reward']:.3f} done={d['done']}")
        if d["done"]:
            break

    state_r = client.get(f"/state?session_id={session_id}")
    episode_log = state_r.json()["episode_log"]

    r = client.post("/grader", json={
        "task_id": "task_hard",
        "episode_log": episode_log,
    })
    assert r.status_code == 200
    grade_data = r.json()
    print(f"  /grader score: {grade_data['score']}  (expect 1.0 for perfect hard agent)")
    print(f"  breakdown: {grade_data['breakdown']}")
    assert grade_data["score"] == 1.0, f"Expected 1.0, got {grade_data['score']}"


if __name__ == "__main__":
    test_tasks()
    test_reset_unknown_task()
    test_step_unknown_session()
    _, _, _ = test_full_easy_episode()
    test_full_medium_episode()
    test_hard_episode_grader()
    print("\nAll API tests passed.")
