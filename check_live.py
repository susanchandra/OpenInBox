import httpx
import time

base = "https://susannnnn-openinbox.hf.space"

print("Waiting 30s for Space to finish restarting...")
time.sleep(30)

try:
    r1 = httpx.get(f"{base}/tasks", timeout=60)
    print(f"/tasks   status={r1.status_code}  keys={list(r1.json().keys())}")

    r2 = httpx.post(f"{base}/reset", json={"task_id": "task_easy", "seed": 0}, timeout=60)
    d2 = r2.json()
    print(f"/reset   status={r2.status_code}  thread={d2.get('thread_id')}")

    r3 = httpx.post(f"{base}/reset", json={"task_id": "task_hard", "seed": 0}, timeout=60)
    print(f"/reset hard  status={r3.status_code}  thread={r3.json().get('thread_id')}")

    if all(r.status_code == 200 for r in [r1, r2, r3]):
        print()
        print("All endpoints live. Space is running correctly.")
    else:
        print("One or more endpoints returned non-200.")

except httpx.ReadTimeout:
    print("Still starting up. Wait 30 more seconds then run: python check_live.py")
except Exception as e:
    print(f"Error: {e}")
