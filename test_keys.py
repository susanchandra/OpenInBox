"""Quick verification: checks OpenAI key, HF token, and HF Space endpoints."""
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
HF_TOKEN   = os.getenv("HF_TOKEN")
API_BASE   = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_URL    = "https://susannnnn-openinbox.hf.space"

print("=" * 55)
print("  Bellman_breakers — Key & Endpoint Verification")
print("=" * 55)

# ── 1. Check .env values are present ──────────────────────────
print("\n[1] Checking .env values...")
ok = True
for name, val in [("API_KEY", API_KEY), ("HF_TOKEN", HF_TOKEN), ("API_BASE_URL", API_BASE)]:
    if val and "PASTE" not in val and len(val) > 5:
        print(f"    [OK]   {name} is set")
    else:
        print(f"    [FAIL] {name} is MISSING or placeholder!")
        ok = False
if not ok:
    print("\n[STOP] Fix the .env file first, then re-run.")
    raise SystemExit(1)

# ── 2. Test OpenAI API key ─────────────────────────────────────
print(f"\n[2] Testing OpenAI key with model '{MODEL_NAME}'...")
try:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
        temperature=0,
    )
    print(f"    [OK]   OpenAI key works! Response: '{resp.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"    [FAIL] OpenAI key failed: {e}")

# ── 3. Test HF Space endpoints ─────────────────────────────────
print(f"\n[3] Testing HF Space: {ENV_URL}")
try:
    import httpx
    r_tasks = httpx.get(f"{ENV_URL}/tasks", timeout=30)
    print(f"    [OK]   /tasks  -> status {r_tasks.status_code}")

    r_reset = httpx.post(f"{ENV_URL}/reset", json={"task_id": "task_easy", "seed": 0}, timeout=30)
    d = r_reset.json()
    print(f"    [OK]   /reset  -> status {r_reset.status_code}  thread_id={d.get('thread_id')}")
except Exception as e:
    print(f"    [FAIL] HF Space error: {e}")

print("\n" + "=" * 55)
print("  Done!")
print("=" * 55)
