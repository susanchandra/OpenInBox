# =============================================================================
# OPENINBOX — GRPO TRAINING PIPELINE (Google Colab)
# ENV: https://susannnnn-openinbox.hf.space
# Model: Qwen2.5-1.5B-Instruct
# =============================================================================

# ===========================================================================
# CELL 1 — Installation
# ===========================================================================
# %%
!pip install -q trl transformers torch accelerate peft bitsandbytes \
               requests pydantic numpy matplotlib datasets

import importlib, sys
for pkg in ["trl", "transformers", "torch", "requests", "pydantic", "datasets"]:
    try:
        importlib.import_module(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} MISSING")

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("\n✅ Installation complete")


# ===========================================================================
# CELL 2 — Environment Connection
# ===========================================================================
# %%
import requests, json, time, random

ENV_SERVER_URL = "https://susannnnn-openinbox.hf.space"
TASK_ID        = "task_easy"   # use task_easy for speed during training
FALLBACK_TASK  = "task_hard"

# ── Health check ────────────────────────────────────────────────────────────
def health_check():
    r = requests.get(f"{ENV_SERVER_URL}/health", timeout=15)
    r.raise_for_status()
    return r.json()

# ── Session client ───────────────────────────────────────────────────────────
class EnvClient:
    """Thin stateless wrapper around the OpenInbox REST API."""

    def __init__(self, base_url: str):
        self.base = base_url
        self.session_id: str | None = None
        self.done: bool = False
        self._last_obs: dict = {}

    def reset(self, task_id: str = TASK_ID, seed: int = 0) -> dict:
        r = requests.post(
            f"{self.base}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        self.done       = False
        self._last_obs  = data["observation"]
        return data

    def step(self, action: dict) -> dict:
        r = requests.post(
            f"{self.base}/step",
            json={"session_id": self.session_id, "action": action},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        self.done       = data["done"]
        self._last_obs  = data["observation"]
        return data

    @staticmethod
    def make_action(
        route_to: str = "delegate_fast",
        classification: str = "billing",
        priority: str = "medium",
        escalate: bool = False,
        flag_injection: bool = False,
        extracted_fields: dict | None = None,
        reply_draft: str | None = None,
    ) -> dict:
        return {
            "route_to":         route_to,
            "classification":   classification,
            "priority":         priority,
            "escalate":         escalate,
            "flag_injection":   flag_injection,
            "extracted_fields": extracted_fields or {},
            "reply_draft":      reply_draft,
        }

# ── Test connection ──────────────────────────────────────────────────────────
print("Checking server health …")
h = health_check()
print(f"  Status  : {h['status']}")
print(f"  Version : {h['version']}")
print(f"  Tasks   : {list(h['tasks'].keys())}")

env = EnvClient(ENV_SERVER_URL)
data = env.reset(TASK_ID, seed=0)
print("\nEnvironment connected successfully")
print(f"\nFirst observation structure:")
obs = data["observation"]
print(f"  session_id      : {data['session_id'][:16]}…")
print(f"  thread_id       : {data['thread_id']}")
print(f"  subject         : {obs['current_email']['subject'][:60]}")
print(f"  flags           : {obs['flags']}")
print(f"  delegation_hist : {list(obs['delegation_history'].keys())[:4]}")


# ===========================================================================
# CELL 3 — Timing Test (run BEFORE baseline, so you can plan your schedule)
# ===========================================================================
# %%
import time

TIMING_STEPS = 10
env2 = EnvClient(ENV_SERVER_URL)
env2.reset(TASK_ID, seed=42)

start = time.perf_counter()
steps_done = 0

for i in range(TIMING_STEPS):
    action = env2.make_action(
        route_to=random.choice(["delegate_fast", "delegate_thorough",
                                "handle_self", "wait"]),
        classification=random.choice(["billing", "legal", "spam", "unknown"]),
        priority=random.choice(["high", "medium", "low"]),
    )
    resp = env2.step(action)
    steps_done += 1
    if resp["done"]:
        env2.reset(TASK_ID, seed=steps_done)

elapsed   = time.perf_counter() - start
per_step  = elapsed / TIMING_STEPS
est_150   = per_step * 150 / 60
est_300   = per_step * 300 / 60

print(f"10 steps took {elapsed:.2f} seconds")
print(f"Per-step latency: {per_step*1000:.0f} ms")
print(f"150 steps estimated: {est_150:.1f} minutes")
print(f"300 steps estimated: {est_300:.1f} minutes")
print()
if est_300 < 15:
    print("✅ Speed is good — 300 steps fits in a Colab session")
elif est_300 < 30:
    print("⚠️  300 steps will take ~30 min — use 150 steps for safety")
else:
    print("⚠️  Env is slow — reduce to 100 training steps or add retry logic")


# ===========================================================================
# CELL 4 — Baseline Recording (untrained / random agent)
# ===========================================================================
# %%
import json, random
from pathlib import Path

N_BASELINE_EPISODES = 100
BASELINE_TASK       = TASK_ID
BASELINE_FILE       = Path("/content/baseline_metrics.json")

VALID_ROUTES = ["delegate_fast", "delegate_thorough", "handle_self", "wait", "escalate"]
VALID_CLASSES = ["billing", "legal", "spam", "unknown", "technical", "hr"]
VALID_PRIOS   = ["high", "medium", "low"]


def random_action() -> dict:
    route = random.choice(VALID_ROUTES)
    return EnvClient.make_action(
        route_to=route,
        classification=random.choice(VALID_CLASSES),
        priority=random.choice(VALID_PRIOS),
        escalate=(route == "escalate"),
    )


def run_episode_metrics(task_id: str, seed: int) -> dict:
    """Run one full episode and return scalar metrics."""
    ep_env = EnvClient(ENV_SERVER_URL)
    data   = ep_env.reset(task_id, seed=seed)

    total_reward      = 0.0
    escalation_count  = 0
    wait_total        = 0
    wait_auto_resolve = 0
    steps             = 0
    success           = False
    final_info        = {}
    budget_start      = 1.0   # assumed normalised budget

    while not ep_env.done:
        action = random_action()
        resp   = ep_env.step(action)

        r      = resp["reward"]
        info   = resp["info"]
        obs    = resp["observation"]
        done   = resp["done"]

        total_reward += r
        steps        += 1

        if action["escalate"] or action["route_to"] == "escalate":
            escalation_count += 1

        if action["route_to"] == "wait":
            wait_total += 1
            # auto-resolve happened if ticket_status is "resolved" immediately
            if info.get("ticket_status") == "resolved":
                wait_auto_resolve += 1

        if done:
            final_info = info
            # success = resolved + SLA met + budget > 20%
            bd = info.get("reward_breakdown", {})
            budget_rem = obs.get("budget_remaining", 0.0)
            sla_ok     = not obs["flags"].get("sla_at_risk", False)
            resolved   = info.get("ticket_status") == "resolved"
            success    = resolved and sla_ok and (budget_rem > 0.20)

    wait_rate = (wait_auto_resolve / max(wait_total, 1)) * 100.0

    return {
        "seed":                seed,
        "total_reward":        round(total_reward, 4),
        "escalation_count":    escalation_count,
        "wait_correct_pct":    round(wait_rate, 2),
        "episode_success":     success,
        "steps":               steps,
    }


print(f"Recording baseline over {N_BASELINE_EPISODES} episodes …")
baseline_records = []

for seed in range(N_BASELINE_EPISODES):
    rec = run_episode_metrics(BASELINE_TASK, seed)
    baseline_records.append(rec)
    if (seed + 1) % 10 == 0:
        avg_so_far = sum(r["total_reward"] for r in baseline_records) / len(baseline_records)
        print(f"  [{seed+1:3d}/100] rolling avg reward = {avg_so_far:+.4f}")

BASELINE_FILE.write_text(json.dumps(baseline_records, indent=2))

avg_reward   = sum(r["total_reward"]      for r in baseline_records) / N_BASELINE_EPISODES
avg_esc      = sum(r["escalation_count"]  for r in baseline_records) / N_BASELINE_EPISODES
avg_wait_pct = sum(r["wait_correct_pct"]  for r in baseline_records) / N_BASELINE_EPISODES
success_rate = sum(r["episode_success"]   for r in baseline_records) / N_BASELINE_EPISODES * 100

print(f"\n{'='*50}")
print(f"Baseline avg reward         : {avg_reward:+.4f}")
print(f"Baseline escalation rate    : {avg_esc:.2f} per episode")
print(f"Baseline wait correct usage : {avg_wait_pct:.1f}%")
print(f"Baseline success rate       : {success_rate:.1f}%")
print(f"Saved → {BASELINE_FILE}")
print(f"{'='*50}")

# Store for later cells
BASELINE_REWARD   = avg_reward
BASELINE_ESC      = avg_esc
BASELINE_WAIT_PCT = avg_wait_pct
BASELINE_SUCCESS  = success_rate
