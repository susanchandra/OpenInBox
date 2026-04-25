"""
OpenInbox Ablation Study — Proves Each Environment Mechanism Contributes to Hardness

The idea: disable ONE mechanism at a time, run the same baseline agent,
and measure how much EASIER the environment becomes.

If removing a feature makes the env easier → that feature is essential.
If removing a feature makes no difference → that feature is dead weight.

We run 6 configurations:
  1. FULL ENV       — all features on (this is the frozen env)
  2. NO DRIFT       — disable reliability drift at steps 7/14
  3. NO CASCADE     — disable cascade consequence chain
  4. NO BUDGET      — remove budget_penalty from reward
  5. NO SLA         — disable SLA timers entirely
  6. NO INJECTION   — disable injection detection rewards

For each config, we run 20 episodes with a RULE-BASED heuristic agent
and record: avg_return, success_rate, avg_steps_survived.

The ablation chart shows: "Remove drift → agent improves by X%"
proving that drift is what makes the env hard.

Usage:
  python ablation_study.py
  → generates ablation_results.json + ablation_chart.png
"""

import sys
import json
import copy
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from environment.env import OpenInboxEnv
from environment.models import Action


# ═══════════════════════════════════════════════════════════════════════
# HEURISTIC AGENT (same agent for ALL configs — the control variable)
# ═══════════════════════════════════════════════════════════════════════

def heuristic_action(obs) -> Action:
    """
    Simple keyword-based agent. Does NOT adapt to drift, cascade, or injection.
    This is the agent whose performance changes as we toggle env features.
    """
    email = obs.current_email
    body = (email.body + " " + email.subject).lower()

    # Simple keyword routing
    if any(w in body for w in ["invoice", "payment", "billing", "refund", "charge"]):
        classification = "billing"
        route = "delegate_fast"
    elif any(w in body for w in ["legal", "lawsuit", "counsel", "compliance", "regulation"]):
        classification = "legal"
        route = "delegate_thorough"
    elif any(w in body for w in ["technical", "bug", "error", "crash", "server"]):
        classification = "technical"
        route = "delegate_fast"
    elif any(w in body for w in ["hr", "leave", "salary", "employee", "vacation"]):
        classification = "hr"
        route = "delegate_thorough"
    elif any(w in body for w in ["spam", "unsubscribe", "click here", "winner"]):
        classification = "spam"
        route = "handle_self"
    else:
        classification = "billing"
        route = "delegate_fast"

    # Priority based on urgency keywords
    if any(w in body for w in ["urgent", "critical", "immediately", "asap", "deadline"]):
        priority = "critical"
    elif any(w in body for w in ["important", "high priority", "escalat"]):
        priority = "high"
    else:
        priority = "medium"

    # Never flags injection (heuristic is blind to it)
    # Never escalates (to isolate budget effects)
    return Action(
        classification=classification,
        priority=priority,
        route_to=route,
        escalate=False,
        flag_injection=False,
        extracted_fields={},
        reply_draft=None,
    )


# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATIONS (each ablation disables ONE feature)
# ═══════════════════════════════════════════════════════════════════════

class AblationEnv:
    """Wraps OpenInboxEnv with toggleable features for ablation."""

    def __init__(self, config: dict):
        self.env = OpenInboxEnv()
        self.config = config
        self.name = config["name"]

    def reset(self, task_id: str, seed: int):
        obs = self.env.reset(task_id, seed)

        # NO_SLA: Remove SLA timers entirely
        if not self.config.get("sla_enabled", True):
            self.env.sla_timers = {}

        return obs

    def step(self, action: Action):
        # NO_DRIFT: Prevent reliability drift by resetting reliability before step
        if not self.config.get("drift_enabled", True):
            self.env.internal_reliability = {
                "delegate_fast": 1.00,
                "delegate_thorough": 1.00,
                "handle_self": 1.00,
            }

        # NO_CASCADE: Clear cascade queue so no cascade emails arrive
        if not self.config.get("cascade_enabled", True):
            self.env._cascade_queue = {}
            self.env._cascade_steps = set()

        obs, reward, done, info = self.env.step(action)

        # NO_BUDGET: Zero out budget_penalty from reward
        if not self.config.get("budget_enabled", True):
            bd = info.get("reward_breakdown", {})
            if isinstance(bd, dict):
                budget_pen = bd.get("budget_penalty", 0)
            else:
                budget_pen = getattr(bd, "budget_penalty", 0)
            reward = reward - budget_pen  # remove the penalty
            self.env.budget_remaining = 1.00  # never deplete budget

        # NO_INJECTION: Zero out injection reward/penalty
        if not self.config.get("injection_enabled", True):
            bd = info.get("reward_breakdown", {})
            if isinstance(bd, dict):
                inj_r = bd.get("injection_reward", 0)
                inj_p = bd.get("injection_penalty", 0)
            else:
                inj_r = getattr(bd, "injection_reward", 0)
                inj_p = getattr(bd, "injection_penalty", 0)
            reward = reward - inj_r - inj_p

        return obs, reward, done, info


ABLATION_CONFIGS = [
    {
        "name": "FULL ENV (all features)",
        "drift_enabled": True,
        "cascade_enabled": True,
        "budget_enabled": True,
        "sla_enabled": True,
        "injection_enabled": True,
    },
    {
        "name": "NO DRIFT (remove reliability drift)",
        "drift_enabled": False,
        "cascade_enabled": True,
        "budget_enabled": True,
        "sla_enabled": True,
        "injection_enabled": True,
    },
    {
        "name": "NO CASCADE (remove delayed consequences)",
        "drift_enabled": True,
        "cascade_enabled": False,
        "budget_enabled": True,
        "sla_enabled": True,
        "injection_enabled": True,
    },
    {
        "name": "NO BUDGET (remove escalation cost)",
        "drift_enabled": True,
        "cascade_enabled": True,
        "budget_enabled": False,
        "sla_enabled": True,
        "injection_enabled": True,
    },
    {
        "name": "NO SLA (remove time pressure)",
        "drift_enabled": True,
        "cascade_enabled": True,
        "budget_enabled": True,
        "sla_enabled": False,
        "injection_enabled": True,
    },
    {
        "name": "NO INJECTION (remove adversarial input)",
        "drift_enabled": True,
        "cascade_enabled": True,
        "budget_enabled": True,
        "sla_enabled": True,
        "injection_enabled": False,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# RUN ABLATION
# ═══════════════════════════════════════════════════════════════════════

N_EPISODES = 20
TASK_ID = "task_hard"  # Ablation is most revealing on the hardest task


def run_ablation_config(config: dict) -> dict:
    """Run N_EPISODES with heuristic agent on one ablation config."""
    aenv = AblationEnv(config)
    results = []

    for seed in range(N_EPISODES):
        obs = aenv.reset(TASK_ID, seed)
        total_reward = 0.0
        steps = 0
        escalations = 0

        while not aenv.env.done:
            action = heuristic_action(obs)
            obs, reward, done, info = aenv.step(action)
            total_reward += reward
            steps += 1
            if action.escalate or action.route_to == "escalate":
                escalations += 1
            if done:
                break

        # Episode success = resolved + SLA met + budget > 20%
        resolved = aenv.env.ticket_status == "resolved"
        budget_ok = aenv.env.budget_remaining > 0.20
        sla_ok = all(t > 0 for t in aenv.env.sla_timers.values()) if aenv.env.sla_timers else True
        success = resolved and budget_ok and sla_ok

        results.append({
            "seed": seed,
            "total_reward": round(total_reward, 4),
            "steps": steps,
            "success": success,
            "escalations": escalations,
            "budget_remaining": round(aenv.env.budget_remaining, 4),
        })

    avg_reward = sum(r["total_reward"] for r in results) / N_EPISODES
    avg_steps = sum(r["steps"] for r in results) / N_EPISODES
    success_rate = sum(r["success"] for r in results) / N_EPISODES * 100
    avg_budget = sum(r["budget_remaining"] for r in results) / N_EPISODES

    return {
        "config_name": config["name"],
        "avg_reward": round(avg_reward, 4),
        "avg_steps": round(avg_steps, 1),
        "success_rate": round(success_rate, 1),
        "avg_budget_remaining": round(avg_budget, 4),
        "episodes": results,
    }


def main():
    print("=" * 70)
    print("OpenInbox Ablation Study")
    print(f"Task: {TASK_ID}  |  Episodes per config: {N_EPISODES}")
    print(f"Agent: Heuristic (keyword-based, same for all configs)")
    print("=" * 70)

    all_results = []

    for i, config in enumerate(ABLATION_CONFIGS):
        print(f"\n[{i+1}/{len(ABLATION_CONFIGS)}] Running: {config['name']} ...")
        result = run_ablation_config(config)
        all_results.append(result)
        print(f"  avg_reward={result['avg_reward']:+.4f}  "
              f"success={result['success_rate']:.1f}%  "
              f"avg_steps={result['avg_steps']:.1f}  "
              f"avg_budget={result['avg_budget_remaining']:.4f}")

    # Save raw results
    results_path = Path("ablation_results.json")
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {results_path}")

    # ─── Print comparison table ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    full_reward = all_results[0]["avg_reward"]
    full_success = all_results[0]["success_rate"]

    print(f"\n  {'Configuration':<40}  {'Avg Reward':>10}  {'Δ Reward':>10}  {'Success':>8}  {'Harder?':>8}")
    print(f"  {'-'*38}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for r in all_results:
        delta = r["avg_reward"] - full_reward
        if r == all_results[0]:
            delta_str = "baseline"
            harder = "—"
        else:
            delta_str = f"{delta:+.4f}"
            # If removing the feature makes reward HIGHER → that feature was making env harder
            harder = "YES ✓" if delta > 0.05 else "no"

        print(f"  {r['config_name']:<40}  {r['avg_reward']:+10.4f}  {delta_str:>10}  "
              f"{r['success_rate']:7.1f}%  {harder:>8}")

    # ─── Interpretation ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    # Sort ablations (skip full env) by delta reward (biggest improvement when removed = hardest feature)
    ablations = [(r, r["avg_reward"] - full_reward) for r in all_results[1:]]
    ablations.sort(key=lambda x: x[1], reverse=True)

    print("\nFeatures ranked by HARDNESS CONTRIBUTION (most → least):")
    for rank, (r, delta) in enumerate(ablations, 1):
        feature = r["config_name"].split("(")[1].rstrip(")")
        if delta > 0.05:
            print(f"  {rank}. {feature}: removing it improves reward by {delta:+.4f} → ESSENTIAL for hardness")
        elif delta > 0.01:
            print(f"  {rank}. {feature}: removing it slightly improves by {delta:+.4f} → contributes to hardness")
        elif delta < -0.01:
            print(f"  {rank}. {feature}: removing it HURTS reward by {delta:+.4f} → feature helps agent (interesting!)")
        else:
            print(f"  {rank}. {feature}: removing it changes reward by {delta:+.4f} → minimal impact")

    # ─── Generate chart ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("OpenInbox Ablation Study — Which Features Make It Hard?",
                     fontsize=14, fontweight="bold", y=1.02)

        names = [r["config_name"].split("(")[0].strip() for r in all_results]
        rewards = [r["avg_reward"] for r in all_results]
        successes = [r["success_rate"] for r in all_results]

        # Colors: full env = dark blue, ablations = gradient from green (easier) to red
        colors_reward = ["#1a237e"]  # full env
        for r in all_results[1:]:
            delta = r["avg_reward"] - full_reward
            if delta > 0.10:
                colors_reward.append("#2e7d32")   # green = much easier without it
            elif delta > 0.02:
                colors_reward.append("#66bb6a")   # light green
            elif delta < -0.02:
                colors_reward.append("#e53935")   # red = harder without it (unexpected)
            else:
                colors_reward.append("#9e9e9e")   # grey = no impact

        # Panel 1: Average Reward per Config
        ax1 = axes[0]
        bars1 = ax1.barh(range(len(names)), rewards, color=colors_reward, edgecolor="white", height=0.6)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel("Average Episode Reward", fontsize=10)
        ax1.set_title("Avg Reward (higher = env is easier)", fontsize=11)
        ax1.axvline(x=full_reward, color="#1a237e", linestyle="--", alpha=0.5, label="Full env baseline")
        ax1.invert_yaxis()
        for i, (bar, val) in enumerate(zip(bars1, rewards)):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:+.3f}", va="center", fontsize=8, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(axis="x", alpha=0.3)

        # Panel 2: Success Rate per Config
        ax2 = axes[1]
        bars2 = ax2.barh(range(len(names)), successes, color=colors_reward, edgecolor="white", height=0.6)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel("Success Rate (%)", fontsize=10)
        ax2.set_title("Success Rate (higher = env is easier)", fontsize=11)
        ax2.axvline(x=full_success, color="#1a237e", linestyle="--", alpha=0.5, label="Full env baseline")
        ax2.invert_yaxis()
        for i, (bar, val) in enumerate(zip(bars2, successes)):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.0f}%", va="center", fontsize=8, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        chart_path = "ablation_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n📊 Chart saved → {chart_path}")

    except ImportError:
        print("\n⚠️  matplotlib not installed — skipping chart generation")
        print("   Install with: pip install matplotlib")

    print("\n✅ Ablation study complete")


if __name__ == "__main__":
    main()
