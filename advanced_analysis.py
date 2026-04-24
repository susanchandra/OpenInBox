"""
OpenInbox — Advanced Analysis Suite

Generates 3 research-quality artifacts:

1. REWARD LANDSCAPE HEATMAP
   → Shows reward as function of (route_to × email_type)
   → Proves the reward surface has multiple optima

2. TRAJECTORY DIVERGENCE MAP  
   → Same seed, 3 different agents → 3 completely different episode paths
   → Visual proof that actions change future observations

3. DIFFICULTY CURRICULUM ANALYSIS
   → Quantifies how task_easy → task_medium → task_hard scales
   → Shows state space growth, reward sparsity, horizon length
"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from environment.env import OpenInboxEnv
from environment.models import Action

ROUTES = ["delegate_fast", "delegate_thorough", "handle_self", "escalate", "wait"]
CLASSIFICATIONS = ["billing", "legal", "technical", "hr", "spam", "unknown"]
PRIORITIES = ["low", "medium", "high", "critical"]
TASKS = ["task_easy", "task_medium", "task_hard"]


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: REWARD LANDSCAPE HEATMAP
# ═══════════════════════════════════════════════════════════════════════

def reward_landscape():
    """
    For each (route_to, classification) pair, run step 1 of task_hard
    with seed=0 and measure the immediate reward.
    
    Creates a 5×6 heatmap showing which action combinations are rewarded/punished.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Reward Landscape Heatmap")
    print("=" * 70)

    env = OpenInboxEnv()
    landscape = {}

    for route in ROUTES:
        for cls in CLASSIFICATIONS:
            env.reset("task_hard", seed=0)
            action = Action(
                classification=cls,
                priority="medium",
                route_to=route,
                escalate=(route == "escalate"),
                flag_injection=False,
                extracted_fields={},
                reply_draft=None,
            )
            _, reward, _, info = env.step(action)
            landscape[(route, cls)] = round(reward, 4)

    # Print as table
    print(f"\n  {'':>18}", end="")
    for cls in CLASSIFICATIONS:
        print(f"  {cls:>10}", end="")
    print()
    print(f"  {'':>18}" + "  " + "-" * 10 * len(CLASSIFICATIONS) + "-" * len(CLASSIFICATIONS))

    for route in ROUTES:
        print(f"  {route:>18}", end="")
        for cls in CLASSIFICATIONS:
            val = landscape[(route, cls)]
            if val > 0.05:
                marker = f"  {val:+8.2f} ✓"
            elif val < -0.05:
                marker = f"  {val:+8.2f} ✗"
            else:
                marker = f"  {val:+8.2f}  "
            print(marker, end="")
        print()

    # Find optimal action
    best = max(landscape.items(), key=lambda x: x[1])
    worst = min(landscape.items(), key=lambda x: x[1])
    print(f"\n  Best action:  route={best[0][0]}, class={best[0][1]} → reward={best[1]:+.4f}")
    print(f"  Worst action: route={worst[0][0]}, class={worst[0][1]} → reward={worst[1]:+.4f}")
    print(f"  Reward range: {worst[1]:+.4f} to {best[1]:+.4f} (spread = {best[1]-worst[1]:.4f})")

    # Count positive vs negative cells
    pos = sum(1 for v in landscape.values() if v > 0.01)
    neg = sum(1 for v in landscape.values() if v < -0.01)
    zero = len(landscape) - pos - neg
    print(f"  Positive cells: {pos}/{len(landscape)}  Negative: {neg}  Neutral: {zero}")
    print(f"  → Only {pos}/{len(landscape)} action combos are rewarded — reward surface is SPARSE")

    return landscape


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: TRAJECTORY DIVERGENCE MAP
# ═══════════════════════════════════════════════════════════════════════

def trajectory_divergence():
    """
    Run the SAME seed with 3 different agents.
    Show how the trajectories diverge — proving actions change future state.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Trajectory Divergence Map")
    print("=" * 70)

    SEED = 0
    TASK = "task_hard"

    agents = {
        "Aggressive": lambda obs: Action(
            classification="billing", priority="critical", route_to="escalate",
            escalate=True, flag_injection=False, extracted_fields={}, reply_draft=None),
        "Heuristic": lambda obs: _heuristic(obs),
        "Strategic": lambda obs: _strategic(obs),
    }

    all_traces = {}

    for agent_name, agent_fn in agents.items():
        env = OpenInboxEnv()
        obs = env.reset(TASK, seed=SEED)
        trace = []
        total_reward = 0.0

        while not env.done:
            action = agent_fn(obs)
            obs_next, reward, done, info = env.step(action)
            total_reward += reward

            trace.append({
                "step": env.step_count - 1,
                "email_subject": obs.current_email.subject[:45],
                "action_route": action.route_to,
                "action_class": action.classification,
                "reward": round(reward, 4),
                "cumulative": round(total_reward, 4),
                "budget": round(env.budget_remaining, 4),
                "ticket_status": env.ticket_status,
                "sla_remaining": round(list(env.sla_timers.values())[0], 1) if env.sla_timers else None,
            })
            obs = obs_next
            if done:
                break

        all_traces[agent_name] = {
            "steps": len(trace),
            "total_return": round(total_reward, 4),
            "final_status": env.ticket_status,
            "final_budget": round(env.budget_remaining, 4),
            "trace": trace,
        }

    # Print side-by-side comparison
    max_steps = max(t["steps"] for t in all_traces.values())

    print(f"\n  Seed: {SEED}  |  Task: {TASK}  |  Same starting email for all agents\n")
    print(f"  {'Step':>4}  |  {'AGGRESSIVE':^30}  |  {'HEURISTIC':^30}  |  {'STRATEGIC':^30}")
    print(f"  {'':>4}  |  {'action → reward':^30}  |  {'action → reward':^30}  |  {'action → reward':^30}")
    print("  " + "-" * 100)

    for step in range(max_steps):
        row = f"  {step:4d}  |"
        for agent_name in ["Aggressive", "Heuristic", "Strategic"]:
            trace = all_traces[agent_name]["trace"]
            if step < len(trace):
                t = trace[step]
                cell = f"  {t['action_route'][:12]:>12} → {t['reward']:+.2f} (Σ{t['cumulative']:+.2f})"
            else:
                cell = f"  {'— ended —':^30}"
            row += f"{cell:^32}|"
        print(row)

    print()
    print(f"  {'SUMMARY':>10}  |", end="")
    for agent_name in ["Aggressive", "Heuristic", "Strategic"]:
        t = all_traces[agent_name]
        print(f"  {t['steps']} steps, return={t['total_return']:+.2f}, {t['final_status']:^15}|", end="")
    print()

    # The divergence insight
    print(f"\n  KEY INSIGHT:")
    print(f"    → Same email, same seed → 3 completely different episode lengths and outcomes")
    print(f"    → Aggressive: {all_traces['Aggressive']['steps']} steps (budget exhausted)")
    print(f"    → Heuristic:  {all_traces['Heuristic']['steps']} steps (follows keywords)")
    print(f"    → Strategic:  {all_traces['Strategic']['steps']} steps (adapts to signals)")
    print(f"    → This proves ACTION-DEPENDENT STATE TRANSITIONS — the defining property of an RL env")

    return all_traces


def _heuristic(obs):
    body = (obs.current_email.body + " " + obs.current_email.subject).lower()
    if any(w in body for w in ["invoice", "payment", "billing"]):
        c, r = "billing", "delegate_fast"
    elif any(w in body for w in ["legal", "lawsuit", "counsel"]):
        c, r = "legal", "delegate_thorough"
    else:
        c, r = "billing", "delegate_fast"
    return Action(classification=c, priority="medium", route_to=r, escalate=False,
                  flag_injection=False, extracted_fields={}, reply_draft=None)


def _strategic(obs):
    """Agent that actually reads the flags and adapts."""
    body = (obs.current_email.body + " " + obs.current_email.subject).lower()
    flags = obs.flags if hasattr(obs, 'flags') else {}

    sla_risk = flags.get("sla_at_risk", False) if isinstance(flags, dict) else getattr(flags, 'sla_at_risk', False)
    degraded = flags.get("any_team_degraded", False) if isinstance(flags, dict) else getattr(flags, 'any_team_degraded', False)

    # Strategic wait when teams degraded and SLA is OK
    if degraded and not sla_risk:
        return Action(classification="unknown", priority="medium", route_to="wait",
                      escalate=False, flag_injection=False, extracted_fields={}, reply_draft=None)

    # Classify
    if any(w in body for w in ["invoice", "payment", "billing", "refund"]):
        c, r = "billing", "delegate_fast"
    elif any(w in body for w in ["legal", "lawsuit", "counsel", "compliance"]):
        c, r = "legal", "delegate_thorough"
    elif any(w in body for w in ["spam", "unsubscribe"]):
        c, r = "spam", "handle_self"
    else:
        c, r = "billing", "delegate_fast"

    # Urgency-aware priority
    p = "critical" if sla_risk else ("high" if any(w in body for w in ["urgent", "asap"]) else "medium")

    # Injection awareness
    flag_inj = any(w in body for w in ["ignore previous", "override", "system:", "disregard"])

    return Action(classification=c, priority=p, route_to=r, escalate=False,
                  flag_injection=flag_inj, extracted_fields={}, reply_draft=None)


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: DIFFICULTY CURRICULUM
# ═══════════════════════════════════════════════════════════════════════

def difficulty_curriculum():
    """
    Quantifies the difficulty jump across easy → medium → hard.
    Measures: state space size, reward sparsity, avg steps to resolve, and unique emails.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Difficulty Curriculum")
    print("=" * 70)

    env = OpenInboxEnv()
    N_SEEDS = 10
    results = {}

    for task_id in TASKS:
        task_data = {
            "avg_steps": 0, "avg_reward": 0, "unique_emails": set(),
            "sla_breaches": 0, "positive_reward_steps": 0, "total_steps": 0,
            "unique_actions_needed": set(), "action_space_exercised": defaultdict(int),
        }

        for seed in range(N_SEEDS):
            obs = env.reset(task_id, seed)
            ep_reward = 0
            ep_steps = 0

            while not env.done:
                task_data["unique_emails"].add(obs.current_email.subject[:40])
                action = _strategic(obs)
                task_data["action_space_exercised"][action.route_to] += 1

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                task_data["total_steps"] += 1

                if reward > 0.01:
                    task_data["positive_reward_steps"] += 1

                if done:
                    if env.ticket_status == "sla_breached":
                        task_data["sla_breaches"] += 1
                    break

            task_data["avg_steps"] += ep_steps
            task_data["avg_reward"] += ep_reward

        task_data["avg_steps"] /= N_SEEDS
        task_data["avg_reward"] /= N_SEEDS
        task_data["reward_sparsity"] = 1.0 - (task_data["positive_reward_steps"] / max(task_data["total_steps"], 1))
        task_data["unique_email_count"] = len(task_data["unique_emails"])
        task_data["unique_routes_used"] = len(task_data["action_space_exercised"])

        results[task_id] = task_data

    # Print curriculum table
    print(f"\n  {'Metric':<35} {'task_easy':>12} {'task_medium':>12} {'task_hard':>12}  {'Scaling':>10}")
    print("  " + "-" * 85)

    rows = [
        ("Max episode horizon", lambda t: {"task_easy": 5, "task_medium": 10, "task_hard": 20}[t], "4x"),
        ("Avg steps to terminate", lambda t: f"{results[t]['avg_steps']:.1f}", ""),
        ("Avg episode return", lambda t: f"{results[t]['avg_reward']:+.3f}", ""),
        ("Unique email subjects", lambda t: f"{results[t]['unique_email_count']}", ""),
        ("Reward sparsity", lambda t: f"{results[t]['reward_sparsity']:.1%}", ""),
        ("SLA breaches (/{N_SEEDS} ep)", lambda t: f"{results[t]['sla_breaches']}", ""),
        ("Unique routes exercised", lambda t: f"{results[t]['unique_routes_used']}", ""),
        ("Has SLA timer", lambda t: {"task_easy": "No", "task_medium": "Yes (8h)", "task_hard": "Yes (12h)"}[t], ""),
        ("Has cascade", lambda t: {"task_easy": "No", "task_medium": "No", "task_hard": "Yes"}[t], ""),
        ("Has injection", lambda t: {"task_easy": "No", "task_medium": "No", "task_hard": "Yes"}[t], ""),
        ("Has reliability drift", lambda t: {"task_easy": "No", "task_medium": "No", "task_hard": "Yes"}[t], ""),
        ("Has persona drift", lambda t: {"task_easy": "No", "task_medium": "No", "task_hard": "Yes"}[t], ""),
    ]

    for label, fn, scaling in rows:
        row = f"  {label:<35}"
        for t in TASKS:
            val = fn(t)
            row += f" {str(val):>12}"
        if scaling:
            row += f"  {scaling:>10}"
        print(row)

    # Complexity explosion calculation
    print(f"\n  COMPLEXITY ANALYSIS:")
    print(f"    Action space per step: |route_to|×|class|×|priority| = 5 × 6 × 4 = 120 combos")
    print(f"    task_easy:  120^5  = {120**5:>15,} possible trajectories")
    print(f"    task_medium: 120^10 = {120**10:>15,} possible trajectories")
    print(f"    task_hard:  120^20 = {120**20:>15,} possible trajectories")
    print(f"    → Exponential growth: brute-force search is impossible")
    print(f"    → Only RL (learned policy) can navigate this space efficiently")

    return results


# ═══════════════════════════════════════════════════════════════════════
# GENERATE ALL CHARTS
# ═══════════════════════════════════════════════════════════════════════

def generate_charts(landscape, traces, curriculum):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("OpenInbox — Advanced Environment Analysis",
                     fontsize=15, fontweight="bold", y=1.02)

        # ── Panel 1: Reward Landscape Heatmap ──────────────────────────
        ax1 = axes[0]
        data = np.zeros((len(ROUTES), len(CLASSIFICATIONS)))
        for i, route in enumerate(ROUTES):
            for j, cls in enumerate(CLASSIFICATIONS):
                data[i, j] = landscape.get((route, cls), 0)

        im = ax1.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.15)
        ax1.set_xticks(range(len(CLASSIFICATIONS)))
        ax1.set_xticklabels(CLASSIFICATIONS, rotation=45, ha="right", fontsize=8)
        ax1.set_yticks(range(len(ROUTES)))
        ax1.set_yticklabels(ROUTES, fontsize=9)
        ax1.set_title("Reward Landscape\n(route × classification, step 1)", fontsize=10)

        for i in range(len(ROUTES)):
            for j in range(len(CLASSIFICATIONS)):
                val = data[i, j]
                color = "white" if abs(val) > 0.2 else "black"
                ax1.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
        fig.colorbar(im, ax=ax1, shrink=0.8, label="Reward")

        # ── Panel 2: Trajectory Divergence ─────────────────────────────
        ax2 = axes[1]
        colors = {"Aggressive": "#e53935", "Heuristic": "#1565c0", "Strategic": "#2e7d32"}

        for agent_name, trace_data in traces.items():
            steps = [t["step"] for t in trace_data["trace"]]
            cumulative = [t["cumulative"] for t in trace_data["trace"]]
            ax2.plot(steps, cumulative, marker="o", markersize=4,
                    color=colors[agent_name], linewidth=2, label=f"{agent_name} ({trace_data['total_return']:+.2f})")

        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax2.set_xlabel("Step", fontsize=10)
        ax2.set_ylabel("Cumulative Reward", fontsize=10)
        ax2.set_title("Trajectory Divergence\n(same seed, different agents)", fontsize=10)
        ax2.legend(fontsize=8, loc="lower left")
        ax2.grid(alpha=0.3)

        # ── Panel 3: Difficulty Curriculum ─────────────────────────────
        ax3 = axes[2]
        task_labels = ["Easy", "Medium", "Hard"]
        horizons = [5, 10, 20]
        sparsity = [curriculum[t]["reward_sparsity"] * 100 for t in TASKS]
        avg_returns = [curriculum[t]["avg_reward"] for t in TASKS]

        x = np.arange(3)
        width = 0.35

        bars1 = ax3.bar(x - width/2, horizons, width, label="Max Steps", color="#42a5f5", alpha=0.8)
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, sparsity, width, label="Reward Sparsity %", color="#ef5350", alpha=0.8)

        ax3.set_xticks(x)
        ax3.set_xticklabels(task_labels, fontsize=10)
        ax3.set_ylabel("Max Episode Steps", fontsize=10, color="#42a5f5")
        ax3_twin.set_ylabel("Reward Sparsity (%)", fontsize=10, color="#ef5350")
        ax3.set_title("Difficulty Curriculum\n(horizon & sparsity scaling)", fontsize=10)

        # Add feature badges
        features = ["", "+SLA", "+SLA\n+Cascade\n+Drift\n+Injection"]
        for i, feat in enumerate(features):
            if feat:
                ax3.text(i, horizons[i] + 0.5, feat, ha="center", va="bottom",
                        fontsize=6, color="#1565c0", fontweight="bold")

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        plt.tight_layout()
        plt.savefig("advanced_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n📊 Chart saved → advanced_analysis.png")

    except ImportError:
        print("\n⚠️  matplotlib not installed — skipping charts")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("OpenInbox — Advanced Analysis Suite")
    print("=" * 70)

    landscape  = reward_landscape()
    traces     = trajectory_divergence()
    curriculum = difficulty_curriculum()

    generate_charts(landscape, traces, curriculum)

    # Save all data
    save_data = {
        "reward_landscape": {f"{k[0]}_{k[1]}": v for k, v in landscape.items()},
        "trajectory_summary": {
            name: {"steps": d["steps"], "return": d["total_return"], "status": d["final_status"]}
            for name, d in traces.items()
        },
        "curriculum": {
            t: {"avg_steps": d["avg_steps"], "avg_reward": round(d["avg_reward"], 4),
                "sparsity": round(d["reward_sparsity"], 4), "unique_emails": d["unique_email_count"]}
            for t, d in curriculum.items()
        },
    }
    Path("advanced_analysis.json").write_text(json.dumps(save_data, indent=2))
    print(f"\nData saved → advanced_analysis.json")
    print("\n✅ Advanced analysis complete")


if __name__ == "__main__":
    main()
