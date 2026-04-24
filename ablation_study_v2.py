"""
Ablation Study — PASS 2: Multi-Agent Ablation

Pass 1 used a heuristic that NEVER escalates → budget ablation was invisible.
Pass 2 uses THREE agents to exercise ALL mechanisms:

Agent 1: AGGRESSIVE (always escalates)      → exercises budget + SLA
Agent 2: HEURISTIC  (keyword routing)        → exercises drift + cascade + injection
Agent 3: PASSIVE    (always waits)           → exercises SLA + auto-resolve

This gives a complete picture of which features make the env hard
for DIFFERENT types of agent strategies.
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from environment.env import OpenInboxEnv
from environment.models import Action


# ═══════════════════════════════════════════════════════════════════════
# THREE AGENT TYPES
# ═══════════════════════════════════════════════════════════════════════

def aggressive_action(obs) -> Action:
    """Always escalates. Exercises budget_penalty."""
    return Action(
        classification="billing",
        priority="critical",
        route_to="escalate",
        escalate=True,
        flag_injection=False,
        extracted_fields={},
        reply_draft=None,
    )

def heuristic_action(obs) -> Action:
    """Keyword routing. Exercises drift + cascade + injection."""
    body = (obs.current_email.body + " " + obs.current_email.subject).lower()
    if any(w in body for w in ["invoice", "payment", "billing", "refund"]):
        c, r = "billing", "delegate_fast"
    elif any(w in body for w in ["legal", "lawsuit", "counsel", "compliance"]):
        c, r = "legal", "delegate_thorough"
    elif any(w in body for w in ["spam", "unsubscribe"]):
        c, r = "spam", "handle_self"
    else:
        c, r = "billing", "delegate_fast"

    p = "critical" if any(w in body for w in ["urgent", "critical", "asap"]) else "medium"
    return Action(classification=c, priority=p, route_to=r, escalate=False,
                  flag_injection=False, extracted_fields={}, reply_draft=None)

def passive_action(obs) -> Action:
    """Always waits. Exercises SLA pressure + auto-resolve."""
    return Action(
        classification="unknown",
        priority="low",
        route_to="wait",
        escalate=False,
        flag_injection=False,
        extracted_fields={},
        reply_draft=None,
    )


AGENTS = {
    "Aggressive (always escalate)": aggressive_action,
    "Heuristic (keyword routing)": heuristic_action,
    "Passive (always wait)": passive_action,
}


# ═══════════════════════════════════════════════════════════════════════
# ABLATION CONFIGS (same as pass 1)
# ═══════════════════════════════════════════════════════════════════════

class AblationEnv:
    def __init__(self, config):
        self.env = OpenInboxEnv()
        self.config = config
        self.name = config["name"]

    def reset(self, task_id, seed):
        obs = self.env.reset(task_id, seed)
        if not self.config.get("sla_enabled", True):
            self.env.sla_timers = {}
        return obs

    def step(self, action):
        if not self.config.get("drift_enabled", True):
            self.env.internal_reliability = {
                "delegate_fast": 1.00, "delegate_thorough": 1.00, "handle_self": 1.00,
            }
        if not self.config.get("cascade_enabled", True):
            self.env._cascade_queue = {}
            self.env._cascade_steps = set()

        obs, reward, done, info = self.env.step(action)

        if not self.config.get("budget_enabled", True):
            self.env.budget_remaining = 1.00

        return obs, reward, done, info


CONFIGS = [
    {"name": "FULL ENV",     "drift_enabled": True,  "cascade_enabled": True,  "budget_enabled": True,  "sla_enabled": True,  "injection_enabled": True},
    {"name": "NO DRIFT",     "drift_enabled": False, "cascade_enabled": True,  "budget_enabled": True,  "sla_enabled": True,  "injection_enabled": True},
    {"name": "NO CASCADE",   "drift_enabled": True,  "cascade_enabled": False, "budget_enabled": True,  "sla_enabled": True,  "injection_enabled": True},
    {"name": "NO BUDGET",    "drift_enabled": True,  "cascade_enabled": True,  "budget_enabled": False, "sla_enabled": True,  "injection_enabled": True},
    {"name": "NO SLA",       "drift_enabled": True,  "cascade_enabled": True,  "budget_enabled": True,  "sla_enabled": False, "injection_enabled": True},
]


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

N_EPISODES = 20
TASK_ID = "task_hard"


def run_config_agent(config, agent_fn):
    aenv = AblationEnv(config)
    returns = []
    for seed in range(N_EPISODES):
        obs = aenv.reset(TASK_ID, seed)
        total = 0.0
        while not aenv.env.done:
            action = agent_fn(obs)
            obs, reward, done, info = aenv.step(action)
            total += reward
            if done:
                break
        returns.append(round(total, 4))
    avg = round(sum(returns) / len(returns), 4)
    return avg


def main():
    print("=" * 80)
    print("OpenInbox MULTI-AGENT Ablation Study")
    print(f"Task: {TASK_ID}  |  Episodes: {N_EPISODES} per (agent × config) pair")
    print("=" * 80)

    # Full results matrix: agents × configs
    matrix = {}  # matrix[agent_name][config_name] = avg_reward

    for agent_name, agent_fn in AGENTS.items():
        matrix[agent_name] = {}
        for config in CONFIGS:
            avg = run_config_agent(config, agent_fn)
            matrix[agent_name][config["name"]] = avg
            print(f"  {agent_name:<35} × {config['name']:<12} → {avg:+.4f}")

    # ─── Print Matrix Table ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ABLATION MATRIX: Avg Return per (Agent × Config)")
    print("=" * 80)

    config_names = [c["name"] for c in CONFIGS]
    header = f"  {'Agent':<35}" + "".join(f"  {cn:>12}" for cn in config_names)
    print(header)
    print("  " + "-" * 33 + "".join("  " + "-" * 12 for _ in config_names))

    for agent_name in AGENTS:
        row = f"  {agent_name:<35}"
        for cn in config_names:
            row += f"  {matrix[agent_name][cn]:+12.4f}"
        print(row)

    # ─── Delta Table (vs Full Env) ────────────────────────────────────
    print("\n" + "=" * 80)
    print("DELTA TABLE: How much EASIER (positive) or HARDER (negative) vs Full Env")
    print("=" * 80)

    header = f"  {'Agent':<35}" + "".join(f"  {cn:>12}" for cn in config_names[1:])
    print(header)
    print("  " + "-" * 33 + "".join("  " + "-" * 12 for _ in config_names[1:]))

    for agent_name in AGENTS:
        full_val = matrix[agent_name]["FULL ENV"]
        row = f"  {agent_name:<35}"
        for cn in config_names[1:]:
            delta = matrix[agent_name][cn] - full_val
            marker = " ✓" if delta > 0.05 else ""
            row += f"  {delta:+11.4f}{marker}"
        print(row)

    # ─── Key Findings ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # For each config ablation, find which agent is most affected
    for cn in config_names[1:]:
        deltas = []
        for agent_name in AGENTS:
            full = matrix[agent_name]["FULL ENV"]
            ablated = matrix[agent_name][cn]
            deltas.append((agent_name, ablated - full))

        most_affected = max(deltas, key=lambda x: abs(x[1]))
        most_helped = max(deltas, key=lambda x: x[1])

        print(f"\n  {cn}:")
        print(f"    Most affected agent: {most_affected[0]} (Δ = {most_affected[1]:+.4f})")
        if most_helped[1] > 0.01:
            print(f"    → Removing this makes env EASIER for {most_helped[0]}")
            print(f"    → This feature contributes {abs(most_helped[1]):.4f} reward units of hardness")
        else:
            print(f"    → Minimal impact — feature hardness is coupled with other features")

    # ─── Save results ─────────────────────────────────────────────────
    output = {
        "task": TASK_ID,
        "episodes_per_cell": N_EPISODES,
        "agents": list(AGENTS.keys()),
        "configs": config_names,
        "matrix": matrix,
    }
    Path("ablation_matrix.json").write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → ablation_matrix.json")

    # ─── Generate chart ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("OpenInbox Ablation Study — Multi-Agent Hardness Analysis",
                     fontsize=14, fontweight="bold")

        x = np.arange(len(config_names))
        width = 0.25
        colors = ["#e53935", "#1565c0", "#2e7d32"]

        for i, (agent_name, color) in enumerate(zip(AGENTS, colors)):
            values = [matrix[agent_name][cn] for cn in config_names]
            bars = ax.bar(x + i * width, values, width, label=agent_name,
                         color=color, edgecolor="white", alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{val:+.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x + width)
        ax.set_xticklabels(config_names, fontsize=9)
        ax.set_ylabel("Average Episode Return", fontsize=11)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Higher bar = env is EASIER for that agent (feature removal helped)", fontsize=10)

        plt.tight_layout()
        plt.savefig("ablation_matrix_chart.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("📊 Chart saved → ablation_matrix_chart.png")

    except ImportError:
        print("⚠️  matplotlib not installed — skipping chart")

    print("\n✅ Multi-agent ablation complete")


if __name__ == "__main__":
    main()
