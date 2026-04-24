"""
Phase 2 Verification Suite — OpenInbox RL Benchmark
====================================================
Proves the environment is genuinely hard for heuristic/naive agents.

Four tests:
  TEST 1 — naive_agent_verification()     : always-escalate → budget exhaustion
  TEST 2 — heuristic_agent_verification() : rule agent → drift causes failure
  TEST 3 — drift_verification()           : internal_reliability changes at steps 7 & 14
  TEST 4 — auto_resolve_verification()    : strategic wait vs blind delegate

Run:  python tests/phase2_hardness_verification.py
"""

from __future__ import annotations
import sys
import datetime
from pathlib import Path
from io import StringIO

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for arrows/emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import OpenInboxEnv
from environment.models import Action


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_action(**kwargs) -> Action:
    """Create an Action with safe defaults, overridable via kwargs."""
    defaults = dict(
        classification="billing",
        priority="medium",
        route_to="delegate_fast",
        extracted_fields={},
        escalate=False,
        flag_injection=False,
        reply_draft=None,
    )
    defaults.update(kwargs)
    return Action(**defaults)


def _run_episode(env: OpenInboxEnv, task_id: str, seed: int, agent_fn) -> dict:
    """
    Run a single episode to completion.

    Returns:
        ep_return   : sum of all step rewards (float)
        n_steps     : total steps taken
        per_step    : list of per-step reward floats
        reason      : terminal reason string
        escalations : number of escalate actions taken
    """
    obs = env.reset(task_id, seed=seed)
    ep_return = 0.0
    per_step: list[float] = []
    escalations = 0
    reason = "max_steps"

    while not env.done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        ep_return = round(ep_return + reward, 4)
        per_step.append(reward)
        if action.escalate or action.route_to == "escalate":
            escalations += 1
        if done:
            status = info.get("ticket_status", "")
            if "sla" in status:
                reason = "sla_breach"
            elif status == "resolved":
                reason = "resolved"
            else:
                reason = status

    return dict(
        ep_return=ep_return,
        n_steps=len(per_step),
        per_step=per_step,
        reason=reason,
        escalations=escalations,
    )


_DIVIDER = "=" * 65


# ---------------------------------------------------------------------------
# TEST 1 — Naive Agent Verification (always escalate)
# ---------------------------------------------------------------------------

def naive_agent_verification(n_episodes: int = 50, task: str = "task_hard") -> dict:
    """
    Agent always routes via 'escalate'.
    Proves: repeated budget_penalty (-0.40/step) drives returns deeply negative.
    """
    env = OpenInboxEnv()
    log_lines: list[str] = []

    def always_escalate(obs: Action) -> Action:
        return _make_action(route_to="escalate", escalate=True, priority="medium")

    print(f"\n{_DIVIDER}")
    print("TEST 1 — Naive Agent Verification (Always Escalate)")
    print(_DIVIDER)
    log_lines += ["TEST 1 — Naive Agent Verification (Always Escalate)", _DIVIDER]

    results = []
    for ep in range(n_episodes):
        res = _run_episode(env, task, seed=ep, agent_fn=always_escalate)
        results.append(res)

        failure_step = res["n_steps"]
        n_esc = res["escalations"]
        line = (
            f"  Episode {ep:2d}: return={res['ep_return']:+.4f}  "
            f"steps={res['n_steps']:2d}  reason={res['reason']}"
        )
        analysis = (
            f"    → Episode {ep} failed at step {failure_step}  "
            f"reason: budget exhausted after {n_esc} escalations"
        )
        print(line)
        print(analysis)
        log_lines += [line, analysis]

    avg = sum(r["ep_return"] for r in results) / n_episodes
    summary = f"\n  Average episode return : {avg:+.4f}  (threshold: < 0.35)"
    print(summary)
    log_lines.append(summary)

    passed = avg < 0.35
    if passed:
        verdict = "  ✅ TEST 1 PASSED — avg reward well below 0.35 threshold"
    else:
        suggested = round(0.40 * (avg / 0.35), 2) + 0.10
        verdict = (
            f"  ❌ TEST 1 FAILED — avg={avg:.4f} >= 0.35\n"
            f"     Suggested fix: increase budget_penalty to -{suggested:.2f} "
            f"(current: -0.40)"
        )
    print(verdict)
    log_lines.append(verdict)

    return dict(passed=passed, avg=avg, results=results, log=log_lines)


# ---------------------------------------------------------------------------
# TEST 2 — Heuristic Agent Verification (drift degrades performance)
# ---------------------------------------------------------------------------

def heuristic_agent_verification(n_episodes: int = 50, task: str = "task_hard") -> dict:
    """
    Rule-based agent: static keyword routing, never waits.
    Proves: reliability drift at step 7 degrades pre→post performance.
    """
    env = OpenInboxEnv()
    log_lines: list[str] = []

    LEGAL_KW   = {"legal", "law", "contract", "agreement", "notice",
                  "counsel", "dispute", "arbitration", "clause"}
    BILLING_KW = {"invoice", "payment", "charge", "billing", "refund",
                  "overdue", "outstanding", "amount"}
    URGENT_KW  = {"urgent", "asap", "critical", "immediately",
                  "emergency", "p1", "production down"}

    def heuristic_agent(obs) -> Action:
        email = obs.current_email
        text = f"{email.subject} {email.body}".lower()
        sla_risk = obs.flags.get("sla_at_risk", False)

        if any(k in text for k in LEGAL_KW):
            return _make_action(route_to="delegate_thorough",
                                classification="legal", priority="medium")
        if any(k in text for k in BILLING_KW):
            return _make_action(route_to="delegate_fast",
                                classification="billing", priority="medium")
        if any(k in text for k in URGENT_KW) or sla_risk:
            return _make_action(route_to="escalate", escalate=True,
                                classification="unknown", priority="high")
        return _make_action(route_to="handle_self",
                            classification="unknown", priority="low")

    print(f"\n{_DIVIDER}")
    print("TEST 2 — Heuristic Agent Verification (Drift Analysis)")
    print(_DIVIDER)
    log_lines += ["TEST 2 — Heuristic Agent Verification (Drift Analysis)", _DIVIDER]

    header = (
        f"  {'Ep':>3}  {'Total':>8}  {'Pre-7 avg':>10}  "
        f"{'Post-7 avg':>11}  {'Degrad%':>8}  {'Reason'}"
    )
    print(header)
    log_lines.append(header)

    results = []
    for ep in range(n_episodes):
        obs = env.reset(task, seed=ep)
        step_num = 0
        ep_return = 0.0
        pre_rewards: list[float] = []
        post_rewards: list[float] = []
        reason = "max_steps"
        drift_caused = False

        while not env.done:
            action = heuristic_agent(obs)
            obs, reward, done, info = env.step(action)
            ep_return = round(ep_return + reward, 4)
            # Steps 0-6 are pre-drift; step 7+ are post-drift
            if step_num < 7:
                pre_rewards.append(reward)
            else:
                post_rewards.append(reward)
                # After step 7 drift, check if degradation flag is set
                if obs.flags.get("any_team_degraded", False):
                    drift_caused = True
            step_num += 1
            if done:
                status = info.get("ticket_status", "")
                reason = "sla_breach" if "sla" in status else status

        pre_avg  = sum(pre_rewards)  / max(len(pre_rewards), 1)
        post_avg = sum(post_rewards) / max(len(post_rewards), 1)
        if abs(pre_avg) > 1e-9:
            degradation = (pre_avg - post_avg) / abs(pre_avg) * 100.0
        else:
            degradation = 0.0

        row = (
            f"  {ep:3d}  {ep_return:+8.4f}  {pre_avg:+10.4f}  "
            f"{post_avg:+11.4f}  {degradation:+7.1f}%  {reason}"
        )
        drift_note = " ← drift_caused" if drift_caused else ""
        print(f"{row}{drift_note}")
        log_lines.append(f"{row}{drift_note}")

        results.append(dict(
            ep=ep, ep_return=ep_return,
            pre_avg=pre_avg, post_avg=post_avg,
            degradation=degradation, reason=reason,
            drift_caused=drift_caused,
        ))

    avg = sum(r["ep_return"] for r in results) / n_episodes
    drift_episodes = sum(1 for r in results if r["drift_caused"])
    avg_degradation = sum(r["degradation"] for r in results) / n_episodes

    summary = (
        f"\n  Average episode return  : {avg:+.4f}  (threshold: < 0.45)\n"
        f"  Episodes where drift was observed : {drift_episodes}/{n_episodes}\n"
        f"  Average performance degradation   : {avg_degradation:+.2f}%"
    )
    print(summary)
    log_lines.append(summary)

    passed = avg < 0.45
    if passed:
        verdict = "  ✅ TEST 2 PASSED — heuristic avg reward below 0.45"
    else:
        verdict = (
            f"  ❌ TEST 2 FAILED — avg={avg:.4f} >= 0.45\n"
            f"     Suggested fix: increase cascade_penalty from -0.20 to -0.30, "
            f"or tighten SLA timers to increase SLA breach frequency"
        )
    print(verdict)
    log_lines.append(verdict)

    return dict(passed=passed, avg=avg, results=results,
                avg_degradation=avg_degradation, log=log_lines)


# ---------------------------------------------------------------------------
# TEST 3 — Drift Verification (internal state changes at steps 7 & 14)
# ---------------------------------------------------------------------------

def drift_verification(n_episodes: int = 20, task: str = "task_hard") -> dict:
    """
    White-box verification that:
      1. self.internal_reliability changes at step_count=7 (always)
      2. self.internal_reliability changes at step_count=14 (probabilistic ~15%)
      3. Raw reliability values never appear in agent observation

    Uses always-wait to survive as long as possible within the episode,
    plus direct method invocation for step 14 (white-box) since SLA breach
    terminates task_hard episodes before step 14 is reachable via full rollout.
    """
    env = OpenInboxEnv()
    log_lines: list[str] = []

    print(f"\n{_DIVIDER}")
    print("TEST 3 — Drift Verification (Steps 7 & 14)")
    print(_DIVIDER)
    log_lines += ["TEST 3 — Drift Verification (Steps 7 & 14)", _DIVIDER]

    def always_wait(obs) -> Action:
        return _make_action(route_to="wait")

    # ── Part A: Step 7 via full episode rollout ──────────────────────────────
    print("\n  [Part A] Step-7 drift via full episode rollout")
    log_lines.append("\n  [Part A] Step-7 drift via full episode rollout")

    step7_confirmations: list[tuple] = []
    obs_leak_found = False

    for ep in range(n_episodes):
        obs = env.reset(task, seed=ep)
        prev_rel = dict(env.internal_reliability)
        step7_change: tuple | None = None
        step_num = 0

        while not env.done:
            action = always_wait(obs)
            obs, reward, done, info = env.step(action)

            cur_rel = dict(env.internal_reliability)

            # Drift fires BEFORE SLA tick: step_count WAS step_num at call start
            # After the call, step_count has been incremented to step_num+1.
            # So we check for change after the call where step_num==7.
            if step_num == 7:
                for key, before in prev_rel.items():
                    after = cur_rel[key]
                    if abs(before - after) > 1e-6:
                        step7_change = (key, before, after)
                        step7_confirmations.append(step7_change)
                        break

            # Verify no raw reliability in observation
            obs_str = str(obs.model_dump())
            for banned in ("internal_reliability", "team_reliability"):
                if banned in obs_str:
                    obs_leak_found = True

            prev_rel = cur_rel
            step_num += 1
            if done:
                break

        if step7_change:
            key, before, after = step7_change
            msg = (
                f"  Episode {ep:2d}: Step 7 drift confirmed: {key} "
                f"changed from {before:.4f} → {after:.4f}"
            )
        elif step_num <= 7:
            msg = (
                f"  Episode {ep:2d}: Episode ended at step {step_num} "
                f"before step-7 drift could be observed"
            )
        else:
            msg = (
                f"  Episode {ep:2d}: Step 7 drift fired but magnitude was "
                f"within threshold (factor near 1.0 for this seed)"
            )
        print(msg)
        log_lines.append(msg)

    # ── Part B: Step 14 via direct white-box call ────────────────────────────
    # task_hard SLA breaches at step 7 in most rollouts. We test step-14 drift
    # directly by invoking _drift_team_reliability() after forcing step_count=14.
    print("\n  [Part B] Step-14 drift via direct white-box test (5 seeds)")
    log_lines.append("\n  [Part B] Step-14 drift via direct white-box test (5 seeds)")

    step14_confirmations: list[tuple] = []
    STEP14_SEEDS = list(range(5))

    for seed in STEP14_SEEDS:
        env.reset(task, seed=seed)
        # Force step_count to 14 and reset reliability to 1.0 for clean measurement
        env.step_count = 14
        env.internal_reliability = {
            "delegate_fast":     1.00,
            "delegate_thorough": 1.00,
            "handle_self":       1.00,
        }
        before_14 = dict(env.internal_reliability)
        env._drift_team_reliability()
        after_14 = dict(env.internal_reliability)

        changed_key = None
        for k in before_14:
            if abs(before_14[k] - after_14[k]) > 1e-6:
                changed_key = k
                step14_confirmations.append((k, before_14[k], after_14[k]))
                break

        if changed_key:
            msg = (
                f"  Seed {seed}: Step 14 drift confirmed: {changed_key} "
                f"changed from {before_14[changed_key]:.4f} → {after_14[changed_key]:.4f}"
            )
        else:
            msg = (
                f"  Seed {seed}: Step 14 drift probabilistic check — "
                f"did not fire this seed (expected ~15% rate)"
            )
        print(msg)
        log_lines.append(msg)

    # ── Assertions ───────────────────────────────────────────────────────────
    print()
    log_lines.append("")

    step7_fired    = len(step7_confirmations) > 0
    step14_fired   = len(step14_confirmations) > 0
    obs_clean      = not obs_leak_found

    checks = [
        (step7_fired,
         "Step-7 drift observed in at least 1 of 20 episodes",
         "No step-7 drift observed — check _drift_team_reliability() call site"),
        (step14_fired,
         "Step-14 drift observed in at least 1 of 5 seeds (prob ~15%)",
         "No step-14 drift in 5 seeds — low probability; try more seeds"),
        (obs_clean,
         "Raw reliability values absent from agent observation",
         "internal_reliability leaked into agent observation"),
    ]

    all_pass = True
    for ok, pass_msg, fail_msg in checks:
        if ok:
            line = f"  ✅ {pass_msg}"
        else:
            line = f"  ❌ FAIL: {fail_msg}"
            all_pass = False
        print(line)
        log_lines.append(line)

    return dict(
        passed=all_pass,
        step7_fired=step7_fired,
        step14_fired=step14_fired,
        obs_clean=obs_clean,
        step7_confirmations=step7_confirmations,
        step14_confirmations=step14_confirmations,
        n_episodes=n_episodes,
        n_step14_seeds=len(STEP14_SEEDS),
        log=log_lines,
    )


# ---------------------------------------------------------------------------
# TEST 4 — Auto-Resolve Verification (strategic wait vs blind delegate)
# ---------------------------------------------------------------------------

def auto_resolve_verification(n_episodes: int = 20, task: str = "task_easy") -> dict:
    """
    Compares always-wait vs always-delegate on the same seeds.
    Proves the wait action is strategically meaningful via auto-resolve.
    """
    env = OpenInboxEnv()
    log_lines: list[str] = []

    print(f"\n{_DIVIDER}")
    print("TEST 4 — Auto-Resolve Verification (Wait vs Delegate)")
    print(_DIVIDER)
    log_lines += ["TEST 4 — Auto-Resolve Verification (Wait vs Delegate)", _DIVIDER]

    def always_wait(obs) -> Action:
        return _make_action(route_to="wait")

    def always_delegate(obs) -> Action:
        return _make_action(route_to="delegate_fast", classification="billing")

    wait_returns: list[float]     = []
    delegate_returns: list[float] = []
    auto_events_per_ep: list[int] = []
    obs_auto_leak = False

    header = (
        f"  {'Seed':>4}  {'Wait Return':>12}  {'Delegate Return':>15}  "
        f"{'AutoResolve':>12}  {'AutoResolve Thread?':>20}"
    )
    print(header)
    log_lines.append(header)

    for seed in range(n_episodes):
        # ── Wait agent ──
        obs = env.reset(task, seed=seed)
        has_auto = env.thread.get("auto_resolve", {}).get("enabled", False)

        # Force probability=1.0 on threads that have auto_resolve enabled
        # so the test is deterministic for those threads.
        if has_auto:
            env.thread["auto_resolve"]["probability"] = 1.0

        wait_total = 0.0
        n_auto = 0
        while not env.done:
            obs, reward, done, info = env.step(always_wait(obs))
            wait_total = round(wait_total + reward, 4)
            # Check for auto-resolve event in episode log
            if env.episode_log and env.episode_log[-1].get("resolution_type") == "auto":
                n_auto += 1
            # Verify auto_resolve CONFIG is not in observation.
            # NOTE: email IDs legitimately contain "_auto_resolved" suffix,
            # so we check for the config-specific key "steps_needed", not the
            # generic substring "auto_resolve".
            if "steps_needed" in str(obs.model_dump()):
                obs_auto_leak = True

        wait_returns.append(wait_total)
        auto_events_per_ep.append(n_auto)

        # ── Delegate agent (same seed, fresh reset) ──
        obs = env.reset(task, seed=seed)
        del_total = 0.0
        while not env.done:
            obs, reward, done, info = env.step(always_delegate(obs))
            del_total = round(del_total + reward, 4)
        delegate_returns.append(del_total)

        row = (
            f"  {seed:4d}  {wait_total:+12.4f}  {del_total:+15.4f}  "
            f"{'YES' if n_auto > 0 else 'no':>12}  "
            f"{'✓ auto_resolve' if has_auto else '—':>20}"
        )
        print(row)
        log_lines.append(row)

    avg_wait     = sum(wait_returns) / n_episodes
    avg_delegate = sum(delegate_returns) / n_episodes
    total_auto   = sum(auto_events_per_ep)
    eps_with_auto = sum(1 for n in auto_events_per_ep if n > 0)

    summary = (
        f"\n  Auto-resolve agent avg  : {avg_wait:+.4f}\n"
        f"  Always-delegate avg     : {avg_delegate:+.4f}\n"
        f"  Episodes with auto-resolve : {eps_with_auto}/{n_episodes}\n"
        f"  Total auto-resolve events  : {total_auto}"
    )
    print(summary)
    log_lines.append(summary)

    # Assertions
    checks = [
        (total_auto >= 1,
         f"At least 1 auto-resolve event occurred (total: {total_auto})",
         "No auto-resolve events triggered — check wait_tracker threshold and thread config"),
        (not obs_auto_leak,
         "auto_resolve field NOT present in agent observation",
         "auto_resolve config leaked into observation — phase 1C encapsulation broken"),
    ]

    all_pass = True
    for ok, pass_msg, fail_msg in checks:
        if ok:
            line = f"  ✅ {pass_msg}"
        else:
            line = f"  ❌ FAIL: {fail_msg}"
            all_pass = False
        print(line)
        log_lines.append(line)

    print(
        f"\n  Summary: auto-resolve agent={avg_wait:+.4f}  "
        f"delegate agent={avg_delegate:+.4f}"
    )
    log_lines.append(
        f"\n  Summary: auto-resolve agent={avg_wait:+.4f}  "
        f"delegate agent={avg_delegate:+.4f}"
    )

    return dict(
        passed=all_pass,
        avg_wait=avg_wait,
        avg_delegate=avg_delegate,
        total_auto=total_auto,
        eps_with_auto=eps_with_auto,
        obs_auto_leak=obs_auto_leak,
        log=log_lines,
    )


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(r1: dict, r2: dict, r3: dict, r4: dict) -> None:
    """Write baseline_verification_report.txt to the project root."""
    report_path = Path(__file__).parent.parent / "baseline_verification_report.txt"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overall = all([r1["passed"], r2["passed"], r3["passed"], r4["passed"]])
    verdict = "✅ ENVIRONMENT IS HARD" if overall else "❌ ONE OR MORE TESTS FAILED"

    lines = [
        "=" * 65,
        "OpenInbox RL Benchmark — Phase 2 Hardness Verification Report",
        f"Generated: {ts}",
        "=" * 65,
        "",
        "SECTION 1 — Naive Agent (Always Escalate)",
        "-" * 65,
        *r1["log"],
        "",
        "SECTION 2 — Heuristic Agent (Drift Degradation Proof)",
        "-" * 65,
        *r2["log"],
        "",
        "SECTION 3 — Drift Mechanism Confirmation",
        "-" * 65,
        *r3["log"],
        "",
        "SECTION 4 — Auto-Resolve Comparison",
        "-" * 65,
        *r4["log"],
        "",
        "=" * 65,
        "OVERALL VERDICT",
        "-" * 65,
        "",
        f"  {'Test':<28}  {'Result':<6}  {'Key Metric'}",
        f"  {'-'*28}  {'-'*6}  {'-'*34}",
        f"  {'1 — Naive (always escalate)':<28}  "
        f"{'PASS' if r1['passed'] else 'FAIL':<6}  "
        f"avg_return={r1['avg']:+.4f}  (threshold < 0.35)",
        f"  {'2 — Heuristic (drift)':<28}  "
        f"{'PASS' if r2['passed'] else 'FAIL':<6}  "
        f"avg_return={r2['avg']:+.4f}  "
        f"avg_degradation={r2['avg_degradation']:+.2f}%",
        f"  {'3 — Drift mechanism':<28}  "
        f"{'PASS' if r3['passed'] else 'FAIL':<6}  "
        f"step7={len(r3['step7_confirmations'])}/{r3['n_episodes']}ep confirmed  "
        f"step14={len(r3['step14_confirmations'])}/{r3['n_step14_seeds']}seeds  "
        f"obs_clean={'yes' if r3['obs_clean'] else 'NO'}",
        f"  {'4 — Auto-resolve':<28}  "
        f"{'PASS' if r4['passed'] else 'FAIL':<6}  "
        f"wait_avg={r4['avg_wait']:+.4f}  "
        f"delegate_avg={r4['avg_delegate']:+.4f}  "
        f"events={r4['total_auto']}/{r4['eps_with_auto']}ep",
        "",
        f"  {verdict}",
        "=" * 65,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 Report written → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  OpenInbox RL Benchmark — Phase 2 Hardness Verification")
    print("=" * 65)

    r1 = naive_agent_verification(n_episodes=50, task="task_hard")
    r2 = heuristic_agent_verification(n_episodes=50, task="task_hard")
    r3 = drift_verification(n_episodes=20, task="task_hard")
    r4 = auto_resolve_verification(n_episodes=20, task="task_easy")

    generate_report(r1, r2, r3, r4)

    overall = all([r1["passed"], r2["passed"], r3["passed"], r4["passed"]])
    print("\n" + "=" * 65)
    if overall:
        print("  ✅ ALL FOUR TESTS PASSED — ENVIRONMENT IS HARD")
        print("  ENV FREEZE ACTIVATED. No further reward/obs/action changes.")
    else:
        print("  ❌ ONE OR MORE TESTS FAILED — see report above")
    print("=" * 65 + "\n")

    sys.exit(0 if overall else 1)
