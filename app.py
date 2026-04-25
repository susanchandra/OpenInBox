"""OpenInbox interactive Gradio demo for HuggingFace Spaces.

Run locally:
    python app.py
"""

from __future__ import annotations

import json
import gradio as gr
import requests
import time
from dataclasses import dataclass
from typing import Any

ENV_SERVER_URL = "https://susannnnn-openinbox.hf.space"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3

TASK_ID_MAP = {
    "Easy": "task_easy",
    "Medium": "task_medium",
    "Hard": "task_hard",
}

ACTION_OPTIONS = {
    "Handle Myself": "handle_self",
    "Delegate Fast": "delegate_fast",
    "Delegate Thorough": "delegate_thorough",
    "Escalate": "escalate",
    "Wait": "wait",
}

CLASSIFICATION_OPTIONS = ["billing", "legal", "technical", "hr", "compliance", "general"]
PRIORITY_OPTIONS = ["low", "medium", "high", "critical"]


@dataclass
class ApiResult:
    ok: bool
    data: dict[str, Any] | None
    message: str
    retryable: bool


def api_call(method: str, endpoint: str, payload: dict[str, Any] | None = None) -> ApiResult:
    """Call environment API with retry-safe timeout handling."""
    url = f"{ENV_SERVER_URL}{endpoint}"
    last_error = "Unknown error"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code >= 500:
                last_error = f"Server error {response.status_code}"
                if attempt < MAX_RETRIES:
                    time.sleep(0.6 * attempt)
                    continue
                return ApiResult(False, None, last_error, True)

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                return ApiResult(
                    False,
                    None,
                    f"Request failed ({response.status_code}): {detail}",
                    False,
                )

            return ApiResult(True, response.json(), "ok", False)
        except requests.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            time.sleep(0.6 * attempt)

    return ApiResult(False, None, f"Environment unavailable — retrying ({last_error})", True)


def make_action(route_to: str, classification: str, priority: str) -> dict[str, Any]:
    return {
        "route_to": route_to,
        "classification": classification,
        "priority": priority,
        "escalate": route_to == "escalate",
        "flag_injection": False,
        "extracted_fields": {},
        "reply_draft": None,
    }


def rule_agent_decide(obs: dict[str, Any]) -> tuple[str, str]:
    """Exact rule policy requested for the Watch AI tab."""
    flags = obs.get("flags", {})
    hist = obs.get("delegation_history", {})
    budget = obs.get("budget_remaining", 1.0)
    sla = obs.get("sla_remaining", 1.0)
    email = obs.get("current_email", {})

    team_a_degraded = flags.get("any_team_degraded", False)
    escalate_uses = hist.get("escalate_uses", 0)

    if sla < 0.2 and budget > 0.4 and escalate_uses < 2:
        return "escalate", "SLA critical — escalating immediately"

    if budget < 0.15:
        return "handle_self", "Budget critical — handling internally"

    if team_a_degraded:
        return "delegate_thorough", "Team A degraded — routing to thorough team"

    subject = email.get("subject", "").lower()
    if any(word in subject for word in ["inquiry", "question", "update", "info"]):
        return "wait", "Low urgency — waiting for possible auto-resolve"

    if "legal" in subject or "compliance" in subject:
        return "delegate_thorough", "Legal content — thorough review needed"

    return "delegate_fast", "Standard routing — fast team assigned"


def infer_badge(email: dict[str, Any]) -> str:
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
    checks = [
        ("BILLING", ["invoice", "payment", "billing", "refund", "charge"]),
        ("LEGAL", ["legal", "contract", "liability", "court", "notice"]),
        ("TECHNICAL", ["bug", "error", "outage", "integration", "api"]),
        ("HR", ["leave", "policy", "hiring", "benefits", "manager"]),
        ("COMPLIANCE", ["audit", "compliance", "regulation", "gdpr", "soc2"]),
    ]
    for label, words in checks:
        if any(word in text for word in words):
            return label
    return "GENERAL"


def meter_color(value: float) -> str:
    if value > 0.5:
        return "#22c55e"
    if value >= 0.3:
        return "#fbbf24"
    return "#ef4444"


def meter_html(label: str, value: float, warning: str | None = None) -> str:
    pct = max(0.0, min(1.0, value))
    color = meter_color(pct)
    warn = f'<span class="warn">{warning}</span>' if warning else ""
    return f"""
    <div class="meter-wrap">
      <div class="meter-head"><span>{label}</span><span>{pct * 100:.0f}% {warn}</span></div>
      <div class="meter-track"><div class="meter-fill" style="width:{pct * 100:.1f}%; background:{color};"></div></div>
    </div>
    """


def history_circles(values: list[bool]) -> str:
    circles = []
    for result in values[-3:]:
        color = "#22c55e" if result else "#ef4444"
        circles.append(f'<span class="dot" style="background:{color};"></span>')
    while len(circles) < 3:
        circles.insert(0, '<span class="dot empty"></span>')
    return "".join(circles)


def parse_recent_history(hist_value: Any) -> list[bool]:
    if isinstance(hist_value, list):
        parsed = []
        for item in hist_value:
            if isinstance(item, bool):
                parsed.append(item)
            elif isinstance(item, (int, float)):
                parsed.append(item > 0)
            elif isinstance(item, str):
                parsed.append(item.lower() in {"true", "success", "ok", "1"})
        return parsed
    return []


def base_state() -> dict[str, Any]:
    return {
        "session_id": None,
        "task_label": "Easy",
        "task_id": "task_easy",
        "seed": 42,
        "obs": {},
        "logs": [],
        "status": "Idle",
        "total_reward": 0.0,
        "last_reward": 0.0,
        "last_reasoning": "Awaiting episode start.",
        "last_action": "N/A",
        "done": False,
        "retry_pending": False,
        "retry_action": None,
        "events": {"drift": 0, "cascade": 0, "auto_resolve": 0, "sla_breach": 0},
        "drift_banner_until": -1,
        "end_reason": "",
    }


def normalize_obs(raw_obs: dict[str, Any]) -> dict[str, Any]:
    obs = dict(raw_obs or {})
    obs.setdefault("current_email", {})
    obs.setdefault("flags", {})
    obs.setdefault("delegation_history", {})
    obs.setdefault("budget_remaining", 1.0)
    obs.setdefault("sla_remaining", 1.0)
    obs.setdefault("step_index", obs.get("step", 0))
    obs.setdefault("max_steps", 20)
    return obs


def summarize_end_reason(state: dict[str, Any], obs: dict[str, Any]) -> str:
    budget = obs.get("budget_remaining", 1.0)
    sla = obs.get("sla_remaining", 1.0)
    if budget <= 0.01:
        return "Agent ran out of budget"
    if sla <= 0.01:
        return "SLA breached"
    if state["total_reward"] >= 0:
        return "Agent succeeded"
    return "Episode completed with negative return"


def extract_notes(obs: dict[str, Any], info: dict[str, Any]) -> str:
    notes = []
    step_index = int(obs.get("step_index", 0))
    if step_index in {7, 14}:
        notes.append("DRIFT EVENT")
    flags = obs.get("flags", {})
    if flags.get("sla_at_risk"):
        notes.append("SLA WARNING")
    if obs.get("budget_remaining", 1.0) < 0.3:
        notes.append("BUDGET WARNING")

    info_blob = json.dumps(info or {}).lower()
    if "cascade" in info_blob:
        notes.append("CASCADE")
    if "auto" in info_blob and "resolve" in info_blob:
        notes.append("AUTO-RESOLVED")
    return " | ".join(notes)


def run_untrained_baseline(task_id: str, seed: int) -> float | None:
    reset_res = api_call("POST", "/reset", {"task_id": task_id, "seed": seed})
    if not reset_res.ok or not reset_res.data:
        return None
    sid = reset_res.data["session_id"]
    obs = normalize_obs(reset_res.data.get("observation", {}))
    total = 0.0
    done = False
    guard = 0
    while not done and guard < 40:
        action = make_action("delegate_fast", "general", "medium")
        step_res = api_call("POST", "/step", {"session_id": sid, "action": action})
        if not step_res.ok or not step_res.data:
            return None
        total += float(step_res.data.get("reward", 0.0))
        done = bool(step_res.data.get("done", False))
        obs = normalize_obs(step_res.data.get("observation", obs))
        guard += 1
    return total


def render(state: dict[str, Any], tab_mode: str) -> list[Any]:
    obs = normalize_obs(state.get("obs", {}))
    email = obs.get("current_email", {})
    hist = obs.get("delegation_history", {})
    step_idx = int(obs.get("step_index", 0))
    max_steps = int(obs.get("max_steps", 20))

    header = f"Step {step_idx} of {max_steps}"
    status = state.get("status", "Idle")
    if state.get("retry_pending"):
        status = "Failed (retry available)"

    email_subject = email.get("subject", "No email loaded")
    email_body = (email.get("body") or "Start a new episode to load email context.")[:300]
    sender = email.get("sender", "N/A")
    badge = infer_badge(email)
    badge_html = f'<span class="badge">{badge}</span>'

    budget = float(obs.get("budget_remaining", 1.0))
    sla = float(obs.get("sla_remaining", 1.0))
    degraded = bool(obs.get("flags", {}).get("any_team_degraded", False))
    team_a_meter = meter_html("Team A Reliability", 0.25 if degraded else 0.85, "DEGRADED" if degraded else None)
    team_b_meter = meter_html("Team B Reliability", 0.45 if degraded else 0.88, "DEGRADED" if degraded else None)
    budget_meter = meter_html("Budget Remaining", budget)
    sla_meter = meter_html("SLA Remaining", sla)
    env_meters = f"{budget_meter}{sla_meter}{team_a_meter}{team_b_meter}"

    team_a_recent = history_circles(parse_recent_history(hist.get("team_A_recent", [])))
    team_b_recent = history_circles(parse_recent_history(hist.get("team_B_recent", [])))
    team_a_uses = hist.get("team_A_uses", 0)
    team_b_uses = hist.get("team_B_uses", 0)
    esc_uses = int(hist.get("escalate_uses", 0))
    esc_warn = '<span class="warn">Warning: high escalation usage</span>' if esc_uses > 2 else ""
    history_html = f"""
    <div class="hist-row"><span>Team A recent:</span><span>{team_a_recent}</span><span>Uses: {team_a_uses}</span></div>
    <div class="hist-row"><span>Team B recent:</span><span>{team_b_recent}</span><span>Uses: {team_b_uses}</span></div>
    <div class="hist-row"><span>Escalations:</span><span>{esc_uses}</span><span>{esc_warn}</span></div>
    """

    events_banner = ""
    if step_idx in {7, 14}:
        state["drift_banner_until"] = step_idx + 1
    if step_idx <= state.get("drift_banner_until", -1) and step_idx > 0:
        events_banner += (
            '<div class="banner drift">DRIFT EVENT — Team reliability has shifted. '
            "Agent must adapt strategy.</div>"
        )
    if state.get("logs"):
        last_notes = state["logs"][-1][-1]
        if "CASCADE" in last_notes:
            events_banner += '<div class="banner cascade">CASCADE EVENT — prior errors are compounding.</div>'
        if "AUTO-RESOLVED" in last_notes:
            events_banner += '<div class="banner auto">AUTO-RESOLVE — waiting action resolved this thread.</div>'

    reward = float(state.get("last_reward", 0.0))
    reward_color = "#22c55e" if reward >= 0 else "#ef4444"
    decision_html = (
        f"<div><b>Action:</b> {state.get('last_action')}</div>"
        f"<div><b>Reasoning:</b> {state.get('last_reasoning')}</div>"
        f'<div><b>Step reward:</b> <span style="color:{reward_color};">{reward:+.3f}</span></div>'
        f"<div><b>Total reward:</b> {state.get('total_reward', 0.0):+.3f}</div>"
    )

    logs = state.get("logs", [])
    summary_html = "<i>Episode in progress...</i>"
    if state.get("done"):
        baseline = run_untrained_baseline(state.get("task_id", "task_easy"), int(state.get("seed", 42)))
        baseline_str = "N/A (baseline unavailable)" if baseline is None else f"{state['total_reward']:+.3f} vs {baseline:+.3f}"
        ev = state.get("events", {})
        verdict = state.get("end_reason") or "Episode completed"
        summary_html = (
            f"<div><b>Final total reward:</b> {state['total_reward']:+.3f}</div>"
            f"<div><b>Your score vs untrained agent:</b> {baseline_str}</div>"
            f"<div><b>Key events:</b> drift={ev.get('drift',0)}, cascade={ev.get('cascade',0)}, "
            f"auto-resolve={ev.get('auto_resolve',0)}, SLA breaches={ev.get('sla_breach',0)}</div>"
            f"<div><b>Verdict:</b> {verdict}</div>"
        )

    return [
        header,
        status,
        f"**{email_subject}**",
        email_body,
        sender,
        badge_html,
        env_meters,
        history_html,
        events_banner,
        decision_html if tab_mode == "watch" else gr.update(),
        logs,
        summary_html,
        gr.update(interactive=state.get("retry_pending", False)),
        gr.update(interactive=not state.get("done", False) and bool(state.get("session_id"))),
    ]


def start_episode(task_label: str, seed: int, tab_state: dict[str, Any]) -> tuple[dict[str, Any], *tuple[Any, ...]]:
    state = base_state()
    state["task_label"] = task_label
    state["task_id"] = TASK_ID_MAP[task_label]
    state["seed"] = int(seed)
    state["status"] = "Running"

    reset_payload = {"task_id": state["task_id"], "seed": int(seed)}
    result = api_call("POST", "/reset", reset_payload)
    if not result.ok or not result.data:
        state["status"] = "Failed"
        state["retry_pending"] = bool(result.retryable)
        state["last_reasoning"] = result.message
        state["done"] = True
        render_values = render(state, "watch")
        return (state, *render_values)

    state["session_id"] = result.data["session_id"]
    state["obs"] = normalize_obs(result.data.get("observation", {}))
    state["done"] = False
    state["last_action"] = "Episode started"
    state["last_reasoning"] = "Initial observation loaded from environment."
    render_values = render(state, "watch")
    return (state, *render_values)


def apply_step(state: dict[str, Any], action: dict[str, Any], reasoning: str, tab_mode: str) -> tuple[dict[str, Any], *tuple[Any, ...]]:
    if state.get("done") or not state.get("session_id"):
        state["last_reasoning"] = "Start an episode before stepping."
        values = render(state, tab_mode)
        return (state, *values)

    payload = {"session_id": state["session_id"], "action": action}
    step_res = api_call("POST", "/step", payload)
    if not step_res.ok or not step_res.data:
        state["retry_pending"] = True
        state["retry_action"] = action
        state["status"] = "Failed"
        state["last_reasoning"] = f"{reasoning} | {step_res.message}"
        values = render(state, tab_mode)
        return (state, *values)

    state["retry_pending"] = False
    state["retry_action"] = None
    state["status"] = "Running"
    state["last_action"] = action["route_to"]
    state["last_reasoning"] = reasoning
    state["last_reward"] = float(step_res.data.get("reward", 0.0))
    state["total_reward"] += state["last_reward"]
    obs = normalize_obs(step_res.data.get("observation", {}))
    state["obs"] = obs
    info = step_res.data.get("info", {})
    notes = extract_notes(obs, info)
    if "DRIFT EVENT" in notes:
        state["events"]["drift"] += 1
    if "CASCADE" in notes:
        state["events"]["cascade"] += 1
    if "AUTO-RESOLVED" in notes:
        state["events"]["auto_resolve"] += 1
    if obs.get("sla_remaining", 1.0) <= 0.01:
        state["events"]["sla_breach"] += 1

    email_subject = obs.get("current_email", {}).get("subject", "N/A")
    state["logs"].append(
        [
            int(obs.get("step_index", 0)),
            email_subject[:65],
            action["route_to"],
            round(state["last_reward"], 3),
            round(float(obs.get("budget_remaining", 0.0)), 3),
            round(float(obs.get("sla_remaining", 0.0)), 3),
            notes,
        ]
    )

    state["done"] = bool(step_res.data.get("done", False))
    if state["done"]:
        state["status"] = "Complete" if state["total_reward"] >= 0 else "Failed"
        state["end_reason"] = summarize_end_reason(state, obs)

    values = render(state, tab_mode)
    return (state, *values)


def next_ai_step(state: dict[str, Any]) -> tuple[dict[str, Any], *tuple[Any, ...]]:
    obs = normalize_obs(state.get("obs", {}))
    route_to, reason = rule_agent_decide(obs)
    action = make_action(route_to=route_to, classification=infer_badge(obs.get("current_email", {})).lower(), priority="medium")
    return apply_step(state, action, reason, "watch")


def retry_last(state: dict[str, Any], tab_mode: str) -> tuple[dict[str, Any], *tuple[Any, ...]]:
    retry_action = state.get("retry_action")
    if not retry_action:
        state["last_reasoning"] = "No retryable step action available."
        values = render(state, tab_mode)
        return (state, *values)
    return apply_step(state, retry_action, "Retrying previous action.", tab_mode)


def human_step(state: dict[str, Any], action_label: str, classification: str, priority: str) -> tuple[dict[str, Any], *tuple[Any, ...]]:
    route_to = ACTION_OPTIONS[action_label]
    reasoning = f"Human selected {action_label} with {classification}/{priority}."
    action = make_action(route_to=route_to, classification=classification, priority=priority)
    return apply_step(state, action, reasoning, "play")


CUSTOM_CSS = """
.app {max-width: 1200px; margin: 0 auto;}
.panel {border: 1px solid #2a2a2a; border-radius: 10px; padding: 12px; background: #13151a;}
.badge {display:inline-block;padding:4px 10px;border-radius:999px;background:#1e293b;color:#93c5fd;font-weight:600;}
.meter-wrap {margin-bottom:10px;}
.meter-head {display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;color:#d1d5db;}
.meter-track {height:10px;background:#1f2937;border-radius:999px;overflow:hidden;}
.meter-fill {height:100%;transition:width 0.4s ease-in-out;}
.warn {color:#f59e0b;font-weight:700;margin-left:8px;}
.dot {width:12px;height:12px;display:inline-block;border-radius:999px;margin:0 3px;}
.dot.empty {background:#4b5563;}
.hist-row {display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:8px;}
.banner {padding:10px;border-radius:8px;margin-bottom:8px;font-weight:700;}
.drift {background:#7c2d12;color:#fdba74;}
.cascade {background:#7f1d1d;color:#fecaca;}
.auto {background:#1e3a8a;color:#bfdbfe;}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=CUSTOM_CSS,
    title="OpenInbox Interactive RL Demo",
) as demo:
    gr.Markdown(
        """
        # OpenInbox: AI Ops Manager Simulator
        Interactive RL demo where actions trade off accuracy, budget, and SLA under hidden team reliability drift.
        """,
        elem_classes=["app"],
    )

    with gr.Tabs():
        with gr.Tab("Watch AI Agent"):
            watch_state = gr.State(base_state())
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 1 — Episode Header")
                with gr.Row():
                    watch_task = gr.Dropdown(choices=list(TASK_ID_MAP.keys()), value="Easy", label="Task difficulty")
                    watch_seed = gr.Number(value=42, precision=0, label="Seed")
                    watch_start = gr.Button("Start New Episode", variant="primary")
                    watch_next = gr.Button("Next Step", variant="secondary")
                with gr.Row():
                    watch_step_counter = gr.Textbox(label="Current step", value="Step 0 of 0", interactive=False)
                    watch_status = gr.Textbox(label="Episode status", value="Idle", interactive=False)

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 2 — Current Email Panel")
                watch_subject = gr.Markdown("**No email loaded**")
                watch_body = gr.Textbox(label="Email body (max 300 chars)", lines=6, max_lines=8, interactive=False)
                watch_sender = gr.Textbox(label="Sender", interactive=False)
                watch_badge = gr.HTML("<span class='badge'>GENERAL</span>")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 3 — Environment State Panel")
                watch_meters = gr.HTML("")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 4 — Delegation History Panel")
                watch_history = gr.HTML("")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 5 — Agent Decision Panel")
                watch_event_banner = gr.HTML("")
                watch_decision = gr.HTML("")
                watch_retry = gr.Button("Retry Last Step", interactive=False)

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 6 — Episode Log")
                watch_log = gr.Dataframe(
                    headers=["Step", "Email Subject", "Action", "Reward", "Budget", "SLA", "Notes"],
                    datatype=["number", "str", "str", "number", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 7 — Episode Summary")
                watch_summary = gr.HTML("<i>Episode in progress...</i>")

        with gr.Tab("Play Yourself"):
            play_state = gr.State(base_state())
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 1 — Episode Header")
                with gr.Row():
                    play_task = gr.Dropdown(choices=list(TASK_ID_MAP.keys()), value="Easy", label="Task difficulty")
                    play_seed = gr.Number(value=42, precision=0, label="Seed")
                    play_start = gr.Button("Start New Episode", variant="primary")
                with gr.Row():
                    play_step_counter = gr.Textbox(label="Current step", value="Step 0 of 0", interactive=False)
                    play_status = gr.Textbox(label="Episode status", value="Idle", interactive=False)

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 2 — Current Email Panel")
                play_subject = gr.Markdown("**No email loaded**")
                play_body = gr.Textbox(label="Email body (max 300 chars)", lines=6, max_lines=8, interactive=False)
                play_sender = gr.Textbox(label="Sender", interactive=False)
                play_badge = gr.HTML("<span class='badge'>GENERAL</span>")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 3 — Environment State Panel")
                play_meters = gr.HTML("")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 4 — Delegation History Panel")
                play_history = gr.HTML("")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 5 — Your Decision Panel")
                play_event_banner = gr.HTML("")
                play_action = gr.Radio(
                    choices=list(ACTION_OPTIONS.keys()),
                    value="Delegate Fast",
                    label="Action",
                )
                play_classification = gr.Dropdown(choices=CLASSIFICATION_OPTIONS, value="general", label="Classification")
                play_priority = gr.Dropdown(choices=PRIORITY_OPTIONS, value="medium", label="Priority")
                play_take_action = gr.Button("Take Action", variant="primary")
                play_reward = gr.HTML("")
                play_retry = gr.Button("Retry Last Step", interactive=False)

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 6 — Episode Log")
                play_log = gr.Dataframe(
                    headers=["Step", "Email Subject", "Action", "Reward", "Budget", "SLA", "Notes"],
                    datatype=["number", "str", "str", "number", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Section 7 — Episode Summary")
                play_summary = gr.HTML("<i>Episode in progress...</i>")

    watch_outputs = [
        watch_step_counter,
        watch_status,
        watch_subject,
        watch_body,
        watch_sender,
        watch_badge,
        watch_meters,
        watch_history,
        watch_event_banner,
        watch_decision,
        watch_log,
        watch_summary,
        watch_retry,
        watch_next,
    ]

    play_outputs = [
        play_step_counter,
        play_status,
        play_subject,
        play_body,
        play_sender,
        play_badge,
        play_meters,
        play_history,
        play_event_banner,
        play_reward,
        play_log,
        play_summary,
        play_retry,
        play_take_action,
    ]

    watch_start.click(
        fn=lambda task, seed, state: start_episode(task, int(seed), state),
        inputs=[watch_task, watch_seed, watch_state],
        outputs=[watch_state, *watch_outputs],
    )
    watch_next.click(
        fn=next_ai_step,
        inputs=[watch_state],
        outputs=[watch_state, *watch_outputs],
    )
    watch_retry.click(
        fn=lambda state: retry_last(state, "watch"),
        inputs=[watch_state],
        outputs=[watch_state, *watch_outputs],
    )

    play_start.click(
        fn=lambda task, seed, state: start_episode(task, int(seed), state),
        inputs=[play_task, play_seed, play_state],
        outputs=[play_state, *play_outputs],
    )
    play_take_action.click(
        fn=human_step,
        inputs=[play_state, play_action, play_classification, play_priority],
        outputs=[play_state, *play_outputs],
    )
    play_retry.click(
        fn=lambda state: retry_last(state, "play"),
        inputs=[play_state],
        outputs=[play_state, *play_outputs],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
