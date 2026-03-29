"""
Rule-based fallback agent for OpenInbox.

This is NOT the main baseline. It exists so you can test the environment
without an OpenAI API key. Scoring will be lower than the OpenAI baseline.

Activate with:  python baseline/run_baseline.py --task all --fallback
"""

import re

from environment.injection import detect as detect_injection
from environment.models import Action, Observation


_CLASSIFICATION_KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "invoice", "payment", "charge", "billing", "refund",
        "outstanding", "overdue", "amount", "account",
    ],
    "technical": [
        "error", "outage", "database", "server", "system",
        "api", "production", "incident", "unreachable", "severity", "p1",
    ],
    "hr": [
        "leave", "vacation", "pto", "annual leave",
        "holiday", "sick", "absence",
    ],
    "legal": [
        "legal", "law", "contract", "agreement", "notice",
        "counsel", "dispute", "arbitration", "clause", "formal",
    ],
    "spam": ["lottery", "winner", "prize", "click here", "congratulations"],
}

_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    "critical": [
        "critical", "p1", "immediately", "emergency",
        "production down", "unreachable", "sla breach",
    ],
    "high": [
        "urgent", "asap", "expires", "expiry", "deadline",
        "tomorrow", "formal notice", "required by",
    ],
    "low": ["when possible", "no rush", "fyi", "at your convenience"],
}

_ROUTE_MAP: dict[str, str] = {
    "billing":   "billing_team",
    "technical": "tech_team",
    "hr":        "hr_team",
    "legal":     "legal_team",
    "spam":      "spam_filter",
    "unknown":   "billing_team",   # safe fallback
}


class RuleAgent:
    """
    Keyword-based agent. No API calls, no state kept between steps.
    Each act() call is independent and deterministic.
    """

    def act(self, obs: Observation) -> Action:
        email = obs.current_email
        text = f"{email.subject} {email.body}".lower()

        classification = _classify(text)
        priority = _prioritize(text, obs.flags.get("sla_at_risk", False))
        route_to = _ROUTE_MAP.get(classification, "billing_team")
        extracted = _extract_fields(email.body)
        injection = detect_injection(email.body)

        # Escalate only for legal emails that explicitly mention escalation
        escalate = classification == "legal" and "escalat" in text

        return Action(
            classification=classification,
            priority=priority,
            route_to=route_to,
            extracted_fields=extracted,
            escalate=escalate,
            flag_injection=injection["detected"],
            reply_draft=None,
        )


def _classify(text: str) -> str:
    scores = {cls: 0 for cls in _CLASSIFICATION_KEYWORDS}
    for cls, keywords in _CLASSIFICATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[cls] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def _prioritize(text: str, sla_at_risk: bool) -> str:
    if sla_at_risk:
        return "high"
    for priority, keywords in _PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return priority
    return "medium"


def _extract_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}

    m = re.search(r"invoice\s*#?\s*([A-Z0-9-]+)", body, re.IGNORECASE)
    if m:
        fields["invoice_number"] = m.group(1)

    m = re.search(r"[\$€£]\s*([\d,]+(?:\.\d{1,2})?)", body)
    if m:
        fields["amount"] = m.group(1).replace(",", "")

    m = re.search(r"([A-Z]{2,}-\d{4}-\d{3,})", body)
    if m:
        fields["contract_id"] = m.group(1)

    m = re.search(r"(INC-[\d-]+)", body)
    if m:
        fields["incident_id"] = m.group(1)

    m = re.search(r"severity[:\s]+(\w+)", body, re.IGNORECASE)
    if m:
        fields["severity"] = m.group(1)

    m = re.search(r"(\d+)\s+(?:working |business )?days?", body, re.IGNORECASE)
    if m:
        fields["days_requested"] = m.group(1)

    return fields
