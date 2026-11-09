"""Escalation rule. Deliberately plain code, never the LLM."""

from app.models import Finding
from app.signs import DANGER_SIGNS

ESCALATING_VALUES = {"yes", "not_sure"}


def decide_escalation(findings: list[Finding]) -> tuple[bool, str]:
    """Escalate when any danger sign is reported as yes or not_sure.

    not_sure escalates on purpose: for these signs a false alarm is much
    cheaper than a missed case.
    """
    triggered = [
        f for f in findings if f.sign_id in DANGER_SIGNS and f.value in ESCALATING_VALUES
    ]
    if not triggered:
        names = ", ".join(s.value for s in DANGER_SIGNS)
        return False, f"No danger signs ({names}) reported."

    # Report in the fixed DANGER_SIGNS order so the reason is stable.
    triggered.sort(key=lambda f: DANGER_SIGNS.index(f.sign_id))
    parts = ", ".join(f"{f.sign_id.value} ({f.value})" for f in triggered)
    return True, f"Danger sign(s) reported: {parts}."
