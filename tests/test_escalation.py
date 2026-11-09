import itertools

import pytest

from app.escalation import decide_escalation
from app.models import Finding
from app.signs import DANGER_SIGNS, SignId


def finding(sign: SignId, value: str) -> Finding:
    return Finding(sign_id=sign, value=value, quote="x")


def test_no_findings_does_not_escalate():
    escalate, reason = decide_escalation([])
    assert escalate is False
    assert "No danger signs" in reason


@pytest.mark.parametrize(
    "sign,value",
    list(itertools.product(DANGER_SIGNS, ["yes", "not_sure"])),
)
def test_danger_sign_yes_or_not_sure_escalates(sign, value):
    escalate, reason = decide_escalation([finding(sign, value)])
    assert escalate is True
    assert f"{sign.value} ({value})" in reason


@pytest.mark.parametrize("sign", DANGER_SIGNS)
def test_danger_sign_no_does_not_escalate(sign):
    escalate, _ = decide_escalation([finding(sign, "no")])
    assert escalate is False


@pytest.mark.parametrize(
    "sign,value",
    list(itertools.product(
        [s for s in SignId if s not in DANGER_SIGNS], ["yes", "no", "not_sure"]
    )),
)
def test_other_signs_never_escalate(sign, value):
    escalate, _ = decide_escalation([finding(sign, value)])
    assert escalate is False


def test_reason_lists_all_triggering_signs_in_fixed_order():
    findings = [
        finding(SignId.CLOUDY_EYE, "not_sure"),
        finding(SignId.TICKS, "yes"),
        finding(SignId.FROTH_NOSE_MOUTH, "yes"),
    ]
    escalate, reason = decide_escalation(findings)
    assert escalate is True
    assert reason == "Danger sign(s) reported: froth_nose_mouth (yes), cloudy_eye (not_sure)."
