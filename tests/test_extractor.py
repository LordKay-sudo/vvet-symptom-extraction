import pytest

from app.extractor import ExtractionError, Extractor, quote_in_message
from app.signs import SignId
from tests.fakes import FakeLLM

MESSAGE = "There's foam coming from her nose and she can't get up."


def test_valid_output_is_returned():
    llm = FakeLLM({"findings": [{"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"}]})
    findings = Extractor(llm).extract(MESSAGE)
    assert [(f.sign_id, f.value) for f in findings] == [(SignId.FROTH_NOSE_MOUTH, "yes")]


def test_message_is_sent_to_llm():
    llm = FakeLLM({"findings": []})
    Extractor(llm).extract(MESSAGE)
    assert MESSAGE in llm.calls[0][1]


def test_unknown_sign_id_is_dropped():
    llm = FakeLLM({"findings": [
        {"sign_id": "fever", "value": "yes", "quote": "foam"},
        {"sign_id": "unable_to_stand", "value": "yes", "quote": "can't get up"},
    ]})
    findings = Extractor(llm).extract(MESSAGE)
    assert [f.sign_id for f in findings] == [SignId.UNABLE_TO_STAND]


def test_invalid_value_and_missing_fields_are_dropped():
    llm = FakeLLM({"findings": [
        {"sign_id": "froth_nose_mouth", "value": "maybe", "quote": "foam"},
        {"sign_id": "unable_to_stand", "value": "yes"},
        "not even an object",
    ]})
    assert Extractor(llm).extract(MESSAGE) == []


def test_hallucinated_quote_is_dropped():
    llm = FakeLLM({"findings": [{"sign_id": "red_urine", "value": "yes", "quote": "urine is red"}]})
    assert Extractor(llm).extract(MESSAGE) == []


def test_duplicate_sign_keeps_most_cautious_value():
    llm = FakeLLM({"findings": [
        {"sign_id": "unable_to_stand", "value": "no", "quote": "she can't get up"},
        {"sign_id": "unable_to_stand", "value": "not_sure", "quote": "can't get up"},
    ]})
    findings = Extractor(llm).extract(MESSAGE)
    assert len(findings) == 1
    assert findings[0].value == "not_sure"


def test_malformed_json_is_retried_once():
    llm = FakeLLM("Sure! Here are the findings:", {"findings": []})
    assert Extractor(llm).extract(MESSAGE) == []
    assert len(llm.calls) == 2


def test_wrong_shape_counts_as_malformed():
    llm = FakeLLM('{"signs": []}', '["findings"]')
    with pytest.raises(ExtractionError):
        Extractor(llm).extract(MESSAGE)
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "quote,expected",
    [
        ("foam coming from her nose", True),
        ("FOAM   coming from her nose", True),  # case and spacing
        ("she can\u2019t get up", True),  # curly apostrophe
        ('"she can\'t get up."', True),  # wrapping quotes and punctuation
        ("froth from the mouth", False),
        ("...", False),
    ],
)
def test_quote_in_message(quote, expected):
    assert quote_in_message(quote, MESSAGE) is expected


def test_one_character_quote_slip_is_kept_as_farmers_words():
    message = "Mombe yangu haisi kudya, and her urine is dark like tea."
    llm = FakeLLM({"findings": [
        {"sign_id": "reduced_appetite", "value": "yes", "quote": "haisii kudya"},
    ]})
    findings = Extractor(llm).extract(message)
    assert [(f.sign_id, f.value, f.quote) for f in findings] == [
        (SignId.REDUCED_APPETITE, "yes", "haisi kudya"),
    ]


def test_changed_letter_does_not_ground():
    message = "The cow is not eating well today."
    assert quote_in_message("now eating well today", message) is False
