import pytest
from fastapi.testclient import TestClient

from app.extractor import Extractor
from app.main import app, get_extractor
from tests.fakes import SAMPLE_REPLIES, FakeLLM


@pytest.fixture
def client_with():
    """Build a TestClient whose /extract uses a FakeLLM with the given replies."""

    def _make(*replies):
        app.dependency_overrides[get_extractor] = lambda: Extractor(FakeLLM(*replies))
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


EXPECTED = {
    # message: (set of (sign_id, value), escalate)
    "My cow has not been eating since yesterday and she is breathing very fast.": (
        {("reduced_appetite", "yes"), ("rapid_breathing", "yes")}, False),
    "There's foam coming from her nose and she can't get up.": (
        {("froth_nose_mouth", "yes"), ("unable_to_stand", "yes")}, True),
    "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear.": (
        {("cloudy_eye", "not_sure"), ("swelling", "yes")}, True),
    "The calf is fine, just lots of ticks on the ears.": (
        {("ticks", "yes")}, False),
    "Mombe yangu haisi kudya, and her urine is dark like tea.": (
        {("reduced_appetite", "yes"), ("red_urine", "yes")}, False),
}


@pytest.mark.parametrize("message", EXPECTED)
def test_sample_messages(client_with, message):
    client = client_with(SAMPLE_REPLIES[message])
    response = client.post("/extract", json={"message": message})

    assert response.status_code == 200
    body = response.json()
    expected_findings, expected_escalate = EXPECTED[message]
    assert {(f["sign_id"], f["value"]) for f in body["findings"]} == expected_findings
    assert body["escalate"] is expected_escalate
    assert body["reason"]
    for f in body["findings"]:
        assert f["quote"].lower() in message.lower()


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   "}, {"msg": "hi"}])
def test_bad_request_is_rejected(client_with, payload):
    client = client_with()
    assert client.post("/extract", json=payload).status_code == 422


def test_llm_failure_returns_502(client_with):
    client = client_with("not json", "still not json")
    response = client.post("/extract", json={"message": "My cow is coughing"})
    assert response.status_code == 502


def test_bad_request_is_rejected_even_without_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_extractor.cache_clear()
    response = TestClient(app).post("/extract", json={"message": ""})
    assert response.status_code == 422
    get_extractor.cache_clear()


def test_missing_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_extractor.cache_clear()
    client = TestClient(app)
    response = client.post("/extract", json={"message": "My cow is coughing"})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
    get_extractor.cache_clear()
