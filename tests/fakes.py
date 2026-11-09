"""A fake LLM so tests never touch the network or need an API key."""

import json


class FakeLLM:
    """Returns pre-set replies in order and records every call."""

    def __init__(self, *replies: str | dict):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)


# What a well-behaved model should return for the five sample messages in the brief.
SAMPLE_REPLIES: dict[str, dict] = {
    "My cow has not been eating since yesterday and she is breathing very fast.": {
        "findings": [
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "has not been eating"},
            {"sign_id": "rapid_breathing", "value": "yes", "quote": "breathing very fast"},
        ]
    },
    "There's foam coming from her nose and she can't get up.": {
        "findings": [
            {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
            {"sign_id": "unable_to_stand", "value": "yes", "quote": "she can't get up"},
        ]
    },
    "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear.": {
        "findings": [
            {"sign_id": "cloudy_eye", "value": "not_sure", "quote": "I think the eyes look a bit cloudy but I'm not sure"},
            {"sign_id": "swelling", "value": "yes", "quote": "mbandabvu under the ear"},
        ]
    },
    "The calf is fine, just lots of ticks on the ears.": {
        "findings": [
            {"sign_id": "ticks", "value": "yes", "quote": "lots of ticks on the ears"},
        ]
    },
    "Mombe yangu haisi kudya, and her urine is dark like tea.": {
        "findings": [
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "haisi kudya"},
            {"sign_id": "red_urine", "value": "yes", "quote": "urine is dark like tea"},
        ]
    },
}
