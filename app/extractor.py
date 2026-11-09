"""Turn a farmer's free-text message into validated findings using an LLM."""

import json
import logging

from pydantic import ValidationError

from app.llm import LLMClient
from app.models import Finding
from app.signs import SIGN_DESCRIPTIONS

logger = logging.getLogger(__name__)

# When the model reports the same sign twice, keep the most cautious value.
VALUE_PRIORITY = {"yes": 2, "not_sure": 1, "no": 0}

_SIGN_LINES = "\n".join(f"- {sign.value}: {desc}" for sign, desc in SIGN_DESCRIPTIONS.items())

SYSTEM_PROMPT = f"""You extract livestock health signs from messages that farmers in Zimbabwe send on WhatsApp.
Messages may be in English, Shona, Ndebele, or a mix of them.

Only use these signs (sign_id: meaning):
{_SIGN_LINES}

Shona hints: "kudya" means eating, so "haisi kudya" / "hairi kudya" means not eating;
"mbandabvu" is the swelling sign; "zvikwekwe" means ticks.

Rules:
1. Only report signs the message actually mentions. Do not guess signs that are not described.
2. value is "yes" if the farmer says the sign is present, "no" only if the farmer explicitly
   says that sign is absent (e.g. "she is still eating"), and "not_sure" if the farmer sounds
   unsure (e.g. "I think", "maybe", "not sure", "it looks like").
   A general statement like "the calf is fine" is not a "no" for every sign; report nothing for it.
3. quote must be copied exactly, character for character, from the message: the shortest part
   that shows the sign. Do not translate, rephrase, or correct the farmer's spelling.
4. Report each sign_id at most once.

Reply with only a JSON object in this shape:
{{"findings": [{{"sign_id": "<one of the sign_ids above>", "value": "yes|no|not_sure", "quote": "<exact text>"}}]}}
If no signs are mentioned, reply {{"findings": []}}.

Example
Message: "She is still eating but she has been lying down all day and I saw some red pee"
Reply: {{"findings": [
  {{"sign_id": "reduced_appetite", "value": "no", "quote": "She is still eating"}},
  {{"sign_id": "weakness", "value": "yes", "quote": "lying down all day"}},
  {{"sign_id": "red_urine", "value": "yes", "quote": "red pee"}}
]}}"""


class ExtractionError(RuntimeError):
    """The LLM did not return usable output, even after a retry."""


class Extractor:
    def __init__(self, llm: LLMClient, max_attempts: int = 2):
        self._llm = llm
        self._max_attempts = max_attempts

    def extract(self, message: str) -> list[Finding]:
        user_prompt = f'Farmer\'s message:\n"""{message}"""'
        for attempt in range(1, self._max_attempts + 1):
            raw = self._llm.complete(SYSTEM_PROMPT, user_prompt)
            items = parse_findings_json(raw)
            if items is not None:
                return clean_findings(items, message)
            logger.warning("Malformed LLM output (attempt %d): %r", attempt, raw[:200])
        raise ExtractionError("The LLM did not return valid JSON.")


def parse_findings_json(raw: str) -> list | None:
    """Return the raw "findings" list, or None if the reply is not the expected shape."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        return None
    return data["findings"]


def clean_findings(items: list, message: str) -> list[Finding]:
    """Validate each item, drop ungrounded quotes, and keep one finding per sign."""
    best: dict[str, Finding] = {}
    for item in items:
        try:
            finding = Finding.model_validate(item)
        except ValidationError:
            # Unknown sign_id, bad value, missing quote, ... Drop just this item.
            logger.warning("Dropping invalid finding from LLM: %r", item)
            continue

        grounded = grounded_quote(finding.quote, message)
        if grounded is None:
            logger.warning("Dropping finding with quote not in message: %r", item)
            continue
        if grounded != finding.quote:
            finding = finding.model_copy(update={"quote": grounded})

        current = best.get(finding.sign_id)
        if current is None or VALUE_PRIORITY[finding.value] > VALUE_PRIORITY[current.value]:
            best[finding.sign_id] = finding

    return list(best.values())


# Below this length, one extra or missing character is too large a share of the quote.
_MIN_INDEL_QUOTE_LEN = 8


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercased text with collapsed spaces, and the original index of each kept character."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    chars: list[str] = []
    origins: list[int] = []
    i = 0
    while i < len(text) and text[i].isspace():
        i += 1
    pending_space = False
    while i < len(text):
        if text[i].isspace():
            pending_space = True
            i += 1
            continue
        if pending_space and chars:
            chars.append(" ")
            origins.append(i)
            pending_space = False
        chars.append(text[i].lower())
        origins.append(i)
        i += 1
    return "".join(chars), origins


def _normalise(text: str) -> str:
    return _normalise_with_map(text)[0]


def _one_extra_character(shorter: str, longer: str) -> bool:
    """True when longer is shorter plus exactly one inserted character."""
    if len(longer) != len(shorter) + 1:
        return False
    i = 0
    skipped = False
    for ch in longer:
        if i < len(shorter) and shorter[i] == ch:
            i += 1
        elif not skipped:
            skipped = True
        else:
            return False
    return skipped and i == len(shorter)


def grounded_quote(quote: str, message: str) -> str | None:
    """Return text from the message that supports this quote, or None if it is not grounded.

    An exact match keeps the model's quote. One inserted or deleted character, in a quote
    of at least 8 characters, is accepted and replaced with the farmer's own spelling.
    A substituted character is rejected.
    """
    q = _normalise(quote).strip(" \"'.,!?")
    if not q:
        return None
    normalised, origins = _normalise_with_map(message)
    if q in normalised:
        return quote
    if len(q) < _MIN_INDEL_QUOTE_LEN:
        return None
    return _indel_span(message, normalised, origins, q)


def _indel_span(message: str, normalised: str, origins: list[int], quote: str) -> str | None:
    qlen = len(quote)
    for span_len in (qlen - 1, qlen + 1):
        if span_len <= 0 or span_len > len(normalised):
            continue
        for start in range(len(normalised) - span_len + 1):
            span = normalised[start:start + span_len]
            matches = (
                _one_extra_character(span, quote)
                if span_len < qlen
                else _one_extra_character(quote, span)
            )
            if matches:
                end = start + span_len - 1
                return message[origins[start]:origins[end] + 1]
    return None


def quote_in_message(quote: str, message: str) -> bool:
    """True if the quote appears in the message, ignoring case, spacing and edge punctuation."""
    return grounded_quote(quote, message) is not None
