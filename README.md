# V-Vet symptom extraction

A small FastAPI service with one endpoint, `POST /extract`. It turns a farmer's WhatsApp
message about a sick animal into structured findings from a fixed list of signs. It also
decides, in plain code, whether the case should be escalated.

Author: Lordwill Kandiro

## How to run

Needs Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then put your key in LLM_API_KEY
uvicorn app.main:app --reload
```

**LLM provider.** Any OpenAI-compatible chat API works, set through three env vars:
`LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL`. The default is Groq's free tier with
`openai/gpt-oss-120b`. You can get a free key at https://console.groq.com/keys. To use
another provider, change the base URL and model, for example OpenRouter
(`https://openrouter.ai/api/v1`), OpenAI (`https://api.openai.com/v1`) or Gemini's
OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`).

```bash
curl -s localhost:8000/extract -H "Content-Type: application/json" \
  -d '{"message": "There'\''s foam coming from her nose and she can'\''t get up."}'
```

```json
{
  "findings": [
    {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
    {"sign_id": "unable_to_stand", "value": "yes", "quote": "she can't get up"}
  ],
  "escalate": true,
  "reason": "Danger sign(s) reported: froth_nose_mouth (yes), unable_to_stand (yes)."
}
```

Interactive docs are at http://localhost:8000/docs.

**Tests.** The tests mock the LLM, so they need no network and no key:

```bash
pytest
```

**Eval against the real model** (needs a key): `python -m eval.run_eval`

## Layout

```
app/
  signs.py        the sign list, descriptions and danger signs
  models.py       request/response schemas (Pydantic)
  llm.py          LLMClient protocol + OpenAI-compatible client
  extractor.py    prompt, JSON parsing, validation and grounding of findings
  escalation.py   escalation rule (plain code)
  main.py         FastAPI app
tests/            pytest suite with a FakeLLM
eval/             hand-labelled samples + accuracy script
```

## Decisions

- **The LLM extracts and code decides.** The model only maps words to sign IDs. Validation
  and escalation are deterministic Python, so they are testable and auditable.
- **Validation happens in layers.** `sign_id` is an enum and `value` is a `Literal`, so
  Pydantic rejects anything off the list. Each finding is validated on its own, so one bad
  item is dropped (and logged) without throwing away the rest.
- **Quotes must be grounded.** A finding is kept only if its quote actually appears in the
  message. The check ignores case, spacing, curly apostrophes and surrounding punctuation.
  A quote of at least 8 characters that has one extra or missing character is still kept,
  and the stored quote is replaced with the farmer's own spelling. A changed letter is
  rejected. This stops a doubled letter from dropping a real finding, and still blocks
  symptoms the farmer never mentioned.
- **One finding per sign.** If the model repeats a sign, the more cautious value wins
  (`yes` > `not_sure` > `no`).
- **"no" only when it is explicit.** The prompt asks for `no` only when the farmer says a
  sign is absent ("she is still eating"). So "the calf is fine" does not produce nine
  negatives.
- **not_sure escalates.** The brief requires it. For danger signs, a false alarm costs much
  less than a missed case.
- **Failure is loud.** Bad JSON gets one retry. If it still fails, or the provider errors,
  the API returns `502`. A missing key returns `503`. I chose not to return empty findings
  on failure, because that would look like "no danger signs" and could hide an urgent case.
- **The provider is swappable.** The extractor depends on a one-method `LLMClient` protocol.
  The real client uses the OpenAI-compatible API with JSON mode and `temperature=0`, so it
  works with free tiers. Tests use a `FakeLLM`.
- **Mixed-language input.** The prompt says messages may mix English, Shona and Ndebele. It
  includes a few Shona hints (`kudya`, `mbandabvu`, `zvikwekwe`). Quotes stay in the
  farmer's original words and are never translated.

## Measuring extraction accuracy

`eval/run_eval.py` prints per-sign precision and recall, value agreement and escalation
accuracy against a labelled JSONL file. The bundled `eval/samples.jsonl` only holds the
five brief messages, which I labelled by hand, so it is a smoke test. A real measurement
would use:

1. **A labelled set from real, anonymised messages.** It should include English, Shona,
   Ndebele and code-switching, spelling mistakes, and voice-note transcripts. Vets would
   label it. Measuring agreement between annotators shows how ambiguous the task really is.
2. **Per-sign precision and recall, with recall on the danger signs as the headline
   number.** Missing froth, recumbency or cloudy eyes is the costly error.
3. **Value-level agreement** (yes / no / not_sure) on signs that were correctly detected,
   plus a confusion matrix, because yes vs not_sure decides escalation.
4. **Escalation accuracy end to end**, and especially the false-negative rate.
5. **Quote grounding rate**: how often findings get dropped for ungrounded quotes or
   unknown IDs. A rising rate means the prompt or model is drifting.
6. **Regression runs** in CI whenever the prompt or model changes, with results tracked
   over time and broken down by language.

## With more time

- Grow the labelled set with vets and native speakers, and tune the prompt with
  Shona/Ndebele synonyms per sign, checked by speakers rather than guessed.
- Return the farmer's exact substring as the quote (with character offsets) instead of
  the model's copy.
- Use provider-native JSON-schema / structured outputs where available, and add a fallback
  provider for rate limits and outages.
- Add a rule-based safety net for danger-sign keywords (e.g. "foam", "can't stand"), so an
  LLM miss on an urgent case is less likely.
- Add an async client, timeouts per request, structured logging without storing farmers'
  personal data, and a basic Dockerfile.
