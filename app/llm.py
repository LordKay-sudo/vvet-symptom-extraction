"""A tiny LLM interface so the extractor can be tested with a fake client."""

import os
from typing import Protocol

from openai import OpenAI, OpenAIError

# Defaults point at Groq's free tier (OpenAI-compatible). Override with env vars
# to use OpenRouter, OpenAI, Gemini's OpenAI-compatible endpoint, a local model, etc.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text reply (expected to be a JSON object)."""
        ...


class LLMConfigError(RuntimeError):
    """Raised when the LLM is not configured (e.g. no API key)."""


class LLMError(RuntimeError):
    """Raised when the provider call fails (network, auth, rate limit, ...)."""


class OpenAICompatibleClient:
    def __init__(self, api_key: str | None, base_url: str, model: str, timeout: float = 30.0):
        # A missing key is reported when the client is used, not at startup, so the
        # app (and /docs) still runs and bad requests still get a normal 422.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout) if api_key else None
        self._model = model

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        return cls(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        )

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            raise LLMConfigError("LLM_API_KEY is not set. Copy .env.example to .env and add a key.")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except OpenAIError as exc:
            raise LLMError(f"LLM provider call failed: {exc}") from exc
        return response.choices[0].message.content or ""
