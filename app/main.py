"""FastAPI app exposing POST /extract."""

import logging
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException

from app.escalation import decide_escalation
from app.extractor import ExtractionError, Extractor
from app.llm import LLMConfigError, LLMError, OpenAICompatibleClient
from app.models import ExtractRequest, ExtractResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="V-Vet symptom extraction")


@lru_cache
def get_extractor() -> Extractor:
    return Extractor(OpenAICompatibleClient.from_env())


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, extractor: Extractor = Depends(get_extractor)) -> ExtractResponse:
    try:
        findings = extractor.extract(request.message)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ExtractionError, LLMError) as exc:
        # Fail loudly rather than return empty findings: an empty list would read
        # as "no danger signs" and could hide a case that needs a vet.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    escalate, reason = decide_escalation(findings)
    return ExtractResponse(findings=findings, escalate=escalate, reason=reason)
