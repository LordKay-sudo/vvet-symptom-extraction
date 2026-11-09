"""Request and response schemas for the API."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from app.signs import SignId

Value = Literal["yes", "no", "not_sure"]


class ExtractRequest(BaseModel):
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]


class Finding(BaseModel):
    # Using the enum here means Pydantic rejects any sign_id not on the list.
    sign_id: SignId
    value: Value
    quote: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExtractResponse(BaseModel):
    findings: list[Finding]
    escalate: bool
    reason: str
