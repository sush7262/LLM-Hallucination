"""
Pydantic models for the API request/response contract.
Matches the spec in PROJECT_SPEC.md Section 4.
"""

from pydantic import BaseModel, Field
from typing import Literal


class VerifyRequest(BaseModel):
    """Request body for POST /verify."""

    answer: str = Field(
        ...,
        description="The RAG-generated answer to verify.",
        min_length=1,
    )
    passages: list[str] = Field(
        ...,
        description="Retrieved passages (raw, unchunked). Backend chunks internally.",
        min_length=1,
    )
    decomposition_mode: Literal["rule_based", "llm"] = Field(
        default="rule_based",
        description="Claim decomposition strategy.",
    )


class ClaimResult(BaseModel):
    """Verification result for a single claim."""

    claim: str
    label: Literal["supported", "unsupported", "contradicted"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(
        ...,
        description="The aligned passage text used for the verdict.",
    )


class VerifyResponse(BaseModel):
    """Response body for POST /verify."""

    claims: list[ClaimResult]
    decision: str = Field(
        ...,
        description=(
            "One of: 'PASS (fully supported)' | "
            "'PASS WITH FLAGS (some claims unverifiable)' | "
            "'REJECT / REGENERATE (contradiction found)'"
        ),
    )
    latency_ms: int = Field(
        ...,
        description="End-to-end pipeline latency in milliseconds.",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    device: str
