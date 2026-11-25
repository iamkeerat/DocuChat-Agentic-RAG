"""Pydantic request and response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    """Payload for ``POST /ask``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Natural-language question to answer from the indexed documents.",
        examples=["What is the refund policy?"],
    )

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceChunk(BaseModel):
    """A retrieved passage that supports an answer."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="The retrieved passage text.")
    source: str = Field(..., description="Original filename of the document.")
    page: int | None = Field(
        default=None,
        ge=1,
        description="1-based page number, when the source is paginated.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score in the range 0-1 (higher is better).",
    )


class AgentStep(BaseModel):
    """One observable step in the Researcher/Critic workflow."""

    model_config = ConfigDict(extra="forbid")

    agent: Literal["researcher", "critic"]
    action: str = Field(..., description="Draft, review, revision, or fallback.")
    status: Literal["completed", "approved", "flagged", "revised", "fallback"]
    detail: str | None = Field(default=None, description="Short human-readable note.")
    duration_ms: int = Field(..., ge=0)


class AskResponse(BaseModel):
    """Verified answer plus the evidence and agent trace."""

    answer: str
    verified: bool = Field(
        ...,
        description=(
            "True only when the Critic Agent approved the answer against the "
            "retrieved source passages."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Critic Agent confidence in its verdict.",
    )
    attempts: int = Field(
        default=0,
        ge=0,
        description="Number of Researcher drafts produced (0 when no LLM is configured).",
    )
    critic_feedback: str | None = Field(
        default=None,
        description="Critic findings when an answer is flagged or unverified.",
    )
    sources: list[SourceChunk] = Field(default_factory=list)
    model: str = Field(
        ...,
        description=(
            "Provider/model used for generation, 'extractive' for the retrieval-only "
            "fallback, or 'none' when nothing was retrieved."
        ),
    )
    workflow: list[AgentStep] = Field(
        default_factory=list,
        description="Observable Researcher and Critic steps for this question.",
    )


class IngestResponse(BaseModel):
    """Result of indexing a single PDF."""

    document_id: str = Field(..., description="Stable SHA-256 prefix of the uploaded file.")
    filename: str
    pages: int = Field(..., ge=0)
    chunks_added: int = Field(..., ge=0)


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: Literal["ok"]
    version: str
    environment: str
    indexed_chunks: int = Field(..., ge=0)
