"""Core data models for bifrost-rag."""

from __future__ import annotations

from pydantic import BaseModel, Field, NonNegativeFloat


class Document(BaseModel):
    """A retrievable unit of text."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Query(BaseModel):
    """A retrieval query."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=1000)


class RetrievalResult(BaseModel):
    """A single retrieved document with its similarity score."""

    document: Document
    score: NonNegativeFloat
    rank: int = Field(ge=1)
