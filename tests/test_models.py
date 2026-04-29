"""Validation tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bifrost_rag.models import Document, Query, RetrievalResult


class TestDocument:
    def test_minimal_document(self) -> None:
        doc = Document(id="doc-1", text="hello world")
        assert doc.id == "doc-1"
        assert doc.text == "hello world"
        assert doc.metadata == {}

    def test_metadata_preserved(self) -> None:
        doc = Document(id="doc-1", text="x", metadata={"source": "wiki", "lang": "en"})
        assert doc.metadata["source"] == "wiki"
        assert doc.metadata["lang"] == "en"

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(id="", text="hello")

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(id="doc-1", text="")


class TestQuery:
    def test_default_top_k(self) -> None:
        q = Query(id="q-1", text="what is rag")
        assert q.top_k == 5

    def test_custom_top_k(self) -> None:
        q = Query(id="q-1", text="x", top_k=20)
        assert q.top_k == 20

    def test_top_k_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Query(id="q-1", text="x", top_k=0)

    def test_top_k_too_large_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Query(id="q-1", text="x", top_k=10_000)


class TestRetrievalResult:
    def test_construction(self) -> None:
        doc = Document(id="d", text="t")
        r = RetrievalResult(document=doc, score=0.87, rank=1)
        assert r.document.id == "d"
        assert r.score == 0.87
        assert r.rank == 1

    def test_negative_score_rejected(self) -> None:
        doc = Document(id="d", text="t")
        with pytest.raises(ValidationError):
            RetrievalResult(document=doc, score=-0.1, rank=1)

    def test_zero_rank_rejected(self) -> None:
        doc = Document(id="d", text="t")
        with pytest.raises(ValidationError):
            RetrievalResult(document=doc, score=0.5, rank=0)
