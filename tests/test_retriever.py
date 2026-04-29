"""Tests for the in-memory retriever."""

from __future__ import annotations

import pytest

from bifrost_rag.embedder import HashEmbedder
from bifrost_rag.models import Document
from bifrost_rag.retriever import InMemoryRetriever


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text)


class TestInMemoryRetriever:
    def test_starts_empty(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        assert retriever.size == 0

    def test_index_sets_size(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index(
            [_doc("a", "apple banana cherry"), _doc("b", "elephant eagle owl")]
        )
        assert retriever.size == 2

    def test_index_replaces_corpus(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index([_doc("a", "first corpus document text")])
        retriever.index([_doc("b", "second corpus replaces first")])
        assert retriever.size == 1
        results = retriever.search("second corpus", top_k=1)
        assert results[0].document.id == "b"

    def test_index_empty_clears(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index([_doc("a", "to be removed")])
        retriever.index([])
        assert retriever.size == 0
        assert retriever.search("anything", top_k=5) == []

    def test_search_returns_ranked_results(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index(
            [
                _doc("apple", "red apple fruit sweet juicy"),
                _doc("banana", "yellow banana fruit sweet potassium"),
                _doc("car", "automobile transportation engine wheels metal"),
            ]
        )
        results = retriever.search("red apple fruit sweet juicy", top_k=3)
        assert len(results) == 3
        assert results[0].document.id == "apple"
        # Ranks are 1-indexed and ascending
        for i, result in enumerate(results):
            assert result.rank == i + 1
        # Scores must be descending
        assert results[0].score >= results[1].score >= results[2].score

    def test_search_top_k_truncates(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index(
            [_doc(f"doc-{i}", f"sample text content number {i} alpha beta") for i in range(10)]
        )
        results = retriever.search("sample text content", top_k=3)
        assert len(results) == 3

    def test_search_top_k_capped_at_corpus_size(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index([_doc("a", "alpha beta gamma")])
        results = retriever.search("alpha beta gamma", top_k=100)
        assert len(results) == 1

    def test_search_empty_corpus_returns_empty(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        assert retriever.search("anything", top_k=5) == []

    def test_search_top_k_zero_rejected(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index([_doc("a", "alpha beta gamma")])
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retriever.search("query", top_k=0)

    def test_search_scores_non_negative(self) -> None:
        retriever = InMemoryRetriever(HashEmbedder())
        retriever.index(
            [_doc("a", "alpha beta gamma delta"), _doc("b", "epsilon zeta eta theta")]
        )
        results = retriever.search("zebra giraffe lion tiger", top_k=2)
        for result in results:
            assert result.score >= 0.0
