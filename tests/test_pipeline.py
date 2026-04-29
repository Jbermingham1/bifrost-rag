"""End-to-end pipeline tests."""

from __future__ import annotations

import pytest

from bifrost_rag.embedder import HashEmbedder
from bifrost_rag.generator import EchoGenerator
from bifrost_rag.models import Document
from bifrost_rag.pipeline import RAGPipeline
from bifrost_rag.retriever import InMemoryRetriever


def _build() -> RAGPipeline:
    return RAGPipeline(
        retriever=InMemoryRetriever(HashEmbedder()),
        generator=EchoGenerator(),
    )


class TestRAGPipeline:
    def test_index_and_run(self) -> None:
        pipeline = _build()
        pipeline.index(
            [
                Document(id="a", text="apple banana cherry fruit"),
                Document(id="b", text="elephant zebra giraffe animal"),
            ]
        )
        result = pipeline.run("apple banana", top_k=2)
        assert len(result.retrieved) == 2
        assert "Q: apple banana" in result.answer

    def test_latency_tracked(self) -> None:
        pipeline = _build()
        pipeline.index([Document(id="a", text="alpha beta gamma delta epsilon")])
        result = pipeline.run("alpha beta", top_k=1)
        assert result.retrieval_ms >= 0.0
        assert result.generation_ms >= 0.0
        assert result.total_ms == result.retrieval_ms + result.generation_ms

    def test_empty_query_rejected(self) -> None:
        pipeline = _build()
        pipeline.index([Document(id="a", text="alpha beta")])
        with pytest.raises(ValueError, match="query must not be empty"):
            pipeline.run("   ", top_k=5)

    def test_top_k_zero_rejected(self) -> None:
        pipeline = _build()
        pipeline.index([Document(id="a", text="alpha beta")])
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            pipeline.run("query", top_k=0)

    def test_retriever_property(self) -> None:
        pipeline = _build()
        assert pipeline.retriever is not None

    def test_generator_property(self) -> None:
        pipeline = _build()
        assert pipeline.generator is not None

    def test_top_k_caps_at_corpus_size(self) -> None:
        pipeline = _build()
        pipeline.index(
            [Document(id="a", text="alpha beta gamma"), Document(id="b", text="delta epsilon zeta")]
        )
        result = pipeline.run("alpha", top_k=10)
        assert len(result.retrieved) == 2
