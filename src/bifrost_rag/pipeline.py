"""End-to-end RAG pipeline: embed -> retrieve -> generate."""

from __future__ import annotations

import time
from dataclasses import dataclass

from bifrost_rag.generator import Generator
from bifrost_rag.models import Document, RetrievalResult
from bifrost_rag.retriever import Retriever


@dataclass(frozen=True, slots=True)
class RAGResult:
    """The output of a single pipeline run."""

    answer: str
    retrieved: list[RetrievalResult]
    retrieval_ms: float
    generation_ms: float

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


class RAGPipeline:
    """Composes a Retriever and a Generator into an end-to-end RAG flow.

    Indexing is delegated to the retriever; the pipeline itself is stateless across
    queries (apart from the retriever's index). Latency is measured per stage so
    downstream evaluators can attribute slow runs.
    """

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def index(self, documents: list[Document]) -> None:
        self._retriever.index(documents)

    def run(self, query: str, top_k: int = 5) -> RAGResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        retrieval_start = time.perf_counter()
        retrieved = self._retriever.search(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        generation_start = time.perf_counter()
        answer = self._generator.generate(query, retrieved)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0

        return RAGResult(
            answer=answer,
            retrieved=retrieved,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def generator(self) -> Generator:
        return self._generator
