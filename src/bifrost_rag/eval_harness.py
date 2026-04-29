"""Evaluation harness for end-to-end retrieval pipelines.

Given a corpus, a labelled set of evaluation cases (query + relevant document IDs),
and a `RAGPipeline`, runs each case through the pipeline and aggregates retrieval
quality metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from bifrost_rag.metrics import (
    f1_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from bifrost_rag.models import Document
from bifrost_rag.pipeline import RAGPipeline


class EvalCase(BaseModel):
    """A single evaluation case: a query plus the relevant document IDs."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_ids: list[str] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Per-case outcome captured during evaluation."""

    query_id: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision: float
    recall: float
    f1: float
    reciprocal_rank: float
    retrieval_ms: float
    generation_ms: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregated evaluation results across all cases."""

    k: int
    cases: list[CaseOutcome] = field(default_factory=list[CaseOutcome])

    @property
    def num_cases(self) -> int:
        return len(self.cases)

    @property
    def mean_precision(self) -> float:
        return self._mean(c.precision for c in self.cases)

    @property
    def mean_recall(self) -> float:
        return self._mean(c.recall for c in self.cases)

    @property
    def mean_f1(self) -> float:
        return self._mean(c.f1 for c in self.cases)

    @property
    def mean_reciprocal_rank(self) -> float:
        return self._mean(c.reciprocal_rank for c in self.cases)

    @property
    def mean_retrieval_ms(self) -> float:
        return self._mean(c.retrieval_ms for c in self.cases)

    @property
    def mean_generation_ms(self) -> float:
        return self._mean(c.generation_ms for c in self.cases)

    def summary(self) -> dict[str, float | int]:
        """One-line summary suitable for logging or JSON export."""
        return {
            "num_cases": self.num_cases,
            "k": self.k,
            "precision_at_k": round(self.mean_precision, 4),
            "recall_at_k": round(self.mean_recall, 4),
            "f1_at_k": round(self.mean_f1, 4),
            "mrr": round(self.mean_reciprocal_rank, 4),
            "retrieval_ms_p50": round(self.mean_retrieval_ms, 2),
            "generation_ms_p50": round(self.mean_generation_ms, 2),
        }

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        items = list(values)
        if not items:
            return 0.0
        return sum(items) / len(items)


class RetrievalEvalSuite:
    """Run an `EvalCase` set through a `RAGPipeline` and produce an `EvalReport`."""

    def __init__(self, corpus: list[Document], cases: list[EvalCase], k: int = 5) -> None:
        if not corpus:
            raise ValueError("corpus must not be empty")
        if not cases:
            raise ValueError("cases must not be empty")
        if k < 1:
            raise ValueError("k must be >= 1")
        self._corpus = list(corpus)
        self._cases = list(cases)
        self._k = k

    @property
    def k(self) -> int:
        return self._k

    def evaluate(self, pipeline: RAGPipeline) -> EvalReport:
        pipeline.index(self._corpus)
        outcomes: list[CaseOutcome] = []
        for case in self._cases:
            result = pipeline.run(case.query, top_k=self._k)
            retrieved_ids = [r.document.id for r in result.retrieved]
            outcomes.append(
                CaseOutcome(
                    query_id=case.query_id,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=list(case.relevant_ids),
                    precision=precision_at_k(retrieved_ids, case.relevant_ids, self._k),
                    recall=recall_at_k(retrieved_ids, case.relevant_ids, self._k),
                    f1=f1_at_k(retrieved_ids, case.relevant_ids, self._k),
                    reciprocal_rank=mean_reciprocal_rank(
                        [retrieved_ids], [case.relevant_ids]
                    ),
                    retrieval_ms=result.retrieval_ms,
                    generation_ms=result.generation_ms,
                )
            )
        return EvalReport(k=self._k, cases=outcomes)
