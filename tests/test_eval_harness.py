"""Tests for the retrieval evaluation harness."""

from __future__ import annotations

import pytest

from bifrost_rag.embedder import HashEmbedder
from bifrost_rag.eval_harness import EvalCase, RetrievalEvalSuite
from bifrost_rag.generator import EchoGenerator
from bifrost_rag.models import Document
from bifrost_rag.pipeline import RAGPipeline
from bifrost_rag.retriever import InMemoryRetriever


def _build_pipeline() -> RAGPipeline:
    return RAGPipeline(
        retriever=InMemoryRetriever(HashEmbedder()),
        generator=EchoGenerator(),
    )


def _corpus() -> list[Document]:
    return [
        Document(id="apple", text="red apple fruit sweet juicy crisp"),
        Document(id="banana", text="yellow banana fruit sweet potassium tropical"),
        Document(id="elephant", text="grey elephant animal large mammal trunk"),
        Document(id="zebra", text="black white striped zebra animal african"),
    ]


class TestEvalCaseValidation:
    def test_valid_case(self) -> None:
        case = EvalCase(query_id="q1", query="apple", relevant_ids=["apple"])
        assert case.query == "apple"

    def test_empty_relevant_rejected(self) -> None:
        with pytest.raises(Exception):
            EvalCase(query_id="q1", query="x", relevant_ids=[])


class TestRetrievalEvalSuite:
    def test_empty_corpus_rejected(self) -> None:
        with pytest.raises(ValueError, match="corpus must not be empty"):
            RetrievalEvalSuite(
                corpus=[],
                cases=[EvalCase(query_id="q1", query="x", relevant_ids=["a"])],
            )

    def test_empty_cases_rejected(self) -> None:
        with pytest.raises(ValueError, match="cases must not be empty"):
            RetrievalEvalSuite(corpus=_corpus(), cases=[])

    def test_k_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            RetrievalEvalSuite(
                corpus=_corpus(),
                cases=[EvalCase(query_id="q1", query="x", relevant_ids=["a"])],
                k=0,
            )

    def test_evaluate_runs_all_cases(self) -> None:
        suite = RetrievalEvalSuite(
            corpus=_corpus(),
            cases=[
                EvalCase(query_id="q1", query="red apple fruit sweet", relevant_ids=["apple"]),
                EvalCase(
                    query_id="q2",
                    query="elephant mammal trunk grey",
                    relevant_ids=["elephant"],
                ),
            ],
            k=3,
        )
        report = suite.evaluate(_build_pipeline())
        assert report.num_cases == 2
        assert report.k == 3

    def test_summary_contains_expected_keys(self) -> None:
        suite = RetrievalEvalSuite(
            corpus=_corpus(),
            cases=[EvalCase(query_id="q1", query="apple fruit", relevant_ids=["apple"])],
            k=3,
        )
        summary = suite.evaluate(_build_pipeline()).summary()
        expected = {"num_cases", "k", "precision_at_k", "recall_at_k", "f1_at_k", "mrr"}
        assert expected <= summary.keys()

    def test_metrics_bounded(self) -> None:
        suite = RetrievalEvalSuite(
            corpus=_corpus(),
            cases=[
                EvalCase(query_id="q1", query="red apple fruit sweet", relevant_ids=["apple"]),
                EvalCase(
                    query_id="q2",
                    query="elephant mammal trunk grey",
                    relevant_ids=["elephant"],
                ),
            ],
            k=2,
        )
        report = suite.evaluate(_build_pipeline())
        assert 0.0 <= report.mean_precision <= 1.0
        assert 0.0 <= report.mean_recall <= 1.0
        assert 0.0 <= report.mean_f1 <= 1.0
        assert 0.0 <= report.mean_reciprocal_rank <= 1.0

    def test_empty_report_means_zero(self) -> None:
        from bifrost_rag.eval_harness import EvalReport

        report = EvalReport(k=5)
        assert report.mean_precision == 0.0
        assert report.mean_recall == 0.0
