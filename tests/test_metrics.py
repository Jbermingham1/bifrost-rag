"""Tests for retrieval quality metrics, including property-based fuzzing."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from bifrost_rag.metrics import (
    f1_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_no_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_partial(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "x"}, 3) == 1 / 3

    def test_truncation_to_k(self) -> None:
        # Top-1 of [a, x, b] vs {a} = 1/1
        assert precision_at_k(["a", "x", "b"], {"a"}, 1) == 1.0

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_k_negative_returns_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, -1) == 0.0

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 0.0


class TestRecallAtK:
    def test_full_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_recall(self) -> None:
        # Top-1 of [a, b] vs {a, b} = 1/2
        assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5

    def test_no_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], set(), 3) == 0.0

    def test_zero_recall(self) -> None:
        assert recall_at_k(["a", "b"], {"x"}, 3) == 0.0


class TestF1AtK:
    def test_perfect_f1(self) -> None:
        assert f1_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero_when_no_overlap(self) -> None:
        assert f1_at_k(["a"], {"b"}, 1) == 0.0

    def test_harmonic_mean(self) -> None:
        # P=1.0 (1/1), R=0.5 (1/2 relevant)
        # F1 = 2 * 1.0 * 0.5 / 1.5 = 0.6667
        assert abs(f1_at_k(["a"], {"a", "b"}, 1) - (2 / 3)) < 1e-9


class TestReciprocalRank:
    def test_first_relevant_at_rank_1(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_first_relevant_at_rank_3(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3

    def test_no_relevant(self) -> None:
        assert reciprocal_rank(["a", "b"], {"x"}) == 0.0

    def test_only_first_relevant_counts(self) -> None:
        # Both 'a' and 'c' are relevant; only the first one (rank 1) counts
        assert reciprocal_rank(["a", "b", "c"], {"a", "c"}) == 1.0


class TestMeanReciprocalRank:
    def test_mean_of_two_queries(self) -> None:
        # Q1: relevant at rank 1 (RR=1.0); Q2: relevant at rank 2 (RR=0.5)
        rankings = [["a"], ["x", "b"]]
        relevant = [{"a"}, {"b"}]
        assert mean_reciprocal_rank(rankings, relevant) == 0.75

    def test_empty(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0

    def test_length_mismatch_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


class TestPropertyBased:
    @given(st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=20))
    def test_precision_bounded(self, retrieved: list[str]) -> None:
        score = precision_at_k(retrieved, set(retrieved), len(retrieved) or 1)
        assert 0.0 <= score <= 1.0

    @given(
        st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=20),
        st.integers(min_value=1, max_value=20),
    )
    def test_recall_bounded(self, relevant: list[str], k: int) -> None:
        score = recall_at_k(relevant, set(relevant), k)
        assert 0.0 <= score <= 1.0

    @given(
        st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=20),
        st.integers(min_value=1, max_value=20),
    )
    def test_f1_bounded(self, retrieved: list[str], k: int) -> None:
        # Use retrieved as both retrieved and relevant — F1 must be bounded
        score = f1_at_k(retrieved, set(retrieved), k)
        assert 0.0 <= score <= 1.0

    @given(st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=20))
    def test_reciprocal_rank_bounded(self, retrieved: list[str]) -> None:
        score = reciprocal_rank(retrieved, set(retrieved) if retrieved else set())
        assert 0.0 <= score <= 1.0
