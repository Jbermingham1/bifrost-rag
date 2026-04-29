"""Retrieval quality metrics: Precision@K, Recall@K, F1@K, MRR.

All metrics operate on rank-ordered lists of retrieved document IDs and a set of
relevant ground-truth IDs. Returns are bounded in `[0.0, 1.0]`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant.

    Returns 0.0 when k <= 0 or when no documents are retrieved.
    """
    if k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents present in the top-k retrieved.

    Returns 0.0 when there are no relevant documents (recall is undefined).
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(list(retrieved)[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def f1_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    relevant_list = list(relevant)
    precision = precision_at_k(retrieved, relevant_list, k)
    recall = recall_at_k(retrieved, relevant_list, k)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal of the rank of the first relevant document, or 0.0 if none.

    Ranks are 1-indexed: the first document has rank 1.
    """
    relevant_set = set(relevant)
    for index, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (index + 1)
    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[Sequence[str]],
    relevant_per_query: Iterable[Iterable[str]],
) -> float:
    """Mean of `reciprocal_rank` across queries.

    Args:
        rankings: One rank-ordered retrieved list per query.
        relevant_per_query: Iterables of relevant document IDs per query, aligned to `rankings`.

    Raises:
        ValueError: if the two iterables produce different lengths.
    """
    rankings_list = [list(r) for r in rankings]
    relevant_list = [list(r) for r in relevant_per_query]
    if len(rankings_list) != len(relevant_list):
        raise ValueError("rankings and relevant_per_query must have the same length")
    if not rankings_list:
        return 0.0
    return sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(rankings_list, relevant_list, strict=True)
    ) / len(rankings_list)
