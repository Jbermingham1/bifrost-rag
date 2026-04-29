"""Retriever protocol and an in-memory cosine-similarity reference implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from bifrost_rag.embedder import Embedder
from bifrost_rag.models import Document, RetrievalResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class Retriever(Protocol):
    """Indexes documents and retrieves the top-K most similar to a query."""

    def index(self, documents: list[Document]) -> None:
        """Index a corpus. Replaces any previously indexed corpus."""
        ...

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return up to top_k results in descending score order."""
        ...

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        ...


class InMemoryRetriever:
    """Cosine-similarity retriever backed by a numpy matrix.

    Suitable for corpora up to the low hundreds of thousands of documents. Beyond that,
    swap in a dedicated vector index (FAISS, Qdrant, pgvector) by implementing the
    `Retriever` protocol.

    Embeddings are L2-normalised at index time so search reduces to a single matmul.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._documents: list[Document] = []
        self._matrix: NDArray[np.float32] = np.zeros((0, embedder.dimensions), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._documents)

    def index(self, documents: list[Document]) -> None:
        self._documents = list(documents)
        if not documents:
            self._matrix = np.zeros((0, self._embedder.dimensions), dtype=np.float32)
            return
        vectors = self._embedder.embed([d.text for d in documents])
        self._matrix = self._l2_normalise(vectors)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._documents:
            return []
        query_vector = self._l2_normalise(self._embedder.embed([query]))[0]
        scores = self._matrix @ query_vector
        k = min(top_k, len(self._documents))
        # argpartition for top-k, then sort just those k for ordering
        partitioned = np.argpartition(-scores, k - 1)[:k]
        ordered = partitioned[np.argsort(-scores[partitioned])]
        return [
            RetrievalResult(
                document=self._documents[int(idx)],
                score=float(max(0.0, scores[int(idx)])),
                rank=rank + 1,
            )
            for rank, idx in enumerate(ordered)
        ]

    @staticmethod
    def _l2_normalise(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).astype(np.float32)
        safe_norms = np.where(norms > 0, norms, np.float32(1.0)).astype(np.float32)
        return (matrix / safe_norms).astype(np.float32)
