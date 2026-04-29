"""Embedder protocol and reference implementations.

The framework is embedder-agnostic — any callable that maps text to a fixed-dimensional
vector satisfies the `Embedder` protocol. Two reference implementations ship in-tree:

- `HashEmbedder` — deterministic, dependency-free. Suitable for tests, CI, and offline
  experiments. Not semantically meaningful; do not use in production.
- `VoyageEmbedder` — wraps the Voyage AI embeddings API. Requires the `voyageai` extra
  (`pip install bifrost-rag[voyage]`) and a `VOYAGE_API_KEY` environment variable.

Bring your own embedder by implementing the `Embedder` protocol — for example, OpenAI
text-embedding-3, sentence-transformers, or Cohere embeddings.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a fixed-dimensional vector."""

    @property
    def dimensions(self) -> int:
        """The dimensionality of the embedding space."""
        ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts. Returns a (len(texts), dimensions) float32 array."""
        ...


class HashEmbedder:
    """Deterministic hash-based embedder for tests and offline use.

    Maps text to a fixed-dimensional vector by hashing fixed-width token windows and
    bucketing the hashes. Output is L2-normalised. Two identical inputs always produce
    identical vectors; semantically related inputs do NOT cluster meaningfully — this
    is a test fixture, not a production embedder.
    """

    def __init__(self, dimensions: int = 384, window: int = 3) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        if window < 1:
            raise ValueError("window must be >= 1")
        self._dimensions = dimensions
        self._window = window

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            for j in range(len(tokens) - self._window + 1):
                window_text = " ".join(tokens[j : j + self._window])
                digest = hashlib.blake2b(window_text.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self._dimensions
                sign = 1.0 if (digest[4] & 1) else -1.0
                vectors[i, bucket] += sign
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class VoyageEmbedder:
    """Production embedder backed by the Voyage AI embeddings API.

    Requires `pip install bifrost-rag[voyage]` and `VOYAGE_API_KEY` set in the
    environment. The default model `voyage-3` produces 1024-dimensional vectors.
    """

    def __init__(self, model: str = "voyage-3", api_key: str | None = None) -> None:
        try:
            import voyageai  # pyright: ignore[reportMissingImports,reportUnknownVariableType,reportMissingTypeStubs]
        except ImportError as exc:
            raise ImportError(
                "voyageai is required. Install with: pip install bifrost-rag[voyage]"
            ) from exc

        resolved_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "VOYAGE_API_KEY not set. Pass api_key=... or export the environment variable."
            )

        # voyageai SDK has no shipped type stubs; surface the boundary as `object`
        # and re-cast on use. Exercised via integration tests with a real API key.
        self._client: object = voyageai.Client(api_key=resolved_key)  # pyright: ignore[reportUnknownMemberType]
        self._model = model
        self._dimensions = self._probe_dimensions()

    def _probe_dimensions(self) -> int:
        result = self._client.embed(["probe"], model=self._model)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        first: list[float] = list(result.embeddings[0])  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        return len(first)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        result = self._client.embed(texts, model=self._model)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        embeddings: list[list[float]] = [list(row) for row in result.embeddings]  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType]
        return np.asarray(embeddings, dtype=np.float32)
