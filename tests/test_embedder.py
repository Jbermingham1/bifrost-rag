"""Tests for embedders."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bifrost_rag.embedder import HashEmbedder


class TestHashEmbedder:
    def test_default_dimensions(self) -> None:
        embedder = HashEmbedder()
        assert embedder.dimensions == 384

    def test_custom_dimensions(self) -> None:
        embedder = HashEmbedder(dimensions=128)
        assert embedder.dimensions == 128

    def test_dimensions_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be >= 8"):
            HashEmbedder(dimensions=4)

    def test_window_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            HashEmbedder(window=0)

    def test_embed_returns_correct_shape(self) -> None:
        embedder = HashEmbedder(dimensions=64)
        vectors = embedder.embed(["hello world", "another sample"])
        assert vectors.shape == (2, 64)
        assert vectors.dtype == np.float32

    def test_deterministic(self) -> None:
        embedder = HashEmbedder()
        first = embedder.embed(["repeatable input text here"])
        second = embedder.embed(["repeatable input text here"])
        np.testing.assert_array_equal(first, second)

    def test_different_inputs_different_outputs(self) -> None:
        embedder = HashEmbedder()
        a = embedder.embed(["alpha beta gamma delta"])
        b = embedder.embed(["completely unrelated text content"])
        # Cosine similarity should be < 1.0 for different windows
        sim = float(np.dot(a[0], b[0]))
        assert sim < 0.99

    def test_l2_normalised(self) -> None:
        embedder = HashEmbedder()
        vectors = embedder.embed(["hello world this is a test sentence"])
        norm = float(np.linalg.norm(vectors[0]))
        # Norm should be 1.0 (or 0.0 for empty/short inputs)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_text_yields_zero_vector(self) -> None:
        embedder = HashEmbedder()
        vectors = embedder.embed([""])
        assert vectors.shape == (1, 384)
        assert float(np.linalg.norm(vectors[0])) == 0.0

    def test_short_text_below_window_yields_zero_vector(self) -> None:
        embedder = HashEmbedder(window=3)
        # "hi" has 1 token, below window=3
        vectors = embedder.embed(["hi"])
        assert float(np.linalg.norm(vectors[0])) == 0.0

    def test_empty_batch(self) -> None:
        embedder = HashEmbedder(dimensions=128)
        vectors = embedder.embed([])
        assert vectors.shape == (0, 128)

    @given(st.lists(st.text(min_size=10, max_size=50), min_size=1, max_size=8))
    def test_property_norms_bounded(self, texts: list[str]) -> None:
        embedder = HashEmbedder(dimensions=64)
        vectors = embedder.embed(texts)
        for vec in vectors:
            norm = float(np.linalg.norm(vec))
            # Either zero (empty/short) or unit-length
            assert norm == 0.0 or abs(norm - 1.0) < 1e-5
