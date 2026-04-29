"""Tests for generators (the EchoGenerator reference implementation).

ClaudeGenerator is not unit-tested here — it requires a live Anthropic API key and is
covered by manual smoke testing. Its construction-time validation IS tested below.
"""

from __future__ import annotations

import pytest

from bifrost_rag.generator import ClaudeGenerator, EchoGenerator
from bifrost_rag.models import Document, RetrievalResult


def _result(doc_id: str, rank: int = 1) -> RetrievalResult:
    document = Document(id=doc_id, text=f"text-{doc_id}")
    return RetrievalResult(document=document, score=0.5, rank=rank)


class TestEchoGenerator:
    def test_no_context(self) -> None:
        gen = EchoGenerator()
        answer = gen.generate("what is rag", [])
        assert "Q: what is rag" in answer
        assert "Context: (none)" in answer

    def test_with_context(self) -> None:
        gen = EchoGenerator()
        answer = gen.generate("query", [_result("a", 1), _result("b", 2)])
        assert "Q: query" in answer
        assert "a; b" in answer

    def test_preserves_order(self) -> None:
        gen = EchoGenerator()
        answer = gen.generate("q", [_result("z", 1), _result("a", 2), _result("m", 3)])
        assert answer.endswith("z; a; m")


class TestClaudeGeneratorConstruction:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Skip if anthropic package not installed — its absence is a separate code path
        pytest.importorskip("anthropic")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            ClaudeGenerator(api_key=None)

    def test_explicit_api_key_accepted(self) -> None:
        pytest.importorskip("anthropic")
        # Constructing with a fake key should succeed; only actual `.generate()` calls hit the API
        generator = ClaudeGenerator(api_key="sk-fake-test-key-not-real")
        assert generator is not None


class TestClaudeGeneratorPromptShape:
    def test_no_context_system_prompt(self) -> None:
        # Use the static method directly — no API client needed
        prompt = ClaudeGenerator._build_system_prompt([])
        assert "No retrieved context is available" in prompt

    def test_with_context_system_prompt(self) -> None:
        prompt = ClaudeGenerator._build_system_prompt([_result("a", 1), _result("b", 2)])
        assert "References:" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Cite reference numbers" in prompt
