"""Generator protocol and reference implementations.

The generator stage produces a final answer given a query and the retrieved context.
Two implementations ship in-tree:

- `EchoGenerator` — returns a deterministic stub answer. Test-only.
- `ClaudeGenerator` — calls the Anthropic Messages API with retrieved context injected
  as a structured system prompt. Requires the `anthropic` extra and an `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from bifrost_rag.models import RetrievalResult


@runtime_checkable
class Generator(Protocol):
    """Produces a final answer from a query and the retrieved context."""

    def generate(self, query: str, context: list[RetrievalResult]) -> str:
        ...


class EchoGenerator:
    """Deterministic test generator. Returns a structured echo of inputs.

    Output format:
        Q: <query>
        Context: <id1>; <id2>; ...

    Useful for asserting that the pipeline wired the retrieved context through correctly
    without paying for LLM calls in tests.
    """

    def generate(self, query: str, context: list[RetrievalResult]) -> str:
        ids = "; ".join(r.document.id for r in context) if context else "(none)"
        return f"Q: {query}\nContext: {ids}"


class ClaudeGenerator:
    """Anthropic Claude generator with retrieved-context injection.

    Builds a system prompt containing the retrieved documents as numbered references
    and asks Claude to answer the query citing the references it used.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            import anthropic  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
        except ImportError as exc:
            raise ImportError(
                "anthropic is required. Install with: pip install bifrost-rag[anthropic]"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key=... or export the environment variable."
            )

        # Anthropic SDK is fully typed at the public-API surface but pyright in strict
        # mode flags external SDK call sites because we don't ship anthropic stubs.
        # The client is exercised in integration tests with a real API key.
        self._client: object = anthropic.Anthropic(api_key=resolved_key)  # pyright: ignore[reportUnknownMemberType]
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, query: str, context: list[RetrievalResult]) -> str:
        system = self._build_system_prompt(context)
        message = self._client.messages.create(  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        # Anthropic returns a list of content blocks; concatenate text blocks
        parts: list[str] = []
        for block in message.content:  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            block_type: object = getattr(block, "type", None)  # pyright: ignore[reportUnknownArgumentType]
            if block_type == "text":
                text_value: object = getattr(block, "text", "")  # pyright: ignore[reportUnknownArgumentType]
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts).strip()

    @staticmethod
    def _build_system_prompt(context: list[RetrievalResult]) -> str:
        if not context:
            return (
                "You are a helpful assistant. No retrieved context is available; "
                "answer from general knowledge and acknowledge the absence of references."
            )
        references = "\n\n".join(
            f"[{r.rank}] (id={r.document.id}, score={r.score:.3f})\n{r.document.text}"
            for r in context
        )
        return (
            "You are a helpful assistant answering from a fixed set of references.\n"
            "Cite reference numbers in square brackets (e.g. [1], [2]) for any claim you make.\n"
            "If the references do not contain the answer, say so explicitly.\n\n"
            "References:\n"
            f"{references}"
        )
