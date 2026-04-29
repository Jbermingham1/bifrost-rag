# Changelog

All notable changes to bifrost-rag will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-29

### Added
- Initial release.
- `RAGPipeline` orchestrating embed → retrieve → generate stages.
- Pluggable component protocols: `Embedder`, `Retriever`, `Generator`.
- Reference embedders: `HashEmbedder` (deterministic, dependency-free) and `VoyageEmbedder` (optional, requires `voyageai`).
- Reference retriever: `InMemoryRetriever` (numpy cosine similarity, top-K).
- Reference generators: `EchoGenerator` (test-only) and `ClaudeGenerator` (requires `anthropic`).
- Retrieval evaluation metrics: Precision@K, Recall@K, F1@K, MRR.
- `RetrievalEvalSuite` for end-to-end pipeline evaluation against labelled query sets.
- CLI entry point `bifrost-rag` with `index`, `query`, and `eval` subcommands.
- pyright strict-mode type checking, hypothesis property-based tests, 80% coverage gate.

### Notes
- Composes with [`bifrost-eval`](https://github.com/Jbermingham1/bifrost-eval) — RAG metrics adhere to the same `Metric` protocol so they can be embedded in larger evaluation suites.
