# bifrost-rag

[![CI](https://github.com/Jbermingham1/bifrost-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Jbermingham1/bifrost-rag/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked: pyright strict](https://img.shields.io/badge/pyright-strict-brightgreen)](https://github.com/microsoft/pyright)

**A small, opinionated RAG pipeline for Python — embed, retrieve, generate — paired with a retrieval-quality evaluation harness (Precision@K, Recall@K, F1, MRR).**

## Why

Most "production RAG" libraries are either thin wrappers over a single vector DB or sprawling multi-vendor frameworks. Neither is what you want when you're iterating on retrieval quality.

`bifrost-rag` is small enough to read in one sitting and structured around three protocols — `Embedder`, `Retriever`, `Generator` — that you can swap in any direction. Pair it with the bundled `RetrievalEvalSuite` to grade your changes against a labelled set of queries, and you have a feedback loop that turns "I think this is better" into "Precision@5 went from 0.62 to 0.78."

## Architecture

```
                ┌──────────────────────────────────────────────────┐
                │                    RAGPipeline                   │
                │                                                  │
   query ─────▶ │  Embedder  ─▶  Retriever  ─▶  Generator  ──────▶ │ ─▶ answer
                │                  │                               │       +
   corpus ────▶ │      (index time)│                               │     refs
                │                  ▼                               │
                │           top-k results                          │
                └──────────────────────────────────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────────────────┐
                │                RetrievalEvalSuite                │
                │  Precision@K · Recall@K · F1@K · MRR · latency   │
                └──────────────────────────────────────────────────┘
```

## How

```bash
pip install bifrost-rag
```

```python
from bifrost_rag import (
    Document,
    EvalCase,
    HashEmbedder,
    InMemoryRetriever,
    RAGPipeline,
    RetrievalEvalSuite,
)
from bifrost_rag.generator import ClaudeGenerator  # or EchoGenerator for tests

# 1. Build a pipeline. Components are pluggable — bring your own embedder, retriever,
#    or generator by implementing the Embedder / Retriever / Generator protocols.
pipeline = RAGPipeline(
    retriever=InMemoryRetriever(HashEmbedder()),
    generator=ClaudeGenerator(),  # requires bifrost-rag[anthropic] + ANTHROPIC_API_KEY
)

# 2. Index a corpus.
corpus = [
    Document(id="apple",    text="A red, sweet, crisp fruit grown on a tree."),
    Document(id="banana",   text="A yellow, sweet, soft fruit rich in potassium."),
    Document(id="elephant", text="A grey mammal with a long trunk and large ears."),
]
pipeline.index(corpus)

# 3. Query.
result = pipeline.run("what fruit is rich in potassium?", top_k=2)
print(result.answer)
print(f"retrieval: {result.retrieval_ms:.1f}ms · generation: {result.generation_ms:.1f}ms")

# 4. Evaluate retrieval quality on a labelled set.
suite = RetrievalEvalSuite(
    corpus=corpus,
    cases=[
        EvalCase(query_id="q1", query="potassium fruit", relevant_ids=["banana"]),
        EvalCase(query_id="q2", query="long trunk grey",  relevant_ids=["elephant"]),
    ],
    k=2,
)
report = suite.evaluate(pipeline)
print(report.summary())
# {'num_cases': 2, 'k': 2, 'precision_at_k': 0.5, 'recall_at_k': 1.0, 'f1_at_k': 0.667, 'mrr': 1.0, ...}
```

## CLI

```bash
# Index a JSONL corpus (one {"id":"...","text":"..."} per line).
bifrost-rag index examples/corpus.jsonl

# Run a single query.
bifrost-rag query examples/corpus.jsonl "what fruit is rich in potassium?" --top-k 3

# Run an evaluation suite.
bifrost-rag eval examples/corpus.jsonl examples/cases.jsonl --k 5
```

## When to use this (and when not to)

| Use `bifrost-rag` when… | Reach for something else when… |
|---|---|
| You want to iterate on retrieval quality and need fast metric feedback | You need millions of vectors and a managed vector DB (use Qdrant, pgvector, Pinecone) |
| You want a small library you can read end-to-end | You want a full RAG product with chunking, ingestion pipelines, and a UI |
| You're embedding RAG into a larger Python service | You're building a no-code RAG experience |
| You want to swap embedder/retriever/generator independently | You're locked into a specific vendor stack |

The library is deliberately small. The protocols are the product.

## Components

### Embedders
- `HashEmbedder` — deterministic, dependency-free. Test fixture only.
- `VoyageEmbedder` — production. Requires `bifrost-rag[voyage]` and `VOYAGE_API_KEY`.
- Or implement the `Embedder` protocol with sentence-transformers, OpenAI, Cohere, anything else.

### Retrievers
- `InMemoryRetriever` — numpy cosine similarity. Suitable up to the low hundreds of thousands of documents.
- Implement the `Retriever` protocol to plug in FAISS, Qdrant, pgvector, or any other index.

### Generators
- `EchoGenerator` — deterministic stub. Test-only.
- `ClaudeGenerator` — Anthropic Claude with retrieved-context injection. Requires `bifrost-rag[anthropic]` and `ANTHROPIC_API_KEY`.

### Metrics
- `precision_at_k`, `recall_at_k`, `f1_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`.
- Property-based tested with [Hypothesis](https://hypothesis.readthedocs.io/) — output bounded in `[0.0, 1.0]` over arbitrary inputs.

## Composes with bifrost-eval

`bifrost-rag` is one of three repos in the same family:

- [`bifrost-rag`](https://github.com/Jbermingham1/bifrost-rag) — the pipeline + retrieval metrics (this repo).
- [`bifrost-eval`](https://github.com/Jbermingham1/bifrost-eval) — broader evaluation toolkit for multi-step agent pipelines.
- [`bifrost-monitor`](https://github.com/Jbermingham1/bifrost-monitor) — runtime observability for AI agents.

Use them together: `bifrost-rag` to build and grade the retrieval layer, `bifrost-eval` to grade the broader workflow, `bifrost-monitor` to watch it in production.

## Engineering bar

- **pyright strict** type checking
- **80% test coverage gate** enforced in CI
- **Hypothesis property-based tests** on metrics
- **Minimal runtime dependencies**: `pydantic` and `numpy` only. LLM and embedding providers are optional extras.
- **MIT licensed** — use freely.

## License

MIT — see [LICENSE](LICENSE).
