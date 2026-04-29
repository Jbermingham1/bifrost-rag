"""Command-line interface: index, query, eval.

Run `bifrost-rag --help` for usage. The CLI is intentionally minimal — its purpose is
to demonstrate the pipeline end-to-end, not to replace a full RAG product. Real
deployments should import the library and wire it into their own application.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bifrost_rag.embedder import HashEmbedder
from bifrost_rag.eval_harness import EvalCase, RetrievalEvalSuite
from bifrost_rag.generator import EchoGenerator
from bifrost_rag.models import Document
from bifrost_rag.pipeline import RAGPipeline
from bifrost_rag.retriever import InMemoryRetriever


def _load_corpus(path: Path) -> list[Document]:
    """Load a corpus from a JSONL file. Each line is `{"id": "...", "text": "..."}`."""
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")
    documents: list[Document] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        documents.append(Document.model_validate(payload))
    return documents


def _load_cases(path: Path) -> list[EvalCase]:
    """Load eval cases from a JSONL file. Each line is an `EvalCase` payload."""
    if not path.exists():
        raise FileNotFoundError(f"cases file not found: {path}")
    cases: list[EvalCase] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        cases.append(EvalCase.model_validate(payload))
    return cases


def _build_pipeline() -> RAGPipeline:
    """The default reference pipeline: HashEmbedder + InMemory retriever + Echo generator."""
    embedder = HashEmbedder()
    retriever = InMemoryRetriever(embedder)
    generator = EchoGenerator()
    return RAGPipeline(retriever=retriever, generator=generator)


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline()
    pipeline.index(_load_corpus(Path(args.corpus)))
    result = pipeline.run(args.query, top_k=args.top_k)
    payload = {
        "query": args.query,
        "answer": result.answer,
        "retrieved": [
            {
                "id": r.document.id,
                "rank": r.rank,
                "score": round(r.score, 4),
                "text": r.document.text,
            }
            for r in result.retrieved
        ],
        "retrieval_ms": round(result.retrieval_ms, 2),
        "generation_ms": round(result.generation_ms, 2),
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    suite = RetrievalEvalSuite(
        corpus=_load_corpus(Path(args.corpus)),
        cases=_load_cases(Path(args.cases)),
        k=args.k,
    )
    report = suite.evaluate(_build_pipeline())
    sys.stdout.write(json.dumps(report.summary(), indent=2) + "\n")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    documents = _load_corpus(Path(args.corpus))
    pipeline = _build_pipeline()
    pipeline.index(documents)
    sys.stdout.write(
        json.dumps({"indexed": pipeline.retriever.size, "corpus": str(args.corpus)}, indent=2)
        + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    description = (__doc__ or "bifrost-rag CLI").splitlines()[0]
    parser = argparse.ArgumentParser(prog="bifrost-rag", description=description)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Validate and index a corpus.")
    p_index.add_argument("corpus", help="Path to JSONL corpus file.")
    p_index.set_defaults(func=_cmd_index)

    p_query = sub.add_parser("query", help="Run a single query through the pipeline.")
    p_query.add_argument("corpus", help="Path to JSONL corpus file.")
    p_query.add_argument("query", help="The query text.")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5).")
    p_query.set_defaults(func=_cmd_query)

    p_eval = sub.add_parser("eval", help="Run an evaluation suite and report metrics.")
    p_eval.add_argument("corpus", help="Path to JSONL corpus file.")
    p_eval.add_argument("cases", help="Path to JSONL eval cases file.")
    p_eval.add_argument(
        "--k", type=int, default=5, help="K for Precision@K / Recall@K (default: 5)."
    )
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
