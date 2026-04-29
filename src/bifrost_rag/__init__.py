"""bifrost-rag — RAG pipeline with vector retrieval and an evaluation harness."""

from bifrost_rag.embedder import Embedder, HashEmbedder
from bifrost_rag.eval_harness import EvalCase, EvalReport, RetrievalEvalSuite
from bifrost_rag.generator import EchoGenerator, Generator
from bifrost_rag.metrics import f1_at_k, mean_reciprocal_rank, precision_at_k, recall_at_k
from bifrost_rag.models import Document, Query, RetrievalResult
from bifrost_rag.pipeline import RAGPipeline, RAGResult
from bifrost_rag.retriever import InMemoryRetriever, Retriever

__all__ = [
    "Document",
    "EchoGenerator",
    "Embedder",
    "EvalCase",
    "EvalReport",
    "Generator",
    "HashEmbedder",
    "InMemoryRetriever",
    "Query",
    "RAGPipeline",
    "RAGResult",
    "RetrievalEvalSuite",
    "RetrievalResult",
    "Retriever",
    "f1_at_k",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]

__version__ = "0.1.0"
