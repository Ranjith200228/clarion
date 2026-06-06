"""Unstructured retrieval: chunks → embeddings → FAISS → top-k."""

from clarion.rag.builder import build_customer_index, load_customer_retriever
from clarion.rag.embeddings import (
    Embedder,
    OpenAIEmbedder,
    TfidfEmbedder,
    default_embedder,
)
from clarion.rag.retriever import RetrievalHit, Retriever

__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "RetrievalHit",
    "Retriever",
    "TfidfEmbedder",
    "build_customer_index",
    "default_embedder",
    "load_customer_retriever",
]
