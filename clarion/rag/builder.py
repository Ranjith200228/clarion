"""End-to-end builder: customer config → chunks → embeddings → FAISS index.

This is what the ingest CLI calls for the unstructured stage. Returns the
number of chunks indexed so the CLI can print a tidy summary.
"""

from __future__ import annotations

from pathlib import Path

from clarion.config import CustomerConfig
from clarion.pipelines.unstructured import chunk_rules_dir
from clarion.rag.embeddings import Embedder, default_embedder
from clarion.rag.retriever import Retriever


def customer_index_dir(customer_id: str, data_dir: Path) -> Path:
    return data_dir / customer_id


def build_customer_index(
    cfg: CustomerConfig,
    *,
    data_dir: Path,
    embedder: Embedder | None = None,
) -> int:
    """Chunk, embed, and persist a FAISS index for one customer."""
    embedder = embedder or default_embedder()
    chunks = chunk_rules_dir(cfg.rules_path)
    if not chunks:
        raise RuntimeError(
            f"No markdown rules found under {cfg.rules_path}. "
            f"Add files before running unstructured ingest."
        )
    out_dir = customer_index_dir(cfg.customer_id, data_dir)
    Retriever.build_and_save(chunks, embedder=embedder, out_dir=out_dir)
    return len(chunks)


def load_customer_retriever(
    cfg: CustomerConfig,
    *,
    data_dir: Path,
    embedder: Embedder | None = None,
) -> Retriever:
    """Load a prebuilt retriever for one customer."""
    embedder = embedder or default_embedder()
    return Retriever.load(customer_index_dir(cfg.customer_id, data_dir), embedder)
