"""End-to-end builder: customer config → chunks → embeddings → FAISS index.

This is what the ingest CLI calls for the unstructured stage. Returns the
number of chunks indexed so the CLI can print a tidy summary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss

from clarion.config import CustomerConfig
from clarion.pipelines.unstructured import chunk_rules_dir
from clarion.rag.embeddings import (
    Embedder,
    OpenAIEmbedder,
    TfidfEmbedder,
    default_embedder,
)
from clarion.rag.retriever import Retriever

log = logging.getLogger(__name__)

# OpenAI text-embedding-3-small is exactly 1536-dim. TF-IDF on our rule
# corpora is far smaller (typically 500 to 4096 depending on vocabulary).
# We use this constant to decide which embedder built the index on disk.
_OPENAI_EMBED_DIM = 1536


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
    """Load a prebuilt retriever, matching the embedder to the index.

    Critical fix: we used to call ``default_embedder()`` here, which picks
    OpenAI when ``OPENAI_API_KEY`` is set and TF-IDF otherwise. That's
    wrong at *load* time — the on-disk FAISS index has a fixed dim baked
    in by whatever embedder *built* it. If the build was TF-IDF (500 to
    4096 dims typical) and the runtime then flipped to OpenAI (1536), the
    query vector and the index disagree and FAISS asserts on .search()
    with a 500 to the client.

    The fix: peek at the FAISS index's stored dimension and pick the
    matching embedder, ignoring env vars. If the caller passes an
    explicit ``embedder`` we still honor it (tests + scripts know what
    they're doing).
    """
    index_dir = customer_index_dir(cfg.customer_id, data_dir)
    if embedder is not None:
        return Retriever.load(index_dir, embedder)

    embedder = _embedder_matching_index(index_dir)
    return Retriever.load(index_dir, embedder)


def _embedder_matching_index(index_dir: Path) -> Embedder:
    """Read the persisted FAISS index's dim and pick an embedder that
    produces vectors of that same dim.

    Raises FileNotFoundError if the index isn't there — callers (the
    session manager) already swallow that and run the agent without
    retrieval.
    """
    index_path = index_dir / "rules.faiss"
    if not index_path.is_file():
        raise FileNotFoundError(f"No prebuilt FAISS index at {index_path}")
    index = faiss.read_index(str(index_path))
    persisted_dim = int(index.d)

    if persisted_dim == _OPENAI_EMBED_DIM:
        # 1536 is unambiguous — only text-embedding-3-small produces that.
        # If the key isn't set at runtime we can't query OpenAI, so we'd
        # rather fail clearly than silently regress to a wrong embedder.
        log.info(
            "load: index_dim=%d -> using OpenAIEmbedder", persisted_dim
        )
        return OpenAIEmbedder()

    # Anything else is TF-IDF. TfidfEmbedder.fit() will rebuild the
    # vocabulary from the persisted chunks (Retriever.load() calls fit
    # with the corpus). The fitted vocab size must match persisted_dim;
    # if it doesn't, the deeper assertion later is the right error
    # rather than silently embedding into a different space here.
    log.info(
        "load: index_dim=%d -> using TfidfEmbedder", persisted_dim
    )
    return TfidfEmbedder()
