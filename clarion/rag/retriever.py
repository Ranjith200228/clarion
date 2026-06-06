"""FAISS-backed retriever.

A retriever is built once (``build_index``) and then loaded from disk by the
agent at runtime. Persistence layout::

    <data_dir>/<customer_id>/rules.faiss        # FAISS index
    <data_dir>/<customer_id>/rules_meta.json    # per-chunk metadata
    <data_dir>/<customer_id>/rules_embedder.json # which embedder built it

The metadata file is the source of truth for what each FAISS row means
(chunk text, source path, heading). The retriever never reads markdown
at query time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss

from clarion.pipelines.unstructured.chunker import RuleChunk
from clarion.rag.embeddings import Embedder

_INDEX_FILENAME = "rules.faiss"
_META_FILENAME = "rules_meta.json"


@dataclass(frozen=True)
class RetrievalHit:
    """One result returned by ``Retriever.retrieve``."""

    chunk: RuleChunk
    score: float  # cosine similarity in [-1, 1] (L2-normalized inputs)


class Retriever:
    """Loads a prebuilt FAISS index + chunk metadata and answers queries.

    Use ``Retriever.build_and_save`` to create one, then ``Retriever.load``
    to use it at runtime.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[RuleChunk],
        embedder: Embedder,
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError(f"Index has {index.ntotal} vectors but {len(chunks)} chunks")
        self._index = index
        self._chunks = chunks
        self._embedder = embedder

    # ---------- query ----------

    def retrieve(self, query: str, *, k: int = 4) -> list[RetrievalHit]:
        """Return the top-``k`` chunks ranked by cosine similarity."""
        if not query.strip():
            return []
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        qvec = self._embedder.embed([query])
        # We use IndexFlatIP on L2-normalized vectors → IP == cosine.
        scores, indices = self._index.search(qvec, k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            hits.append(RetrievalHit(chunk=self._chunks[idx], score=float(score)))
        return hits

    # ---------- build & persist ----------

    @classmethod
    def build_and_save(
        cls,
        chunks: list[RuleChunk],
        embedder: Embedder,
        *,
        out_dir: Path,
    ) -> Retriever:
        """Embed ``chunks``, build a FAISS index, write it to ``out_dir``."""
        if not chunks:
            raise ValueError("Cannot build a retriever from zero chunks")
        out_dir.mkdir(parents=True, exist_ok=True)

        texts = [c.text for c in chunks]
        embedder.fit(texts)
        vectors = embedder.embed(texts)
        if vectors.shape[0] != len(chunks):
            raise RuntimeError(
                f"Embedder returned {vectors.shape[0]} vectors for {len(chunks)} chunks"
            )

        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        faiss.write_index(index, str(out_dir / _INDEX_FILENAME))
        _write_meta(out_dir / _META_FILENAME, chunks)
        return cls(index=index, chunks=chunks, embedder=embedder)

    @classmethod
    def load(cls, src_dir: Path, embedder: Embedder) -> Retriever:
        """Load a prebuilt index. ``embedder`` must match what built it."""
        index_path = src_dir / _INDEX_FILENAME
        meta_path = src_dir / _META_FILENAME
        if not index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"No prebuilt RAG index at {src_dir}. Run the ingest CLI first."
            )
        index = faiss.read_index(str(index_path))
        chunks = _read_meta(meta_path)
        # Fit embedder so it can transform queries. For TF-IDF we need the
        # original corpus; for stateless embedders (OpenAI) this is a no-op.
        embedder.fit([c.text for c in chunks])
        return cls(index=index, chunks=chunks, embedder=embedder)


# ---------- metadata I/O ----------


def _write_meta(path: Path, chunks: list[RuleChunk]) -> None:
    payload = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "heading": c.heading,
        }
        for c in chunks
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_meta(path: Path) -> list[RuleChunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [RuleChunk(**item) for item in raw]


__all__ = ["RetrievalHit", "Retriever"]
