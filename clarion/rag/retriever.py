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
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import faiss

from clarion.pipelines.unstructured.chunker import RuleChunk
from clarion.rag.embeddings import Embedder

log = logging.getLogger(__name__)

_INDEX_FILENAME = "rules.faiss"
_META_FILENAME = "rules_meta.json"

# Long conversations hit the retriever repeatedly with semantically
# overlapping queries. A small per-instance LRU lets us skip the
# embedder + FAISS round-trip for verbatim repeats — embedder calls
# dominate latency for the OpenAI backend (~150 ms each) so even a
# modest hit rate pays for itself.
DEFAULT_CACHE_SIZE = 64


@dataclass(frozen=True)
class RetrievalHit:
    """One result returned by ``Retriever.retrieve``."""

    chunk: RuleChunk
    score: float  # cosine similarity in [-1, 1] (L2-normalized inputs)


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of the retriever's instance-level LRU.

    Surfaced for tests + the eval dashboard so we can prove the cache
    is doing useful work (and tune ``cache_size`` if hit rate stays
    low).
    """

    hits: int
    misses: int
    size: int
    capacity: int


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
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError(f"Index has {index.ntotal} vectors but {len(chunks)} chunks")
        if cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        self._index = index
        self._chunks = chunks
        self._embedder = embedder

        # Manual LRU. functools.lru_cache would work, but a hand-rolled
        # OrderedDict gives us inspectable stats + per-instance scoping
        # without playing games with method binding.
        self._cache_capacity = cache_size
        self._cache: OrderedDict[tuple[str, int], list[RetrievalHit]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    # ---------- query ----------

    def retrieve(self, query: str, *, k: int = 4) -> list[RetrievalHit]:
        """Return the top-``k`` chunks ranked by cosine similarity.

        Cached at the instance level keyed on ``(query, k)``. Empty
        queries + zero-vector indices short-circuit BEFORE the cache
        check so they never displace useful entries.
        """
        if not query.strip():
            return []
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)

        key = (query, k)
        if self._cache_capacity > 0 and key in self._cache:
            # Move-to-end on hit so the LRU ordering reflects recency.
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return list(self._cache[key])

        if self._cache_capacity > 0:
            self._cache_misses += 1

        qvec = self._embedder.embed([query])
        # We use IndexFlatIP on L2-normalized vectors → IP == cosine.
        scores, indices = self._index.search(qvec, k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            hits.append(RetrievalHit(chunk=self._chunks[idx], score=float(score)))

        if self._cache_capacity > 0:
            self._cache[key] = list(hits)
            if len(self._cache) > self._cache_capacity:
                # Drop the oldest entry.
                self._cache.popitem(last=False)

        return hits

    def cache_stats(self) -> CacheStats:
        """Inspectable snapshot — hits, misses, current size, capacity."""
        return CacheStats(
            hits=self._cache_hits,
            misses=self._cache_misses,
            size=len(self._cache),
            capacity=self._cache_capacity,
        )

    def cache_clear(self) -> None:
        """Reset the LRU. Useful when the underlying index is rebuilt."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

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


__all__ = ["CacheStats", "DEFAULT_CACHE_SIZE", "RetrievalHit", "Retriever"]
