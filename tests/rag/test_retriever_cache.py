"""Tests for the Retriever's instance-level LRU cache (P18)."""

from __future__ import annotations

from pathlib import Path

import pytest
from clarion.config import Settings, load_customer
from clarion.pipelines.unstructured import chunk_rules_dir
from clarion.rag.embeddings import TfidfEmbedder
from clarion.rag.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _retriever(tmp_path: Path, *, cache_size: int) -> Retriever:
    cfg = load_customer(
        "ophthalmology",
        settings=Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=DATA_DIR),
    )
    chunks = chunk_rules_dir(cfg.rules_path)
    embedder = TfidfEmbedder()
    embedder.fit([c.text for c in chunks])
    # Build with a small custom cache_size; bypass build_and_save since
    # that path doesn't expose the knob (and we don't want to widen the
    # API surface just for tests).
    import faiss

    vectors = embedder.embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    return Retriever(index=index, chunks=chunks, embedder=embedder, cache_size=cache_size)


def test_cache_hit_returns_same_result_and_increments_hit_counter(tmp_path: Path) -> None:
    r = _retriever(tmp_path, cache_size=4)
    first = r.retrieve("cataract surgery", k=3)
    second = r.retrieve("cataract surgery", k=3)
    assert first == second
    stats = r.cache_stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.size == 1


def test_cache_distinguishes_query_and_k(tmp_path: Path) -> None:
    r = _retriever(tmp_path, cache_size=4)
    r.retrieve("cataract surgery", k=3)
    r.retrieve("cataract surgery", k=4)  # different k -> different key
    r.retrieve("glaucoma", k=3)  # different query
    stats = r.cache_stats()
    assert stats.hits == 0
    assert stats.misses == 3
    assert stats.size == 3


def test_cache_evicts_oldest_when_capacity_exceeded(tmp_path: Path) -> None:
    r = _retriever(tmp_path, cache_size=2)
    r.retrieve("q1", k=3)
    r.retrieve("q2", k=3)
    r.retrieve("q3", k=3)  # evicts q1
    stats = r.cache_stats()
    assert stats.size == 2
    # Now hitting q2 should still be a cache hit; q1 should be a miss.
    r.retrieve("q2", k=3)
    assert r.cache_stats().hits == 1
    r.retrieve("q1", k=3)
    assert r.cache_stats().misses == 4


def test_cache_size_zero_disables_caching(tmp_path: Path) -> None:
    r = _retriever(tmp_path, cache_size=0)
    r.retrieve("cataract surgery", k=3)
    r.retrieve("cataract surgery", k=3)
    stats = r.cache_stats()
    # Hit + miss counters stay at 0 when cache is disabled.
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.size == 0
    assert stats.capacity == 0


def test_cache_clear_resets_counters_and_entries(tmp_path: Path) -> None:
    r = _retriever(tmp_path, cache_size=4)
    r.retrieve("cataract surgery", k=3)
    r.retrieve("cataract surgery", k=3)
    assert r.cache_stats().size == 1
    r.cache_clear()
    stats = r.cache_stats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.size == 0


def test_empty_query_short_circuits_before_cache(tmp_path: Path) -> None:
    r = _retriever(tmp_path, cache_size=4)
    r.retrieve("   ", k=3)
    r.retrieve("", k=3)
    stats = r.cache_stats()
    # Empty queries return [] without touching the cache.
    assert stats.size == 0
    assert stats.misses == 0


def test_negative_cache_size_rejected(tmp_path: Path) -> None:
    cfg = load_customer(
        "ophthalmology",
        settings=Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=DATA_DIR),
    )
    chunks = chunk_rules_dir(cfg.rules_path)
    embedder = TfidfEmbedder()
    embedder.fit([c.text for c in chunks])
    import faiss

    vectors = embedder.embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)

    with pytest.raises(ValueError, match="cache_size"):
        Retriever(index=index, chunks=chunks, embedder=embedder, cache_size=-1)
