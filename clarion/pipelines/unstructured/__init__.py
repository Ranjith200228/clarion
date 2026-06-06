"""Unstructured pipeline — rules markdown → chunks → embeddings → FAISS."""

from clarion.pipelines.unstructured.chunker import RuleChunk, chunk_rules_dir

__all__ = ["RuleChunk", "chunk_rules_dir"]
