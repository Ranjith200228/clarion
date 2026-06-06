"""Embedders for the RAG pipeline.

Two implementations:

* ``TfidfEmbedder`` — sklearn TF-IDF + L2 normalization, returns dense
  ``np.float32`` so FAISS can index it directly. Zero network calls, no
  API key — the spec's "TF-IDF fallback" and the default everywhere tests
  and CI run.
* ``OpenAIEmbedder`` — text-embedding-3-small, opt-in when ``OPENAI_API_KEY``
  is set. Stateless (no ``fit`` needed).

Both implement the ``Embedder`` Protocol so callers (index builder,
retriever) can swap implementations without changing code.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

# Shape (n, dim), dtype float32, L2-normalized.
EmbeddingMatrix = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a dense, L2-normalized vector matrix."""

    @property
    def dim(self) -> int:
        """Embedding dimension (after fit, for stateful embedders)."""

    def fit(self, corpus: list[str]) -> None:
        """Build any state needed before ``embed`` (no-op for stateless)."""

    def embed(self, texts: list[str]) -> EmbeddingMatrix:
        """Return an ``(n, dim)`` float32 matrix, L2-normalized."""


def _l2_normalize(mat: NDArray[np.float32] | NDArray[np.float64]) -> EmbeddingMatrix:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: EmbeddingMatrix = (mat / norms).astype(np.float32, copy=False)
    return out


class TfidfEmbedder:
    """TF-IDF + dense L2 norm. Default embedder."""

    def __init__(self, *, max_features: int = 4096) -> None:
        # min_df=1 because rule corpora are small (~30 chunks) — every word
        # may legitimately appear once.
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._fitted = False
        self._dim = 0

    @property
    def dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.dim is unknown until fit() is called")
        return self._dim

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("TfidfEmbedder.fit requires a non-empty corpus")
        self._vec.fit(corpus)
        self._dim = len(self._vec.vocabulary_)
        self._fitted = True

    def embed(self, texts: list[str]) -> EmbeddingMatrix:
        if not self._fitted:
            raise RuntimeError("Call fit(corpus) before embed()")
        sparse = self._vec.transform(texts)
        dense = sparse.toarray().astype(np.float32, copy=False)
        return _l2_normalize(dense)


class OpenAIEmbedder:
    """text-embedding-3-small (1536 dims), opt-in when an API key is set.

    Stateless — ``fit`` is a no-op so it interchanges with TfidfEmbedder.
    """

    _MODEL = "text-embedding-3-small"
    _DIM = 1536

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        # Import lazily so the TF-IDF path never pays the openai import cost.
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAIEmbedder requires OPENAI_API_KEY (env or constructor arg).")
        self._client = OpenAI(api_key=key)
        self._model = model or self._MODEL

    @property
    def dim(self) -> int:
        return self._DIM

    def fit(self, corpus: list[str]) -> None:
        return

    def embed(self, texts: list[str]) -> EmbeddingMatrix:
        if not texts:
            return np.zeros((0, self._DIM), dtype=np.float32)
        resp = self._client.embeddings.create(model=self._model, input=texts)
        mat = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # OpenAI returns unit-norm but we re-normalize defensively in case
        # the dimension is reduced or the model changes.
        return _l2_normalize(mat)


def default_embedder() -> Embedder:
    """Return ``OpenAIEmbedder`` if a key is set, else ``TfidfEmbedder``."""
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    return TfidfEmbedder()
