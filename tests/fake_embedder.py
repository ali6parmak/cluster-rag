"""Deterministic fake embedder for unit tests (AGENTS.md testing rules).

Hash-based vectors: an identical (kind, text) pair always embeds to the same
(unit-norm) vector, and different pairs embed to different ones, which makes
retrieval-style assertions possible without network access or model downloads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from cluster_rag.embed.base import (
    KINDS,
    EmbedderInfo,
    l2_normalize,
    validate_texts,
)


class FakeEmbedder:
    """Minimal :class:`~cluster_rag.embed.base.Embedder` implementation.

    Vectors come from a seeded RNG over ``sha256(kind + "\\x00" + text)``.
    ``call_count`` / ``encode_calls`` record every ``encode`` invocation so
    tests can prove that the disk cache skips re-encoding on hits.
    """

    def __init__(
        self,
        *,
        model_id: str = "fake-embedder",
        dimension: int = 8,
        prefixes: Mapping[str, str] | None = None,
        revision_or_digest: str = "fake-revision",
    ) -> None:
        self._info = EmbedderInfo(
            model_id=model_id,
            backend="fake",
            dimension=dimension,
            revision_or_digest=revision_or_digest,
            prefixes=dict(prefixes or {}),
        )
        self.call_count = 0
        self.encode_calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def info(self) -> EmbedderInfo:
        """Interface facts of this fake embedder."""
        return self._info

    def encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        """Hash-deterministic embeddings in input order, float32, L2-normalized."""
        if kind not in KINDS:
            raise ValueError(f"Unknown kind {kind!r}; expected one of {KINDS}")
        texts = validate_texts(texts)
        self.call_count += 1
        self.encode_calls.append((kind, tuple(texts)))
        vectors = np.zeros((len(texts), self._info.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            seed = int(hashlib.sha256(f"{kind}\x00{text}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors[row] = rng.uniform(-1.0, 1.0, size=self._info.dimension)
        return l2_normalize(vectors)


class CountingEmbedder:
    """Wrapper that forwards to an inner embedder and records raw call sizes.

    Used to prove caching: after a warm run, the second identical call must
    leave ``total_texts_encoded`` unchanged.
    """

    def __init__(self, inner: FakeEmbedder) -> None:
        self._inner = inner
        self.total_texts_encoded = 0

    @property
    def info(self) -> EmbedderInfo:
        """Provenance comes from the wrapped embedder."""
        return self._inner.info

    def encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        """Forward and count the number of texts sent to the inner model."""
        self.total_texts_encoded += len(list(texts))
        return self._inner.encode(texts, kind)
