"""The random-vector embedder: the text-processing track's Stage-3 random floor.

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3, "Sanity floors") asks for **two**
floors, so that a "win" can be checked against chance:

- the **label-permutation** floor (done in CS-5): the fit side's labels are
  destroyed, so the arm has no label information at all;
- the **random-vector** floor (this module, CS-6d): the arm is fitted on
  **random vectors instead of the model's embeddings**, so a "win" can be
  checked against what the *training design* alone buys.

The floor replaces the embedding **sides** of the feature pack: the pack's
lexical, question-part and question-type features never used an embedding, so
they stay real and the floor measures exactly one thing — how much of a
learned arm's number survives when the embedding geometry is replaced by
noise. The run's own baseline arm (the full-text cosine) keeps the model's
embeddings, so the arm-vs-cosine comparison stays meaningful.

Three properties the floor relies on:

- the vector of a text is a **deterministic function of (seed, kind, text)**,
  so the pack's pair-locality and its memoisation hold (the same text always
  embeds to the same vector, on every side, in every run) and the floor is
  reproducible from the seed alone;
- the vector is a **unit vector of the model's dimension**, so the feature
  shapes and the cosine scale are unchanged — the floor is a like-for-like
  replacement, not a different feature set;
- the vector is **never written to the embedding cache**: it is free to
  compute, and it must not pollute the model's cache (the cache key carries
  the model's id, so a random vector under the model's key would be served to
  a later real run).

No model, no network, no I/O: numpy + hashlib only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from cluster_rag.embed.base import (
    KINDS,
    EmbedderInfo,
    ensure_embedding_contract,
    validate_texts,
)

# The draw rule, recorded in the artifact: the recipe is part of the run's
# provenance, so a reader can reproduce any vector from the seed alone.
DRAW_RULE = "sha256(seed|kind|text) -> default_rng(seed).uniform(-1, 1) -> L2-normalized"


class RandomEmbedder:
    """An :class:`~cluster_rag.embed.base.Embedder` that carries no text signal.

    The vector of a text is a seeded random unit vector, drawn from
    ``sha256(seed | kind | text)``. Two texts therefore get two unrelated
    vectors, and the same text always gets the same one — the property the
    feature pack's memoisation relies on.

    Args:
        dimension: The dimension of the vectors (the model's dimension, so the
            feature shapes and the cosine scale are unchanged).
        seed: The seed of the draw. The same seed reproduces the same vectors.
        model_id: The id the embedder reports in its provenance. It defaults to
            ``random-<seed>`` and is deliberately **not** a real model's id:
            the random vectors must never share a cache key with a real run.
    """

    def __init__(self, *, dimension: int, seed: int, model_id: str | None = None) -> None:
        """Build the embedder (no model is loaded, nothing is read or written).

        Raises:
            ValueError: If the dimension is not positive.
        """
        if dimension <= 0:
            raise ValueError(f"the random embedder needs a positive dimension, got {dimension}")
        self._seed = int(seed)
        self._info = EmbedderInfo(
            model_id=model_id or f"random-{self._seed}",
            backend="random",
            dimension=int(dimension),
            revision_or_digest=None,
            prefixes={},
            extra={"seed": self._seed, "draw_rule": DRAW_RULE},
        )

    @property
    def info(self) -> EmbedderInfo:
        """Interface facts of this embedder (the draw's seed and rule)."""
        return self._info

    @property
    def seed(self) -> int:
        """The seed of the draw."""
        return self._seed

    def encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        """The seeded random unit vectors of the texts, in input order.

        Args:
            texts: The texts to embed (blank ones are rejected).
            kind: The encode kind ("document" / "query" / "label"). It is part
                of the draw's key, so the same text gets different vectors on
                different sides.

        Raises:
            ValueError: If the kind is unknown or a text is blank.
        """
        if kind not in KINDS:
            raise ValueError(f"Unknown kind {kind!r}; expected one of {KINDS}")
        checked = validate_texts(texts)
        vectors = np.zeros((len(checked), self._info.dimension), dtype=np.float32)
        for row, text in enumerate(checked):
            digest = hashlib.sha256(f"{self._seed}\x00{kind}\x00{text}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            vectors[row] = rng.uniform(-1.0, 1.0, size=self._info.dimension)
        return ensure_embedding_contract(vectors, dimension=self._info.dimension)

    def provenance(self) -> dict[str, Any]:
        """The JSON-friendly provenance of the floor (recorded in the artifact)."""
        return {
            "mode": "random",
            "seed": self._seed,
            "dimension": self._info.dimension,
            "draw_rule": DRAW_RULE,
            "cached": False,
            "note": (
                "the feature pack's embedding sides are random unit vectors: a "
                "deterministic function of (seed, kind, text). The pack's "
                "lexical, question-part and question-type features do not use "
                "an embedding, so they stay real, and the run's baseline arm "
                "keeps the model's embeddings. The vectors are never written "
                "to the embedding cache."
            ),
        }
