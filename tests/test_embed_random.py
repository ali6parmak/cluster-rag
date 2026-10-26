"""Tests for the random-vector embedder (the text-processing track's Stage-3 floor).

The random floor (TEXT_PROCESSING_PLAN.md section 3, Stage 3, "Sanity floors")
replaces the model's embeddings with noise, so the contracts under test are the
ones the floor relies on:

- the vector of a text is a **deterministic function of (seed, kind, text)**,
  so the feature pack's memoisation and its pair-locality hold;
- the vectors are **unit vectors of the model's dimension**, so the feature
  shapes and the cosine scale are unchanged;
- the embedder **never touches the embedding cache**: its model id is not a
  real model's id, so a random vector can never be served to a real run.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.embed.base import KINDS
from cluster_rag.embed.random import DRAW_RULE, RandomEmbedder

TEXTS = ("The Committee notes with concern.", "Answer choices:\n- Often")


def test_vectors_are_unit_norm_float32_of_the_models_dimension() -> None:
    """The floor is a like-for-like replacement: same dimension, same scale."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    vectors = embedder.encode(TEXTS, "document")
    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_the_same_text_always_gets_the_same_vector() -> None:
    """The pack's memoisation relies on it: one text, one vector, every time."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    first = embedder.encode(TEXTS, "document")
    second = embedder.encode(list(reversed(TEXTS)), "document")
    assert np.allclose(first[0], second[1])
    assert np.allclose(first[1], second[0])


def test_the_seed_is_part_of_the_key() -> None:
    """A different seed is a different random space."""
    one = RandomEmbedder(dimension=8, seed=42).encode(TEXTS, "document")
    other = RandomEmbedder(dimension=8, seed=7).encode(TEXTS, "document")
    assert not np.allclose(one, other)


def test_the_kind_is_part_of_the_key() -> None:
    """The same text on two sides is two different vectors."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    document = embedder.encode(TEXTS, "document")
    query = embedder.encode(TEXTS, "query")
    assert not np.allclose(document, query)


def test_different_texts_get_unrelated_vectors() -> None:
    """No text signal survives: two texts are two unrelated unit vectors."""
    embedder = RandomEmbedder(dimension=64, seed=42)
    vectors = embedder.encode(TEXTS, "document")
    assert abs(float(vectors[0] @ vectors[1])) < 0.5


def test_the_provenance_records_the_seed_and_the_rule() -> None:
    """The artifact carries the recipe, so a reader can reproduce any vector."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    provenance = embedder.provenance()
    assert provenance["mode"] == "random"
    assert provenance["seed"] == 42
    assert provenance["dimension"] == 8
    assert provenance["draw_rule"] == DRAW_RULE
    assert provenance["cached"] is False
    assert embedder.info.backend == "random"
    assert embedder.info.model_id == "random-42"
    assert embedder.info.extra["seed"] == 42


def test_the_model_id_is_never_a_real_model_id() -> None:
    """A random vector must never share a cache key with a real run."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    assert "random" in embedder.info.model_id
    assert embedder.info.revision_or_digest is None


def test_an_unknown_kind_is_rejected() -> None:
    """A typo in the kind would silently change the draw's key."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    with pytest.raises(ValueError, match="kind"):
        embedder.encode(TEXTS, "documents")  # type: ignore[arg-type]


def test_a_blank_text_is_rejected() -> None:
    """An empty text cannot be embedded reliably (the shared contract)."""
    embedder = RandomEmbedder(dimension=8, seed=42)
    with pytest.raises(ValueError, match="blank"):
        embedder.encode(["ok", "  "], "document")


def test_a_non_positive_dimension_is_rejected() -> None:
    """The dimension must be a real one."""
    with pytest.raises(ValueError, match="dimension"):
        RandomEmbedder(dimension=0, seed=42)


def test_the_embedder_satisfies_the_embedder_protocol() -> None:
    """The port is what the feature pack takes, so the floor plugs into it."""
    from cluster_rag.embed.base import EmbedderInfo

    embedder = RandomEmbedder(dimension=8, seed=42)
    assert isinstance(embedder.info, EmbedderInfo)
    assert callable(embedder.encode)
    for kind in KINDS:
        assert embedder.encode(["a text"], kind).shape == (1, 8)
