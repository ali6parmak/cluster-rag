"""Tests for the HF (sentence-transformers) backend — all integration-marked.

These need the real model download/load and touch torch; they are skipped by
default (``addopts = -m 'not integration'``). Run them explicitly with
``uv run pytest -m integration``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import EmbedderConfig, PrefixesConfig
from cluster_rag.embed.hf import HFEmbedder

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # current HF cache snapshot


@pytest.mark.integration
def test_hf_encode_contract_and_prefixes() -> None:
    """Real model, GPU/CPU auto: float32, unit rows, prefixes applied per kind."""
    config = EmbedderConfig(
        backend="hf",
        model_id=MINILM,
        revision=MINILM_REVISION,
        batch_size=8,
        prefixes=PrefixesConfig(query="query: ", document="passage: "),
    )
    embedder = HFEmbedder(config)
    query_vecs = embedder.encode(["forced displacement rose sharply", "x"], "query")
    document_vecs = embedder.encode(["forced displacement rose sharply", "x"], "document")

    assert query_vecs.dtype == np.float32 and query_vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(query_vecs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(document_vecs, axis=1), 1.0, atol=1e-5)
    # the same text under different kinds embeds differently on this backend
    assert abs(float(query_vecs[1] @ document_vecs[1])) < 0.999999

    info = embedder.info
    assert info.dimension == 384 and info.context_tokens == 256  # MiniLM-L6-v2 is a 256-token model
    assert info.revision_or_digest == MINILM_REVISION


@pytest.mark.integration
def test_hf_token_counts_and_empty_batch() -> None:
    """Exact tokenizer counts for truncation diagnostics; empty batch contract."""
    config = EmbedderConfig(backend="hf", model_id=MINILM, revision=MINILM_REVISION)
    embedder = HFEmbedder(config)
    counts = embedder.token_counts(["one two three", "x" * 400])
    assert counts.tolist() == [5, 3]  # +2 special tokens; 400 x's collapse to one wordpiece
    empty = embedder.encode([], "query")
    assert empty.shape == (0, 384) and empty.dtype == np.float32


@pytest.mark.integration
def test_hf_e5_prefixes_change_vectors() -> None:
    """The e5 "query: " prefix measurably changes the vector (guards against
    silently dropping instruction prefixes)."""
    bare = HFEmbedder(
        EmbedderConfig(
            backend="hf",
            model_id="intfloat/e5-base-v2",
            revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        )
    )
    prefixed = HFEmbedder(
        EmbedderConfig(
            backend="hf",
            model_id="intfloat/e5-base-v2",
            revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
            prefixes=PrefixesConfig(query="query: ", document="passage: "),
        )
    )
    text = ["The State party should ratify the treaty"]
    a = bare.encode(text, "query")[0]
    b = prefixed.encode(text, "query")[0]
    assert abs(float(a @ b)) < 0.99999  # prefix changed the vector
    assert prefixed.info.dimension == 768 and prefixed.info.context_tokens == 512
