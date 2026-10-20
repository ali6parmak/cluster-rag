"""Tests for the HF (sentence-transformers) backend — all integration-marked.

These need the real model download/load and touch torch; they are skipped by
default (``addopts = -m 'not integration'``). Run them explicitly with
``uv run pytest -m integration``.
"""

from __future__ import annotations

import dataclasses
from importlib.util import find_spec

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


# --- Phase 4 pool additions (card-verified wiring, real-model sanity) -------
# One plan row per new HF model: expected dimension/context plus the encode
# kind whose prefix the "exactly-once" check probes. The check embeds a text
# with the config prefix against the same text manually prefixed and embedded
# prefix-free: near-identical cosine means the config-owned prefix was applied
# exactly once and nothing (server template, ST autoload) stacked a second
# copy (the Phase 4 opening reason for per-model sanity checks).


@dataclasses.dataclass(frozen=True)
class _PoolPlan:
    """One Phase 4 pool model's wiring expectations (integration tests)."""

    config: EmbedderConfig
    dimension: int
    context: int | None  # asserted only when the card pins the number
    check_kind: str | None  # None = no prefixes (nothing to exactly-once check)
    needs_peft: bool = False  # jina-v5: remote code imports peft at load time


NEW_HF_MODELS: dict[str, _PoolPlan] = {
    "e5-large-v2": _PoolPlan(
        config=EmbedderConfig(
            backend="hf",
            model_id="intfloat/e5-large-v2",
            revision="f169b11e22de13617baa190a028a32f3493550b6",
            batch_size=4,
            prefixes=PrefixesConfig(query="query: ", document="passage: ", label="query: "),
        ),
        dimension=1024,  # card: 24 layers, 1024-d
        context=512,
        check_kind="query",
    ),
    "mpnet": _PoolPlan(
        config=EmbedderConfig(
            backend="hf",
            model_id="sentence-transformers/all-mpnet-base-v2",
            revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            batch_size=4,
        ),
        dimension=768,
        context=384,  # the ST max_seq_length (sentence_bert_config.json)
        check_kind=None,
    ),
    "jina-v5": _PoolPlan(
        config=EmbedderConfig(
            backend="hf",
            model_id="jinaai/jina-embeddings-v5-text-small",
            revision="dd76d535f5447ca3897a9c893fb1e612ead98192",
            batch_size=4,
            trust_remote_code=True,
            model_kwargs={"default_task": "retrieval"},
            dimension=1024,  # custom ST module does not report its output dimension
            context_tokens=None,
            prefixes=PrefixesConfig(query="Query: ", document="Document: ", label="Query: "),
        ),
        dimension=1024,
        context=32768,
        check_kind="document",
        needs_peft=True,  # remote code does `from peft import PeftMixedModel, PeftConfig`
    ),
    "voyage-4-nano": _PoolPlan(
        config=EmbedderConfig(
            backend="hf",
            model_id="voyageai/voyage-4-nano",
            revision="433ece5e27a1611e9360dd7134ecaaf79fb3681d",
            batch_size=4,
            trust_remote_code=True,
            prefixes=PrefixesConfig(
                query="Represent the query for retrieving supporting documents: ",
                document="Represent the document for retrieval: ",
            ),
        ),
        dimension=2048,
        context=32768,  # ST loads Qwen3's 32768; the card's 32000 is the API-side value
        check_kind="document",
    ),
    "granite-english-r2": _PoolPlan(
        config=EmbedderConfig(
            backend="hf",
            model_id="ibm-granite/granite-embedding-english-r2",
            revision="47ea694b257b703fee9253d75c2b1f2985180498",
            batch_size=4,
        ),
        dimension=768,
        context=8192,
        check_kind=None,
    ),
    "f2llm-80m": _PoolPlan(
        config=EmbedderConfig(
            backend="hf",
            model_id="codefuse-ai/F2LLM-v2-80M",
            revision="ad88d7a126711f1490cd4bad645dc9d3acc2af6a",
            batch_size=4,
            prefixes=PrefixesConfig(
                query=(
                    "Instruct: Given a question, retrieve passages that can help answer "
                    "the question.\nQuery: "
                ),
            ),
        ),
        dimension=320,
        context=None,
        check_kind="query",
    ),
}

RAW_TEXT = "Ensure that all children have access to free education."


@pytest.mark.integration
@pytest.mark.parametrize("name", NEW_HF_MODELS.keys())
def test_hf_new_pool_model_wiring(name: str) -> None:
    """Per-model Phase 4 wiring: contract shape, context, exactly-once prefix."""
    plan = NEW_HF_MODELS[name]
    if plan.needs_peft and find_spec("peft") is None:
        pytest.skip(
            f"{name}: remote code imports peft — dependency decision pending (PLAN Phase 4 opening)"
        )
    embedder = HFEmbedder(plan.config)

    vectors = embedder.encode([RAW_TEXT], plan.check_kind or "query")
    assert vectors.shape == (1, plan.dimension)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    if plan.context is not None:
        assert embedder.info.context_tokens == plan.context
    if plan.check_kind is None:
        return
    prefix = (
        plan.config.prefixes.query if plan.check_kind == "query" else plan.config.prefixes.document
    )
    bare_embedder = HFEmbedder(plan.config.model_copy(update={"prefixes": PrefixesConfig()}))
    auto = vectors[0]
    manual = bare_embedder.encode([f"{prefix}{RAW_TEXT}"], plan.check_kind)[0]
    cosine = float(manual @ auto)
    assert cosine > 0.999999, (
        f"{name}: {plan.check_kind} prefix not applied exactly once (cos {cosine})"
    )
    bare = bare_embedder.encode([RAW_TEXT], plan.check_kind)[0]
    assert abs(float(bare @ auto)) < 0.999999  # the prefix is not a no-op
