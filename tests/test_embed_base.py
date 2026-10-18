"""Tests for the Embedder protocol helpers (Phase 2): contract, prefixes, kinds."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.embed import (
    CONTENT_PLACEHOLDER,
    KINDS,
    EmbedderInfo,
    apply_prefix,
    empty_embeddings,
    ensure_embedding_contract,
    l2_normalize,
    resolve_prefix,
    validate_texts,
)
from fake_embedder import FakeEmbedder


def test_kinds_are_the_plan_defined_three() -> None:
    """Only the three encode kinds from PLAN.md Phase 2 exist."""
    assert KINDS == ("query", "document", "label")


def test_resolve_prefix_per_kind_and_default_empty() -> None:
    """Configured prefixes resolve per kind; unset kinds fall back to no prefix."""
    prefixes = {"query": "query: ", "document": "passage: "}
    assert resolve_prefix(prefixes, "query") == "query: "
    assert resolve_prefix(prefixes, "document") == "passage: "
    assert resolve_prefix(prefixes, "label") == ""
    assert resolve_prefix({}, "label") == ""


def test_resolve_prefix_rejects_unknown_kind() -> None:
    """A typo in the kind would silently drop instruction prefixes, so it raises."""
    with pytest.raises(ValueError, match="kind"):
        resolve_prefix({}, "documents")  # type: ignore[arg-type]


def test_apply_prefix_plain_and_template() -> None:
    """Plain prefixes are concatenated; {content} templates wrap the text."""
    assert apply_prefix("query: ", ["a", "b"]) == ["query: a", "query: b"]
    assert apply_prefix("", ["a"]) == ["a"]
    template = f"task: search result | query: {CONTENT_PLACEHOLDER}"
    assert apply_prefix(template, ["red planet", "x{y}"]) == [
        "task: search result | query: red planet",
        "task: search result | query: x{y}",
    ]


def test_l2_normalize_hand_computed() -> None:
    """Rows become unit vectors; dot product between equal rows is exactly ~1."""
    matrix = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    out = l2_normalize(matrix)
    assert out.dtype == np.float32
    assert np.allclose(out, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-6)
    assert np.allclose((out @ out.T).diagonal(), 1.0, atol=1e-6)


def test_l2_normalize_rejects_zero_rows_and_bad_shape() -> None:
    """Zero rows (empty texts) and non-2-D inputs are contract violations."""
    with pytest.raises(ValueError, match="all-zero"):
        l2_normalize(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        l2_normalize(np.zeros(4, dtype=np.float32))


def test_ensure_embedding_contract_casts_and_normalizes() -> None:
    """float64 input is cast to float32 and re-normalized; shape validated."""
    out = ensure_embedding_contract([[2.0, 0.0], [0.0, 10.0]], dimension=2)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_ensure_embedding_contract_rejects_bad_inputs() -> None:
    """NaN/inf, wrong dimension and 0-column matrices fail with clear errors."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        ensure_embedding_contract(np.ones((2, 3)), dimension=2)
    with pytest.raises(ValueError, match="non-finite"):
        ensure_embedding_contract([[float("nan"), 1.0]])
    with pytest.raises(ValueError, match="at least one dimension"):
        ensure_embedding_contract(np.ones((2, 0)))


def test_validate_texts_and_empty_embeddings() -> None:
    """Blank texts are rejected; empty batches get a valid (0, d) matrix."""
    assert validate_texts(["a", " b "]) == ["a", " b "]
    with pytest.raises(ValueError, match="position 1"):
        validate_texts(["a", "   "])
    with pytest.raises(ValueError, match="position 0"):
        validate_texts([""])
    out = empty_embeddings(7)
    assert out.shape == (0, 7) and out.dtype == np.float32


def test_build_embedder_unknown_backend_raises() -> None:
    """An unknown backend in the config fails loudly (typos in configs/)."""
    from cluster_rag.config import EmbedderConfig
    from cluster_rag.embed import build_embedder

    with pytest.raises(ValueError, match="Unknown embed backend"):
        build_embedder(EmbedderConfig(backend="llamafile", model_id="m"))
    with pytest.raises(ValueError, match="expects an EmbedderConfig"):
        build_embedder("not a config")  # type: ignore[arg-value]


def test_info_default_prefixes_are_empty() -> None:
    """EmbedderInfo defaults keep meta.json stable for prefixless models."""
    info = EmbedderInfo(model_id="m", backend="fake", dimension=4)
    assert info.prefixes == {} and info.context_tokens is None


def test_fake_embedder_is_deterministic_and_contract_abiding() -> None:
    """Same (kind, text) -> same unit vector; different text -> different vector."""
    embedder = FakeEmbedder(dimension=6, prefixes={"query": "query: "})
    first = embedder.encode(["alpha", "beta"], "query")
    second = embedder.encode(["alpha", "beta"], "query")
    assert first.dtype == np.float32 and first.shape == (2, 6)
    assert np.allclose(first, second, atol=1e-7)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    other = embedder.encode(["alpha different"], "query")
    assert abs(float(first[0] @ other[0])) < 0.999  # hashed vectors differ
    # calls are recorded so cache tests can prove hits skip re-encoding
    assert embedder.call_count == 3
    assert embedder.encode_calls[0] == ("query", ("alpha", "beta"))


def test_fake_embedder_distinguishes_kinds() -> None:
    """The kind is part of the vector seed (query != document for same text)."""
    embedder = FakeEmbedder(dimension=6)
    query_vec = embedder.encode(["same text"], "query")[0]
    document_vec = embedder.encode(["same text"], "document")[0]
    assert abs(float(query_vec @ document_vec)) < 0.999
