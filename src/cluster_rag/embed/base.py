"""The :class:`Embedder` protocol and the shared embedding contract (Phase 2).

Every backend (HF, Ollama, fakes in tests) implements:

    encode(texts, kind: "query" | "document" | "label") -> np.ndarray

returning a ``np.float32`` matrix of shape ``(n, d)`` whose rows are
L2-normalized, so the dot product of two rows equals their cosine similarity
(AGENTS.md "Embeddings are np.float32, shape (n, d), L2-normalized"; enforced
by :func:`ensure_embedding_contract`).

Model-specific prefixes (e5 ``query: "/"passage: "``, nomic
``classification:"/ search_document:", EmbeddingGemma task prompts, ...) are
never hard-coded here: they come from the config (``embed.prefixes.<kind>``)
and are applied by the backend via :func:`apply_prefix`. Templates containing
``{content}`` are expanded per text (documented prompts like
``task: search result | query: {content}``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# EncodeKind/KINDS are owned by the config schema (cluster_rag.config) — see
# there; this protocol module re-exports them so backends, caches and tests
# keep importing the embed API surface from here.
from cluster_rag.config import KINDS, EncodeKind  # noqa: F401  (re-export)

CONTENT_PLACEHOLDER = "{content}"


@dataclass(frozen=True)
class EmbedderInfo:
    """Immutable provenance and interface facts about an embedder.

    ``revision_or_digest`` is the HF revision pin or the primary Ollama model
    digest; backend-specific extras (server info, parameter counts, resolved
    revisions, ...) go into ``extra`` as a JSON-serializable dict and are
    written into each run's ``meta.json``.
    """

    model_id: str
    backend: str  # "hf" | "ollama" | "fake" (tests)
    dimension: int
    revision_or_digest: str | None = None
    prefixes: Mapping[str, str] = field(default_factory=dict)  # kind -> configured prefix/template
    context_tokens: int | None = None  # model max input context in tokens, if known
    extra: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    """Protocol every embedding backend implements (typing only, no runtime check).

    ``encode`` must be deterministic for a fixed backend, model, ``kind`` and
    text (identical calls give identical vectors), because the disk cache keys
    on ``(model_id, prefix, preprocessing_version, sha1(text))`` and assumes
    reproducibility (AGENTS.md Phase 2 cache rule).
    """

    info: EmbedderInfo

    def encode(self, texts: Sequence[str], kind: EncodeKind) -> np.ndarray:
        """Embed texts; return a ``(n, d)`` float32 matrix with unit rows.

        Raises:
            ValueError: If ``kind`` is unknown or a text is blank/empty.
        """
        ...


def resolve_prefix(prefixes: Mapping[str, str], kind: EncodeKind) -> str:
    """Return the configured prefix (or template) for ``kind``; "" if unset.

    Unknown ``kind`` values raise, since a typo would silently drop an
    instruction prefix and change the embedding space (PLAN.md section 6).
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown encode kind {kind!r}; expected one of {KINDS}")
    return prefixes.get(kind, "")


def apply_prefix(prefix: str, texts: Sequence[str]) -> list[str]:
    """Prepend ``prefix`` to every text, or expand it as a ``{content}`` template.

    Models whose documented prompts wrap the text (EmbeddingGemma:
    ``task: search result | query: {content}``) are configured with a template;
    plain prefixes are concatenated in front (e5 ``query: ``, nomic
    ``classification: ``). An empty prefix returns the texts unchanged.
    ``.replace`` (not ``str.format``) is used so braces inside real texts or
    unrelated placeholders in the template never raise.
    """
    if CONTENT_PLACEHOLDER in prefix:
        return [prefix.replace(CONTENT_PLACEHOLDER, text) for text in texts]
    return [f"{prefix}{text}" for text in texts] if prefix else list(texts)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D array; float32 output (dot product = cosine).

    Raises:
        ValueError: If the array is not 2-D or contains a zero row (a zero
            vector cannot be normalized: it typically means an empty text
            reached the encoder, which validation should have prevented).
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D (n, d); got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Cannot normalize embeddings: a row is all-zero or non-finite")
    return (matrix / norms).astype(np.float32)


def ensure_embedding_contract(matrix: np.ndarray, *, dimension: int | None = None) -> np.ndarray:
    """Validate and enforce the embedding contract; returns re-normalized float32.

    Backends call this on their raw output and the cache calls it on assembled
    results, so the contract holds at every index boundary regardless of what
    the backend returned.

    Raises:
        ValueError: Shape/dtype/normalizability violations, or a column
            count that disagrees with ``dimension``.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D (n, d); got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        raise ValueError("Embeddings must have at least one dimension (d >= 1)")
    if dimension is not None and matrix.shape[1] != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: got {matrix.shape[1]}, expected {dimension}"
        )
    if matrix.size and not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values (NaN/inf)")
    return l2_normalize(matrix)


def validate_texts(texts: Sequence[str]) -> list[str]:
    """Require non-blank strings; backends call this before prefixing."""
    checked = list(texts)
    for index, text in enumerate(checked):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"text at position {index} is empty or blank; an empty text cannot be "
                "embedded reliably (check preprocessing: a strip variant may have removed it all)"
            )
    return checked


def empty_embeddings(dimension: int) -> np.ndarray:
    """A valid ``(0, dimension)`` float32 matrix for empty encode batches."""
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    return np.zeros((0, dimension), dtype=np.float32)
