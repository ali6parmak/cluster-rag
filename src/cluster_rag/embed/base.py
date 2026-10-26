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

**The optional token-level capability** (Stage 2.7, the late chunking): a
backend that can return the *token-level* embeddings of a text implements the
:class:`TokenEmbedder` protocol on top of :meth:`Embedder.encode`. The
capability is probed, never assumed (:func:`token_capability`): Ollama's
``/api/embed`` returns one vector per text and nothing else, so the Ollama
backend has no token-level capability and the late-chunking arm says so
instead of failing deep inside a run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

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


@dataclass(frozen=True, eq=False)
class TokenEmbedding:
    """One text's **token-level** embeddings and their character spans.

    The optional capability of Stage 2.7 (the late chunking): the arm embeds
    the whole document once and pools its token-level vectors per chunk, so
    every chunk carries the document's context.

    Attributes:
        vectors: One row per token, ``(n_tokens, d)`` float32, in the model's
            own embedding space (the projection the model's own sentence
            embedding would get).
        spans: The character span of every token **in the embedded text** (the
            prompt prefix is already removed), in token order. A token with an
            empty span (a special token, or one of the prefix's own tokens)
            carries no text and joins no chunk.
        prefix_tokens: How many of the leading tokens belong to the prompt
            prefix. They carry no text (no chunk can own them), but the
            model's own embedding averages them, so a caller that wants to
            reproduce the model's own vector needs to know them.
    """

    vectors: np.ndarray
    spans: tuple[tuple[int, int], ...]
    prefix_tokens: int = 0

    def __post_init__(self) -> None:
        """Check that the vectors and the spans describe the same tokens.

        Raises:
            ValueError: If the shapes disagree or a vector is not finite.
        """
        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2:
            raise ValueError(f"Token vectors must be 2-D (n_tokens, d); got {vectors.shape}")
        if len(self.spans) != vectors.shape[0]:
            raise ValueError(
                f"Token vectors and spans disagree: {vectors.shape[0]} vectors, "
                f"{len(self.spans)} spans"
            )
        if not 0 <= self.prefix_tokens <= vectors.shape[0]:
            raise ValueError(
                f"prefix_tokens must be between 0 and {vectors.shape[0]}, got {self.prefix_tokens}"
            )
        if vectors.size and not np.isfinite(vectors).all():
            raise ValueError("Token vectors contain non-finite values (NaN/inf)")


@runtime_checkable
class TokenEmbedder(Protocol):
    """The optional **token-level** capability of an embedder (Stage 2.7).

    Not every backend can return token-level embeddings: Ollama's
    ``/api/embed`` returns one vector per text and nothing else, so the
    Ollama backend does not implement this protocol. A caller that needs the
    capability asks for it (:func:`token_capability`) instead of assuming it.
    """

    def encode_tokens(self, texts: Sequence[str], kind: EncodeKind) -> list[TokenEmbedding]:
        """Embed texts and return their token-level vectors and spans."""


def token_capability(embedder: object) -> TokenEmbedder | None:
    """The embedder's token-level capability, or ``None`` when it has none.

    The probe is honest through the disk cache too: a
    :class:`~cluster_rag.embed.cache.CachedEmbedder` reports the capability of
    the backend it wraps, so a cached Ollama backend is correctly reported as
    having none.

    Args:
        embedder: The embedder to probe (any object).

    Returns:
        The embedder as a :class:`TokenEmbedder`, or ``None`` when the backend
        cannot return token-level embeddings.
    """
    supports = getattr(embedder, "supports_tokens", None)
    if supports is None:
        supports = hasattr(embedder, "encode_tokens")
    if not supports:
        return None
    return cast(TokenEmbedder, embedder)


def prefix_shift(prefix: str) -> int:
    """The number of characters a prefix adds **in front of** the text.

    A plain prefix is prepended (``len(prefix)``); a ``{content}`` template
    puts the text after the template's head (``len(head)``). The token-level
    path uses it to map the tokenizer's character offsets — which are offsets
    into the *prefixed* text — back onto the text itself.
    """
    if CONTENT_PLACEHOLDER in prefix:
        head, _, _ = prefix.partition(CONTENT_PLACEHOLDER)
        return len(head)
    return len(prefix)


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
