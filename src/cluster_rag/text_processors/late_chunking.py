"""The late chunking's pooling (Stage 2.7): the pure core of the arm.

Late chunking embeds the **whole document once** and pools its
**token-level** embeddings per chunk, so every chunk carries the document's
context. This module is that pooling: a pure function over the token vectors
and the chunks' character spans — no model client, no I/O (AGENTS.md
dependency direction: the track's package imports ``cluster_rag.embed.base``
and nothing heavier).

**The pooling rule.** A model with mean pooling embeds a text as
``Normalize(mean(tokens))``; the late-chunked chunk vector is the same
formula restricted to the chunk's tokens. Two rules are measured (the
reviewer's rule: measure the alternatives rather than pick one):

- ``"mean"`` — the mean of the chunk's token vectors, then L2-normalized:
  the model's own formula;
- ``"norm"`` — every token vector is L2-normalized *before* the mean, so
  each token weighs the same whatever its magnitude; the chunk vector is then
  L2-normalized again.

**Which tokens belong to a chunk.** A token joins a chunk when the two
character spans overlap (half-open) and the token carries text: a token with
an empty span is a special token or one of the prompt prefix's own tokens,
and belongs to no chunk. The overlapping windows of the track's word chunker
share the tokens of their overlap, exactly as the early-chunked arm shares
the text of its overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cluster_rag.embed.base import TokenEmbedding, l2_normalize

# The two pooling rules (both measured).
LATE_POOLING_MEAN = "mean"
LATE_POOLING_NORM = "norm"
LATE_POOLINGS: tuple[str, ...] = (LATE_POOLING_MEAN, LATE_POOLING_NORM)


def token_chunk_indices(
    token_spans: Sequence[tuple[int, int]],
    chunk_spans: Sequence[tuple[int, int]],
) -> list[list[int]]:
    """The token indices of every chunk (the tokens whose span overlaps it).

    Args:
        token_spans: Every token's character span, in token order.
        chunk_spans: Every chunk's character span, in chunk order.

    Returns:
        One list of token indices per chunk (the tokens that overlap it, in
        token order). A token with an empty span (a special token, or one of
        the prompt prefix's own tokens) joins no chunk.
    """
    indices: list[list[int]] = []
    for chunk_start, chunk_end in chunk_spans:
        indices.append(
            [
                index
                for index, (start, end) in enumerate(token_spans)
                if end > start and start < chunk_end and end > chunk_start
            ]
        )
    return indices


def pool_chunk_vectors(
    tokens: TokenEmbedding,
    chunk_spans: Sequence[tuple[int, int]],
    *,
    pooling: str = LATE_POOLING_MEAN,
) -> np.ndarray:
    """One vector per chunk: the pooled token vectors of the chunk's tokens.

    Args:
        tokens: The text's token-level embeddings and their character spans.
        chunk_spans: Every chunk's character span, in chunk order.
        pooling: The pooling rule (``"mean"`` or ``"norm"``).

    Returns:
        A ``(n_chunks, d)`` float32 matrix with unit rows (the embedding
        contract: the dot product of two rows is their cosine similarity).

    Raises:
        ValueError: If the pooling rule is unknown, or a chunk has no token
            (the text was truncated, or the chunker's span does not match the
            text).
    """
    if pooling not in LATE_POOLINGS:
        raise ValueError(
            f"Unknown late pooling rule {pooling!r}; expected one of {list(LATE_POOLINGS)}"
        )
    vectors = np.asarray(tokens.vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Token vectors must be 2-D (n_tokens, d); got {vectors.shape}")
    indices = token_chunk_indices(tokens.spans, chunk_spans)
    pooled = np.zeros((len(chunk_spans), vectors.shape[1]), dtype=np.float32)
    for row, token_indices in enumerate(indices):
        if not token_indices:
            raise ValueError(
                f"the chunk at characters {chunk_spans[row]} has no token: the text "
                "was truncated, or the chunker's span does not match the text"
            )
        selected = vectors[token_indices]
        if pooling == LATE_POOLING_NORM:
            norms = np.linalg.norm(selected, axis=1, keepdims=True)
            selected = selected / np.maximum(norms, 1e-12)
        pooled[row] = selected.mean(axis=0)
    return l2_normalize(pooled)


def pool_document_vector(tokens: TokenEmbedding, *, prefix_tokens: int = 0) -> np.ndarray:
    """The document's **own** pooled vector (the model's formula over its tokens).

    The mean of the text's content tokens (plus the prompt prefix's own
    tokens, when ``prefix_tokens`` says how many there are), L2-normalized:
    what the model itself would embed for this text, computed from the
    token-level vectors. The late-chunking arm uses it as the consistency
    check of the token-level path (the cosine with the model's own embedding).

    Args:
        tokens: The text's token-level embeddings and their character spans.
        prefix_tokens: How many leading tokens belong to the prompt prefix
            (the model's own pooling averages them too; no chunk owns them).

    Raises:
        ValueError: If the text has no token to pool.
    """
    vectors = np.asarray(tokens.vectors, dtype=np.float32)
    content = [index for index, (start, end) in enumerate(tokens.spans) if end > start]
    keep = sorted(set(content) | set(range(prefix_tokens)))
    if not keep:
        raise ValueError("the text has no token to pool")
    pooled = vectors[keep].mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("the document's pooled vector is zero or non-finite")
    return (pooled / norm).astype(np.float32)


def pooling_report(
    tokens: TokenEmbedding,
    chunk_spans: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """The provenance of one text's pooling (the artifact's diagnostics).

    Returns:
        The number of content tokens, the number of tokens that landed in at
        least one chunk (the **coverage**), the tokens-per-chunk range and the
        number of chunks with no token.
    """
    indices = token_chunk_indices(tokens.spans, chunk_spans)
    content = sum(1 for start, end in tokens.spans if end > start)
    covered = {index for chunk in indices for index in chunk}
    per_chunk = [len(chunk) for chunk in indices]
    return {
        "n_content_tokens": content,
        "n_covered_tokens": len(covered),
        "token_coverage": (len(covered) / content) if content else 0.0,
        "min_tokens_per_chunk": min(per_chunk) if per_chunk else 0,
        "max_tokens_per_chunk": max(per_chunk) if per_chunk else 0,
        "n_chunks_without_token": sum(1 for count in per_chunk if count == 0),
    }
