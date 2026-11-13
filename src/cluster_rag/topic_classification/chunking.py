"""Word-chunk splitting and mean pooling for the EURLEX-4K document view.

The baseline embeds the whole document text once, but MiniLM-L6-v2's context is
256 tokens, so the median 447-word document is truncated.  The TC-7 treatment
(item 1 of ``TOPIC_CLASSIFICATION_PLAN.md`` section 6) replaces that single
vector with the **mean of its non-overlapping word chunks**: split the same
title + header + recitals + main-body text into consecutive ~200-word windows,
embed every window, average the chunk vectors and L2-normalize the mean.

Both helpers are pure (numpy + the standard library): the model client stays at
the edge, wired by the experiment script (AGENTS.md dependency rule).
"""

from __future__ import annotations

import numpy as np

DEFAULT_CHUNK_WORDS = 200


def word_chunks(text: str, *, chunk_words: int = DEFAULT_CHUNK_WORDS) -> list[str]:
    """Split ``text`` into non-overlapping consecutive windows of ``chunk_words``.

    Whitespace is the word separator (``str.split``), so every original word
    lands in exactly one chunk; the last chunk carries the remainder and may be
    shorter.  A text of at most ``chunk_words`` words is a single chunk.

    Args:
        text: The document text.
        chunk_words: Window size in words; must be positive.

    Returns:
        The chunk strings, in document order.  An empty/blank text yields an
        empty list (there is nothing to embed).

    Raises:
        ValueError: If ``chunk_words`` is not a positive integer.
    """
    if not isinstance(chunk_words, int) or isinstance(chunk_words, bool) or chunk_words < 1:
        raise ValueError(f"chunk_words must be a positive integer, got {chunk_words!r}")
    words = text.split()
    return [
        " ".join(words[start : start + chunk_words]) for start in range(0, len(words), chunk_words)
    ]


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    """Mean-pool chunk vectors into one unit-norm document vector.

    Args:
        vectors: ``(n_chunks, d)`` float array of L2-normalized chunk vectors.

    Returns:
        The ``(d,)`` float32 mean of the chunk vectors, L2-normalized so the
        dot product with another unit vector is the cosine similarity.

    Raises:
        ValueError: If ``vectors`` is not a non-empty 2-D array.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"mean_pool expects a non-empty (n_chunks, d) matrix; got {matrix.shape}")
    pooled = matrix.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        raise ValueError("mean_pool: the chunk vectors average to the zero vector")
    return (pooled / norm).astype(np.float32)
