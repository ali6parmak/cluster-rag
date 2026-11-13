"""k-NN vote scoring from a precomputed similarity matrix (EURLEX-4K track).

The embedding k-NN in ``cluster_rag.retrieve.knn`` takes dense train/query
vector matrices.  A lexical k-NN cannot: a 45,000 x 200,000 TF-IDF matrix is
too large to densify, so the similarities are computed batch-by-batch from
sparse vectors and only the ``(n_query, n_train)`` similarity block is dense.
This module applies the **same** voting rule to that block, so the lexical arm
and the embedding arm differ only in how similarity is produced:

``score(q, label) = sum_j in topk(q) max(sim(q, j), 0) ** power * has_label(j, label)``

with the same conventions as the main pipeline: stable descending sort (ties
break to the earlier train row), similarities clipped at 0, fractional powers
safe.  Pure numpy.
"""

from __future__ import annotations

import numpy as np


def vote_scores_from_similarity(
    similarity: np.ndarray,
    train_label_matrix: np.ndarray,
    *,
    k: int,
    power: float = 1.0,
) -> np.ndarray:
    """Score labels from a dense ``(n_query, n_train)`` similarity block.

    Args:
        similarity: One block of query-to-train similarities (cosine when both
            sides are unit-norm).
        train_label_matrix: ``(n_train, n_labels)`` multi-hot label indicator.
        k: Number of neighbours per query; clamped to ``n_train``.
        power: Vote-weight exponent (``sim ** power`` after clipping at 0).

    Returns:
        The ``(n_query, n_labels)`` float32 score matrix.

    Raises:
        ValueError: On shape mismatches, a non-positive ``k``, or a
            non-positive/non-finite ``power``.
    """
    similarity = np.asarray(similarity)
    train_label_matrix = np.asarray(train_label_matrix)
    if similarity.ndim != 2 or train_label_matrix.ndim != 2:
        raise ValueError(
            "vote_scores_from_similarity expects 2-D matrices; got "
            f"{similarity.shape} and {train_label_matrix.shape}"
        )
    if train_label_matrix.shape[0] != similarity.shape[1]:
        raise ValueError(
            "train_label_matrix must have one row per similarity column; got "
            f"{train_label_matrix.shape[0]} vs {similarity.shape[1]}"
        )
    if not np.isfinite(similarity).all():
        raise ValueError("similarity contains non-finite values (NaN/inf)")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"number of neighbors k must be a positive integer, got {k!r}")
    power_float = float(power)
    if not np.isfinite(power_float) or power_float <= 0:
        raise ValueError(f"vote power must be a positive finite number, got {power!r}")

    k_eff = min(k, similarity.shape[1])
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :k_eff]
    weights = np.clip(np.take_along_axis(similarity, order, axis=1), 0.0, None) ** power_float
    neighbor_labels = train_label_matrix[order]  # (n_query, k_eff, n_labels)
    scores = np.einsum("qk,qkl->ql", weights, neighbor_labels)
    return scores.astype(np.float32)
