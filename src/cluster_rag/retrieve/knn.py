"""kNN voting scores (Phase 4): similarity-weighted label votes of nearest examples.

Given the train example matrix (document-kind) and a query matrix (query-kind,
both unit-norm so dot products are cosines), each query's score for a label is
the sum of similarity-weighted votes of its top-k nearest train examples:
``score(q, label) = sum_j in topk(q) max(cos(q, j), 0) ** power * has_label(j, label)``.

Conventions (deterministic, documented for the harness):
- ties in similarity break to the *earlier* train row (stable descending sort);
- similarities are clipped at 0 before the power, so a negative-similarity
  neighbor contributes nothing (and fractional ``power`` cannot produce NaNs);
- a label with zero train examples among the neighbors simply scores 0 —
  zero-shot labels are never suggested by this predictor.
"""

from __future__ import annotations

import numpy as np


def knn_vote_scores(
    train_vectors: np.ndarray,
    train_label_matrix: np.ndarray,
    query_vectors: np.ndarray,
    *,
    k: int,
    power: float = 1.0,
) -> np.ndarray:
    """Score every query against every label via weighted top-k neighbor votes.

    Args:
        train_vectors: ``(n_train, d)`` float32 unit-norm document embeddings of
            the labeled train examples (row order matches ``train_label_matrix``).
        train_label_matrix: ``(n_train, n_labels)`` multi-hot indicator of each
            example's labels (bool; the harness's sorted-column convention).
        query_vectors: ``(n_query, d)`` float32 unit-norm query embeddings.
        k: Number of nearest neighbors per query; if larger than ``n_train``,
            all examples vote (clamped, never an error).
        power: Vote-weight exponent (``sim ** power`` after clipping at 0);
            1.0 = plain similarity-weighted votes, 2.0 = sharper top-neighbor
            dominance.

    Returns:
        The ``(n_query, n_labels)`` float32 score matrix, higher = more likely.

    Raises:
        ValueError: On shape/dtype mismatches, a non-positive ``k`` or a
            non-positive/non-finite ``power``.
    """
    train_vectors = np.asarray(train_vectors)
    query_vectors = np.asarray(query_vectors)
    train_label_matrix = np.asarray(train_label_matrix)
    if train_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError(
            "knn_vote_scores expects 2-D (n, d) matrices; got "
            f"train {train_vectors.shape}, query {query_vectors.shape}"
        )
    if train_label_matrix.ndim != 2 or train_label_matrix.shape[0] != train_vectors.shape[0]:
        raise ValueError(
            "train_label_matrix must be (n_train, n_labels) with n_train matching "
            f"train_vectors rows; got {train_label_matrix.shape} vs {train_vectors.shape}"
        )
    if train_vectors.dtype != np.float32 or query_vectors.dtype != np.float32:
        raise ValueError("knn_vote_scores expects float32 embedding matrices")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"number of neighbors k must be a positive integer, got {k!r}")
    power_float = float(power)
    if not np.isfinite(power_float) or power_float <= 0:
        raise ValueError(f"vote power must be a positive finite number, got {power!r}")

    n_train = train_vectors.shape[0]
    k_eff = min(k, n_train)
    # Both sides unit-norm -> dot product = cosine. (660 x 2100 at the current
    # dataset size: one matmul, then one stable argsort — no ANN index needed.)
    similarities = query_vectors @ train_vectors.T
    order = np.argsort(-similarities, axis=1, kind="stable")[:, :k_eff]
    weights = np.clip(np.take_along_axis(similarities, order, axis=1), 0.0, None) ** power_float
    neighbor_labels = train_label_matrix[order]  # (n_query, k_eff, n_labels)
    scores = np.einsum("qk,qkl->ql", weights, neighbor_labels)
    return scores.astype(np.float32)
