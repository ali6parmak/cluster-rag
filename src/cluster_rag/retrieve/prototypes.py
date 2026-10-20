"""Label-prototype scores (Phase 4): cosine to the mean of a label's train examples.

A label's prototype is the L2-normalized mean of its train examples' document
embeddings (the PLAN Phase 4 checkbox; retrieval-agnostic, one vector per
label). A query's score for a label is the cosine between its query embedding
and the prototype — unit rows on both sides make the dot product the cosine.

This module carries the *mean* aggregation; the per-label max-similarity
variant (max cosine over the label's single train examples, the tail-label
complement) joins in the Phase 4 pool-cut sweep as its own aggregation over
the same sentence matrix. Label names/description embeddings (zero-shot) are a
separate mechanism and are not built here.
"""

from __future__ import annotations

import numpy as np


def label_prototypes(train_vectors: np.ndarray, train_label_matrix: np.ndarray) -> np.ndarray:
    """One prototype per label: the renormalized mean of its example embeddings.

    Args:
        train_vectors: ``(n_train, d)`` float32 example embeddings (any matrix
            the caller validated as unit-norm rows).
        train_label_matrix: ``(n_train, n_labels)`` multi-hot indicator; column
            order is the harness's sorted-label convention.

    Returns:
        ``(n_labels, d)`` float32 prototypes. A label with zero train examples
        gets an all-zero prototype (its cosine to anything is 0, exactly like
        the trivial baselines' unseen-label behavior — zero-shot discovery is
        the zero-shot predictor's job, not the prototype mean's).

    Raises:
        ValueError: On shape mismatches or a non-float32 matrix.
    """
    train_vectors = np.asarray(train_vectors)
    train_label_matrix = np.asarray(train_label_matrix)
    if train_vectors.ndim != 2:
        raise ValueError(f"train_vectors must be 2-D (n, d); got {train_vectors.shape}")
    if train_label_matrix.ndim != 2 or train_label_matrix.shape[0] != train_vectors.shape[0]:
        raise ValueError(
            "train_label_matrix must be (n_train, n_labels) with n_train matching "
            f"train_vectors rows; got {train_label_matrix.shape} vs {train_vectors.shape}"
        )
    if train_vectors.dtype != np.float32:
        raise ValueError("label_prototypes expects a float32 embedding matrix")

    # Mean in float64 for accumulation precision, cast back to float32 output.
    indicator = train_label_matrix.astype(np.float64)
    counts = indicator.sum(axis=0)  # (n_labels,)
    safe_counts = np.where(counts > 0, counts, 1.0)  # zero-count labels: sums are 0 too
    means = (train_vectors.astype(np.float64).T @ indicator / safe_counts).T  # (n_labels, d)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return (means / safe).astype(np.float32)  # nonzero rows unit-norm, empty rows zero


def prototype_scores(prototypes: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
    """Cosine between queries and prototypes: ``(n_query, n_labels)`` float32.

    Both sides must be unit rows (prototypes from :func:`label_prototypes`
    are; query vectors are the caller's validated query-kind embeddings), so
    the plain dot product is the cosine.
    """
    prototypes = np.asarray(prototypes)
    query_vectors = np.asarray(query_vectors)
    if prototypes.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError(
            "prototype_scores expects 2-D matrices; got "
            f"prototypes {prototypes.shape}, query {query_vectors.shape}"
        )
    if prototypes.dtype != np.float32 or query_vectors.dtype != np.float32:
        raise ValueError("prototype_scores expects float32 matrices")
    if prototypes.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            f"dimension mismatch: prototypes {prototypes.shape[1]} vs query "
            f"{query_vectors.shape[1]}"
        )
    return (query_vectors @ prototypes.T).astype(np.float32)
