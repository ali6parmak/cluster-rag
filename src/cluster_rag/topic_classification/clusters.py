"""Soft word-cluster bridge for the EURLEX-4K label-sparse regime (TC-21).

The word-cluster arm clusters the **train text** with no labels (TF-IDF + k-means).
A document is soft-assigned to the clusters by the cosine to their centroids; a
label's **text** is soft-assigned the same way.  The score of a (document,
label) pair is the overlap of the two assignments,

``score(d, l) = sum_c membership(d, c) * affinity(l, c)``,

so every label — including one with no train example — gets a score.  The
cluster-level diagnostic asks whether the label's **home cluster** (its argmax
affinity) is one of the document's top-3 clusters.

Pure numpy: the caller supplies the document/centroid and label/centroid
cosines.  The hybrid keeps the retrieval z-score for the seen labels and the
cluster z-score for the unseen labels (the same fixed rule as Arm A).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cluster_rag.topic_classification.transfer import row_zscore_columns


def positive_row_normalize(values: np.ndarray) -> np.ndarray:
    """Clip at 0 and L1-normalise each row into a soft assignment.

    Args:
        values: ``(n, k)`` cosine-like values.

    Returns:
        ``(n, k)`` float32 values with non-negative rows summing to 1 (a row
        with no positive value stays all zeros).

    Raises:
        ValueError: If ``values`` is not 2-D or contains non-finite entries.
    """
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"values must be 2-D; got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("values contain non-finite entries (NaN/inf)")
    positive = np.clip(matrix.astype(np.float64), 0.0, None)
    totals = positive.sum(axis=1, keepdims=True)
    normalized = positive / np.where(totals > 0.0, totals, 1.0)
    return normalized.astype(np.float32)


def cluster_affinity_scores(membership: np.ndarray, affinity: np.ndarray) -> np.ndarray:
    """The soft co-clustering score ``membership @ affinity.T``.

    Args:
        membership: ``(n_documents, n_clusters)`` soft document assignment.
        affinity: ``(n_labels, n_clusters)`` soft label affinity.

    Returns:
        ``(n_documents, n_labels)`` float32 scores.

    Raises:
        ValueError: On non-2-D inputs or mismatched cluster counts.
    """
    documents = np.asarray(membership)
    labels = np.asarray(affinity)
    if documents.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"expected 2-D inputs; got {documents.shape} and {labels.shape}")
    if documents.shape[1] != labels.shape[1]:
        raise ValueError(f"cluster counts differ: {documents.shape[1]} vs {labels.shape[1]}")
    return (documents @ labels.T).astype(np.float32)


def home_clusters(affinity: np.ndarray) -> np.ndarray:
    """Each label's home cluster: the argmax of its affinity row (ties first)."""
    matrix = np.asarray(affinity)
    if matrix.ndim != 2:
        raise ValueError(f"affinity must be 2-D; got {matrix.shape}")
    return np.argmax(matrix, axis=1).astype(np.int64)


def top_clusters(membership: np.ndarray, n: int) -> np.ndarray:
    """Each document's top-``n`` clusters, best first (stable tie-break).

    Args:
        membership: ``(n_documents, n_clusters)`` soft document assignment.
        n: How many clusters per document; clamped to the cluster count.

    Returns:
        ``(n_documents, n_eff)`` int64 cluster indices.

    Raises:
        ValueError: On a non-2-D input or a non-positive ``n``.
    """
    matrix = np.asarray(membership)
    if matrix.ndim != 2:
        raise ValueError(f"membership must be 2-D; got {matrix.shape}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    n_eff = min(n, matrix.shape[1])
    return np.argsort(-matrix, axis=1, kind="stable")[:, :n_eff].astype(np.int64)


def top_cluster_hits(
    gold: Sequence[set[int]], home: np.ndarray, top: np.ndarray
) -> tuple[int, int]:
    """Gold assignments whose home cluster is in the document's top clusters.

    Args:
        gold: One set of gold label indices per document (aligned to ``top``).
        home: ``(n_labels,)`` home-cluster index of each label.
        top: ``(n_documents, n_top)`` the document's top clusters.

    Returns:
        ``(hits, assignments)`` over the gold assignments present in ``gold``.

    Raises:
        ValueError: If the lengths differ or a gold label is out of range.
    """
    home = np.asarray(home)
    top = np.asarray(top)
    if len(gold) != top.shape[0]:
        raise ValueError(f"{len(gold)} gold sets for {top.shape[0]} documents")
    hits = 0
    total = 0
    for row, labels in enumerate(gold):
        if not labels:
            continue
        top_set = set(top[row].tolist())
        for label in labels:
            if label < 0 or label >= home.size:
                raise ValueError(f"gold label {label} outside [0, {home.size})")
            total += 1
            hits += int(int(home[label]) in top_set)
    return hits, total


def cluster_hybrid(
    seen_z: np.ndarray,
    cluster_scores: np.ndarray,
    seen_columns: np.ndarray,
    unseen_columns: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Seen labels take the retrieval z-score; unseen labels the cluster z-score.

    The document side of the cluster score is z-scored **within the unseen
    columns**, matching Arm A's fixed rule, so the two label sets are on one
    scale.

    Args:
        seen_z: ``(n_documents, n_seen)`` retrieval z-scores aligned to
            ``seen_columns``.
        cluster_scores: ``(n_documents, n_labels)`` cluster bridge scores.
        seen_columns: Ascending seen-label columns.
        unseen_columns: Ascending unseen-label columns.
        n_labels: Width of the output.

    Returns:
        ``(n_documents, n_labels)`` float32 hybrid scores.

    Raises:
        ValueError: On mismatched shapes, duplicate columns, or out-of-range
            columns.
    """
    z = np.asarray(seen_z)
    scores = np.asarray(cluster_scores)
    seen = np.asarray(seen_columns, dtype=np.int64)
    unseen = np.asarray(unseen_columns, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != n_labels:
        raise ValueError(f"cluster_scores must be (n_documents, {n_labels}); got {scores.shape}")
    if z.shape != (scores.shape[0], seen.size):
        raise ValueError(f"seen_z must be ({scores.shape[0]}, {seen.size}); got {z.shape}")
    if unseen.size and np.unique(unseen).size != unseen.size:
        raise ValueError("unseen_columns contains duplicates")
    if seen.size and (seen.min() < 0 or seen.max() >= n_labels):
        raise ValueError(f"seen columns must lie in [0, {n_labels})")
    if unseen.size and (unseen.min() < 0 or unseen.max() >= n_labels):
        raise ValueError(f"unseen columns must lie in [0, {n_labels})")
    hybrid = np.zeros((scores.shape[0], n_labels), dtype=np.float32)
    hybrid[:, seen] = z
    if unseen.size:
        block = scores[:, unseen]
        hybrid[:, unseen] = row_zscore_columns(block, np.arange(unseen.size))
    return hybrid
