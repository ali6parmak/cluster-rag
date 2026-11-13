"""Label co-occurrence smoothing for the EURLEX-4K shortlist (menu item 4).

The k-NN votes independently for each label; it never uses the fact that labels
travel together.  This module builds the **train-only** label co-occurrence
distribution and turns a score matrix into a co-occurrence boost: the document's
top-``top_m`` labels (by base score) vote for their conditional neighbours, and
the caller adds ``alpha * z(boost)`` to ``z(base)``.

Definitions, fixed by the experiment's pre-registration:

- ``C[a, b]`` = number of **train** documents carrying both labels ``a`` and
  ``b`` (``a != b``; the diagonal is zero, so a label never boosts itself);
- ``P[b | a] = C[a, b] / sum_b C[a, b]`` (a row-conditional distribution; a
  label with no co-occurrence row keeps an all-zero row);
- the boost of a document is ``sum_{a in top_m(base)} w_a * P[., a]`` with
  ``w`` the top labels' non-negative base scores renormalised to sum to 1.  A
  document whose top-``top_m`` scores are all zero gets an all-zero boost.

Pure numpy; labels with no train example have an all-zero row, so they stay
unreachable (the smoothing never invents a zero-shot candidate).
"""

from __future__ import annotations

import numpy as np


def cooccurrence_matrix(label_matrix: np.ndarray) -> np.ndarray:
    """Row-conditional label co-occurrence from a train label matrix.

    Args:
        label_matrix: ``(n_documents, n_labels)`` multi-hot train indicators.

    Returns:
        The ``(n_labels, n_labels)`` float32 matrix ``P[b | a]`` with a zero
        diagonal; rows with no co-occurrence are all zero.

    Raises:
        ValueError: If ``label_matrix`` is not a 2-D matrix.
    """
    indicators = np.asarray(label_matrix)
    if indicators.ndim != 2:
        raise ValueError(f"label_matrix must be 2-D; got {indicators.shape}")
    n_labels = indicators.shape[1]
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for row in range(indicators.shape[0]):
        present = np.flatnonzero(indicators[row])
        if present.size < 2:
            continue
        rows.append(np.repeat(present, present.size))
        columns.append(np.tile(present, present.size))
    counts = np.zeros((n_labels, n_labels), dtype=np.float32)
    if rows:
        np.add.at(counts, (np.concatenate(rows), np.concatenate(columns)), 1.0)
    np.fill_diagonal(counts, 0.0)
    row_sums = counts.sum(axis=1)
    nonzero = row_sums > 0
    counts[nonzero] /= row_sums[nonzero, None]
    return counts


def cooccurrence_boost(
    scores: np.ndarray,
    matrix: np.ndarray,
    *,
    top_m: int,
) -> np.ndarray:
    """The co-occurrence boost of every document's top-``top_m`` labels.

    Args:
        scores: ``(n_documents, n_labels)`` base scores (higher = better).
        matrix: The ``(n_labels, n_labels)`` co-occurrence distribution.
        top_m: How many top labels per document vote.

    Returns:
        The ``(n_documents, n_labels)`` float32 boost; a row is all zero when
        its top-``top_m`` scores are all non-positive.

    Raises:
        ValueError: On shape mismatches or a non-positive ``top_m``.
    """
    scores = np.asarray(scores)
    matrix = np.asarray(matrix)
    if scores.ndim != 2 or matrix.ndim != 2 or matrix.shape[0] != scores.shape[1]:
        raise ValueError(
            f"expected (n_documents, n_labels) scores and a square matrix; got "
            f"{scores.shape} and {matrix.shape}"
        )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"co-occurrence matrix must be square; got {matrix.shape}")
    if not isinstance(top_m, int) or isinstance(top_m, bool) or top_m < 1:
        raise ValueError(f"top_m must be a positive integer, got {top_m!r}")

    depth = min(top_m, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :depth]
    weights = np.clip(np.take_along_axis(scores, order, axis=1), 0.0, None)
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    voters = np.zeros_like(scores, dtype=np.float32)
    np.put_along_axis(voters, order, weights.astype(np.float32), axis=1)
    return (voters @ matrix).astype(np.float32)
