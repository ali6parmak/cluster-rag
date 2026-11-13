"""Cardinality-targeted score thresholds for the EURLEX-4K set output (menu item 6).

The shortlist ranks labels; the product output is a **label set**.  A global
score threshold adapts the set size per document (1 label here, 17 there) while
a fixed top-k forces every document to the same size.  The threshold is
calibrated to a target mean cardinality — the train split's labels-per-document
mean — by taking the score at the global rank that yields the target number of
predicted (document, label) cells.

Pure numpy; no gold of the evaluated documents is used.
"""

from __future__ import annotations

import numpy as np


def target_cardinality_threshold(
    scores: np.ndarray,
    target_cardinality: float,
) -> float:
    """The global score threshold whose predicted cells match the target mean.

    With ``n`` documents and target ``c``, the predicted-cell budget is
    ``round(n * c)`` and the threshold is the score at that global rank.  Ties
    at the threshold are all predicted, so the achieved mean can exceed the
    target; the caller reports the realised cardinality.

    Args:
        scores: ``(n_documents, n_labels)`` scores (higher = more likely).
        target_cardinality: The desired mean number of labels per document.

    Returns:
        The threshold. ``inf`` when the budget is zero (predict nothing);
        ``-inf`` when the budget covers every cell (predict everything).

    Raises:
        ValueError: On a non-2-D/non-finite score matrix or a non-positive
            target.
    """
    matrix = np.asarray(scores)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be 2-D; got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("scores contain non-finite values (NaN/inf)")
    if not np.isfinite(target_cardinality) or target_cardinality <= 0:
        raise ValueError(f"target_cardinality must be positive; got {target_cardinality!r}")
    n_documents = matrix.shape[0]
    budget = int(round(target_cardinality * n_documents))
    if budget <= 0:
        return float("inf")
    if budget >= matrix.size:
        return float("-inf")
    flat = matrix.ravel()
    # The `budget`-th largest value: its ascending index is size - budget.
    return float(np.partition(flat, flat.size - budget)[flat.size - budget])
