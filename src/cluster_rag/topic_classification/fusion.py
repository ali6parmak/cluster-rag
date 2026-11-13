"""Rank fusion of several k-NN views (EURLEX-4K track, menu item 2).

A multi-view arm keeps one k-NN score matrix per text field and combines their
**rankings** instead of their raw scores: the views' vote sums are not on a
comparable scale, but their orderings are.  The rule is reciprocal-rank fusion
(RRF): each view contributes ``1 / (offset + rank)`` for the labels in its own
top-``top_k`` and nothing beyond, and the fused score is the sum over views.

The parameters are fixed by the caller (the track pre-registers ``top_k`` and
``offset``); nothing here is fitted.  Pure numpy, deterministic tie-break by the
label's column order (the same stable descending sort as the metrics module).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def reciprocal_rank_fusion(
    score_matrices: Sequence[np.ndarray],
    *,
    top_k: int,
    offset: int,
    valid: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Fuse per-view score matrices by reciprocal-rank fusion.

    Args:
        score_matrices: One ``(n_documents, n_labels)`` score matrix per view;
            all must share the same shape. Higher score = better rank.
        top_k: How many of each view's best labels contribute; labels ranked
            below ``top_k`` add nothing.
        offset: The RRF constant: rank ``r`` (1-based) contributes
            ``1 / (offset + r)``.
        valid: Optional per-view ``(n_documents,)`` boolean masks. A row that
            is ``False`` gets no contribution from that view (its text is
            blank and was not embedded).

    Returns:
        The ``(n_documents, n_labels)`` float32 fused score matrix.

    Raises:
        ValueError: On no views, mismatched shapes, or a non-positive
            ``top_k``/``offset``.
    """
    matrices = [np.asarray(matrix) for matrix in score_matrices]
    if not matrices:
        raise ValueError("reciprocal_rank_fusion needs at least one view")
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    shape = matrices[0].shape
    if len(shape) != 2 or any(matrix.shape != shape for matrix in matrices):
        raise ValueError(
            f"every view must be a 2-D matrix of one shape; got {[m.shape for m in matrices]}"
        )
    n_documents, n_labels = shape
    if valid is not None and len(valid) != len(matrices):
        raise ValueError(f"{len(valid)} validity masks for {len(matrices)} views")

    contributions = (1.0 / (offset + np.arange(1, top_k + 1))).astype(np.float32)
    depth = min(top_k, n_labels)
    fused = np.zeros((n_documents, n_labels), dtype=np.float32)
    for view_index, matrix in enumerate(matrices):
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :depth]
        if valid is None:
            rows = np.arange(n_documents)
        else:
            mask = np.asarray(valid[view_index], dtype=bool)
            if mask.shape != (n_documents,):
                raise ValueError(
                    f"view {view_index} validity mask must have shape ({n_documents},); "
                    f"got {mask.shape}"
                )
            rows = np.flatnonzero(mask)
        fused[rows[:, None], order[rows]] += contributions[:depth]
    return fused
