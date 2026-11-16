"""Label-text k-NN transfer and the disjoint hybrid (EURLEX-4K TC-21).

In the label-sparse regime most labels have no train example, so no retriever
can score them.  The transfer arm gives an unseen label the **document evidence
of its nearest seen labels in label-text space**: for unseen label ``u``, take
its top-``k`` seen labels by label-text cosine and average those labels'
retrieval scores for the document.

The hybrid keeps the retrieval ranking for the seen labels and the transfer
ranking for the unseen labels, each z-scored **within its own label set** (the
two scales are not comparable).  Both parts are pure numpy; the label-text
similarity (TF-IDF) and the retrieval scores are supplied by the caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TRANSFER_CHUNK = 512


def top_k_positions(block: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` column positions of each row of a similarity block.

    The label-sparse runs already hold the ``(n_unseen, n_seen)`` block (a full
    ``4,197 x 4,197`` matrix is not needed), so this is the block-level
    primitive underneath :func:`top_k_by_similarity`.

    Args:
        block: ``(n_query, n_candidate)`` similarities.
        k: Positions per row; clamped to the block width.

    Returns:
        ``(n_query, k_eff)`` int64 positions, best first, ties broken by
        position (a stable sort).

    Raises:
        ValueError: On a non-2-D block or a non-positive ``k``.
    """
    matrix = np.asarray(block)
    if matrix.ndim != 2:
        raise ValueError(f"block must be 2-D; got {matrix.shape}")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if matrix.shape[1] == 0:
        raise ValueError("the similarity block has no candidate column")
    k_eff = min(k, matrix.shape[1])
    return np.argsort(-matrix, axis=1, kind="stable")[:, :k_eff].astype(np.int64)


def top_k_by_similarity(
    similarity: np.ndarray,
    candidate_columns: np.ndarray,
    query_columns: np.ndarray,
    k: int,
) -> np.ndarray:
    """Each query column's top-``k`` candidate columns by similarity.

    Args:
        similarity: ``(n_labels, n_labels)`` label-to-label cosine matrix.
        candidate_columns: Ascending columns to choose from (e.g. the seen labels).
        query_columns: Ascending columns to choose for (e.g. the unseen labels).
        k: Neighbours per query; clamped to the number of candidates.

    Returns:
        ``(len(query_columns), k_eff)`` candidate column indices, best first,
        ties broken by the candidate's column order (a stable sort).

    Raises:
        ValueError: On a non-2-D similarity, non-positive ``k``, empty columns,
            or a column outside the matrix.
    """
    similarity = np.asarray(similarity)
    candidates = np.asarray(candidate_columns, dtype=np.int64)
    queries = np.asarray(query_columns, dtype=np.int64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must be a square 2-D matrix; got {similarity.shape}")
    if candidates.size == 0:
        raise ValueError("candidate_columns is empty (no seen label to borrow from)")
    if queries.size == 0:
        raise ValueError("query_columns is empty (no unseen label to transfer to)")
    limit = similarity.shape[0]
    if candidates.min() < 0 or candidates.max() >= limit:
        raise ValueError(f"candidate columns must lie in [0, {limit})")
    if queries.min() < 0 or queries.max() >= limit:
        raise ValueError(f"query columns must lie in [0, {limit})")
    block = similarity[np.ix_(queries, candidates)]
    return candidates[top_k_positions(block, k)]


def neighbour_transfer(
    retrieval_scores: np.ndarray,
    neighbour_columns: np.ndarray,
    *,
    chunk: int = TRANSFER_CHUNK,
) -> np.ndarray:
    """Mean retrieval score over each unseen label's neighbour columns.

    Args:
        retrieval_scores: ``(n_documents, n_labels)`` retrieval scores.
        neighbour_columns: ``(n_unseen, k)`` neighbour label columns, best first.
        chunk: Documents per block (the fancy-indexed block is
            ``chunk x n_unseen x k``, so this bounds peak memory).

    Returns:
        ``(n_documents, n_unseen)`` float32 transfer scores.

    Raises:
        ValueError: On a non-2-D input, a mismatched label dimension, a
            non-positive chunk, or empty neighbour columns.
    """
    scores = np.asarray(retrieval_scores)
    neighbours = np.asarray(neighbour_columns, dtype=np.int64)
    if scores.ndim != 2 or neighbours.ndim != 2:
        raise ValueError(f"expected 2-D inputs; got {scores.shape} and {neighbours.shape}")
    if neighbours.size == 0:
        raise ValueError("neighbour_columns is empty")
    if neighbours.max() >= scores.shape[1] or neighbours.min() < 0:
        raise ValueError(f"neighbour columns must lie in [0, {scores.shape[1]})")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk!r}")
    transfer = np.empty((scores.shape[0], neighbours.shape[0]), dtype=np.float32)
    for start in range(0, scores.shape[0], chunk):
        stop = min(start + chunk, scores.shape[0])
        block = scores[start:stop][:, neighbours]
        transfer[start:stop] = block.mean(axis=2)
    return transfer


def row_zscore_columns(scores: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Per-row z-score of ``scores[:, columns]`` (population SD, zero -> zeros).

    Args:
        scores: ``(n_documents, n_labels)`` score matrix.
        columns: Columns to z-score (ascending, unique).

    Returns:
        ``(n_documents, len(columns))`` float32 z-scores.

    Raises:
        ValueError: On non-2-D input or empty columns.
    """
    matrix = np.asarray(scores)
    columns = np.asarray(columns, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be 2-D; got {matrix.shape}")
    if columns.size == 0:
        raise ValueError("columns is empty")
    block = matrix[:, columns].astype(np.float64)
    mean = block.mean(axis=1, keepdims=True)
    std = block.std(axis=1, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return ((block - mean) / std).astype(np.float32)


def disjoint_hybrid(
    retrieval_scores: np.ndarray,
    transfer_scores: np.ndarray,
    unseen_columns: np.ndarray,
    n_labels: int | None = None,
) -> np.ndarray:
    """Seen labels keep the retrieval z-score; unseen labels take the transfer z-score.

    The two rankings are z-scored **within their own label sets**, so neither
    scale dominates the other.  Every label gets a score (the hybrid is a full
    ranking).

    Args:
        retrieval_scores: ``(n_documents, n_labels)`` retrieval scores.
        transfer_scores: ``(n_documents, n_unseen)`` transfer scores aligned to
            ``unseen_columns``.
        unseen_columns: Label columns the transfer scores describe.
        n_labels: Width of the output; defaults to ``retrieval_scores.shape[1]``.

    Returns:
        ``(n_documents, n_labels)`` float32 hybrid scores.

    Raises:
        ValueError: On shape mismatches, duplicate/out-of-range unseen columns,
            or an inconsistent label count.
    """
    retrieval = np.asarray(retrieval_scores)
    transfer = np.asarray(transfer_scores)
    unseen = np.asarray(unseen_columns, dtype=np.int64)
    width = int(retrieval.shape[1] if n_labels is None else n_labels)
    if retrieval.ndim != 2 or transfer.ndim != 2:
        raise ValueError(f"expected 2-D inputs; got {retrieval.shape} and {transfer.shape}")
    if retrieval.shape[1] != width:
        raise ValueError(f"retrieval has {retrieval.shape[1]} labels, expected {width}")
    if transfer.shape != (retrieval.shape[0], unseen.size):
        raise ValueError(
            "transfer must be (n_documents, n_unseen); got "
            f"{transfer.shape} vs ({retrieval.shape[0]}, {unseen.size})"
        )
    if unseen.size and (np.unique(unseen).size != unseen.size):
        raise ValueError("unseen_columns contains duplicates")
    if unseen.size and (unseen.min() < 0 or unseen.max() >= width):
        raise ValueError(f"unseen columns must lie in [0, {width})")
    seen = np.setdiff1d(np.arange(width, dtype=np.int64), unseen)
    hybrid = np.zeros((retrieval.shape[0], width), dtype=np.float32)
    if seen.size:
        hybrid[:, seen] = row_zscore_columns(retrieval, seen)
    if unseen.size:
        hybrid[:, unseen] = row_zscore_columns(transfer, np.arange(unseen.size))
    return hybrid


def transfer_reading(
    *,
    hybrid_pooled: dict[str, dict[str, float]],
    title_pooled: dict[str, dict[str, float]],
    lexical_pooled: dict[str, dict[str, float]],
    comparison: dict[str, Any],
    minimum_gain: float = 0.05,
    maximum_r50_loss: float = 0.01,
    minimum_seeds: int = 4,
) -> dict[str, Any]:
    """Apply the pre-registered Arm A/B bar to one transfer or cluster hybrid.

    Args:
        hybrid_pooled: The pooled (across-seed) metrics of the hybrid.
        title_pooled: The pooled metrics of the title-cosine reference.
        lexical_pooled: The pooled metrics of the lexical k-NN comparator.
        comparison: The ``hybrid__vs__label_text_title`` aggregate, whose
            ``family_new / unseen_r50`` carries the within-seed intervals.
        minimum_gain: The pre-registered unseen micro recall@50 gain.
        maximum_r50_loss: How much `family_new` R@50 may trail the lexical arm.
        minimum_seeds: How many of the five intervals must exclude zero.

    Returns:
        A JSON-serialisable reading with the three conditions and ``licensed``.
    """
    hybrid_unseen = hybrid_pooled["family_new/label_recall@50/unseen"]["mean"]
    title_unseen = title_pooled["family_new/label_recall@50/unseen"]["mean"]
    gain = round(hybrid_unseen - title_unseen, 6)
    seeds_positive = int(comparison["family_new"]["unseen_r50"]["seeds_ci_positive"])
    r50_loss = round(
        lexical_pooled["family_new/recall@50"]["mean"]
        - hybrid_pooled["family_new/recall@50"]["mean"],
        6,
    )
    conditions = {
        "unseen_gain_at_least": gain >= minimum_gain,
        "seeds_ci_excluding_zero_at_least_4": seeds_positive >= minimum_seeds,
        "family_new_r50_loss_at_most_0.01": r50_loss <= maximum_r50_loss,
    }
    return {
        "hybrid_unseen_micro_recall@50": hybrid_unseen,
        "title_unseen_micro_recall@50": title_unseen,
        "unseen_gain": gain,
        "seeds_ci_positive": seeds_positive,
        "family_new_r50_loss_vs_lexical": r50_loss,
        "conditions": conditions,
        "licensed": all(conditions.values()),
    }
