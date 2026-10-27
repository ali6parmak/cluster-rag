"""The **union** candidate generator, as a number (the track's CS-14).

The union is the per-row **z-max** of two scorers' score matrices: for every
text, each scorer's scores are z-normalized **within the row** (over the 68
labels) and the union's score is the larger of the two. The union is a
**candidate generator**: the question is how much of the gold a shortlist
built from the union would contain.

**This is a measurement, not a component.** The 2026-10-22 trim put "union
shortlists" out of the track's menu (they are main-plan Phase 4/6
machinery), so nothing here enters the pipeline: the block is a number for
PLAN, and the track's own answer to "would a union help?".

**The control** is the same z-max with the decision model's **label-blind**
scores (the wrong pairing, the same model and the same cost): if the blind
union reaches the same recall, the union's gain is a construction artefact
and the number is reported as such.

Pure core: numpy + the standard library only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cluster_rag.eval.multilabel import ranking_metrics, shortlist_recall


def row_z_scores(scores: np.ndarray) -> np.ndarray:
    """Every row's z-scores (the normalization within the text's label list).

    Args:
        scores: The ``(n_texts, n_labels)`` score matrix.

    Returns:
        The z-scored matrix (the same shape). A row with no spread (all the
        values equal) is scored at zero.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"the score matrix must be 2-dimensional, got {matrix.shape}")
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, keepdims=True)
    safe = np.where(stds > 0, stds, 1.0)
    return (matrix - means) / safe


def z_max_union(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The union's score matrix: the per-row z-max of two scorers.

    Args:
        left: The first scorer's ``(n_texts, n_labels)`` score matrix.
        right: The second scorer's score matrix (the same shape).

    Returns:
        The union's score matrix (the same shape).

    Raises:
        ValueError: If the two matrices' shapes disagree.
    """
    first = row_z_scores(left)
    second = row_z_scores(right)
    if first.shape != second.shape:
        raise ValueError(
            f"the two score matrices must share one shape, got {first.shape} and {second.shape}"
        )
    return np.maximum(first, second)


def _view_block(
    matrix: np.ndarray,
    gold: np.ndarray,
    *,
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
) -> dict[str, Any]:
    """The shortlist and ranking view of one score matrix."""
    truth = np.asarray(gold, dtype=bool)
    values = np.asarray(matrix, dtype=np.float64)
    ranking = ranking_metrics(values, truth, tuple(ks))
    shortlist = shortlist_recall(values, truth, tuple(shortlist_ms))
    n_gold = int(truth.sum())
    order = np.argsort(-values, axis=1, kind="stable")
    top1 = order[:, 0]
    n_gold_top1 = int(truth[np.arange(values.shape[0]), top1].sum())
    return {
        "shortlist_recall": {
            **{f"recall@{m}": float(shortlist.get(f"recall@{m}", 0.0)) for m in shortlist_ms}
        },
        "ranking": {
            **{f"precision@{k}": float(ranking.get(f"precision@{k}", 0.0)) for k in ks},
            **{f"recall@{k}": float(ranking.get(f"recall@{k}", 0.0)) for k in ks},
            **{f"ndcg@{k}": float(ranking.get(f"ndcg@{k}", 0.0)) for k in ks},
        },
        "gold_hit@1": (n_gold_top1 / n_gold) if n_gold else None,
        "n_gold_cells": n_gold,
        "note": (
            "the shortlist recall is the share of the gold cells inside the "
            "text's top-m labels; the ranking view is the same curve seen as "
            "the ranking metrics"
        ),
    }


def upr_union_block(
    champion: np.ndarray,
    decision: np.ndarray,
    gold: np.ndarray,
    *,
    blind_decision: np.ndarray | None = None,
    ks: Sequence[int] = (1, 3, 5, 10),
    shortlist_ms: Sequence[int] = (20, 50, 100),
) -> dict[str, Any]:
    """The union's measurement (the number PLAN asked for).

    Args:
        champion: The champion's ``(n_texts, n_labels)`` score matrix.
        decision: The decision model's score matrix (the same shape).
        gold: The bool gold matrix (the same shape).
        blind_decision: The decision model's **label-blind** score matrix
            (the control).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.

    Returns:
        The block: the union's own numbers, the two references' numbers and
        the blind control's numbers.
    """
    truth = np.asarray(gold, dtype=bool)
    union = z_max_union(champion, decision)
    block: dict[str, Any] = {
        "union": _view_block(union, truth, ks=ks, shortlist_ms=shortlist_ms),
        "champion": _view_block(
            np.asarray(champion, dtype=np.float64), truth, ks=ks, shortlist_ms=shortlist_ms
        ),
        "decision_model": _view_block(
            np.asarray(decision, dtype=np.float64), truth, ks=ks, shortlist_ms=shortlist_ms
        ),
        "n_texts": int(union.shape[0]),
        "n_labels": int(union.shape[1]),
        "n_gold_cells": int(truth.sum()),
        "note": (
            "the union is the per-row z-max of the champion's and the decision "
            "model's scores. It is a MEASUREMENT, not a component: the union is "
            "not built and enters no pipeline (the 2026-10-22 trim put union "
            "shortlists out of the track's menu)."
        ),
    }
    if blind_decision is not None:
        blind_union = z_max_union(champion, blind_decision)
        block["label_blind_control"] = {
            **(_view_block(blind_union, truth, ks=ks, shortlist_ms=shortlist_ms)),
            "note": (
                "the control: the same z-max with the decision model's "
                "label-blind scores (the wrong pairing). If the blind union "
                "reaches the same recall, the union's gain is a construction "
                "artefact."
            ),
        }
    return block
