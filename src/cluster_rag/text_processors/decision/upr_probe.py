"""The UPR decision probe's pure core (the track's CS-9).

The OMCT track's decision probe (Stage 3.5) asks a decision model about a
**(paragraph, question)** pair. On UPR the question is **per label**: "is this
text about *<label>*?", so the cell is a **(text, label)** cell and the whole
grid is ``n_texts x 68`` cells.

This module holds the UPR-specific pure pieces:

- the **question text** of a label (the label's name — the topic; a UPR label
  has no description and no parent, the repo has no label catalog);
- the **cells** of the probe (the sampled texts x the labels, and the
  **label-blind counterpart** — the same cells with every text's label list
  replaced by a deterministic shift, so the model keeps its shape, its
  calibration and its cost and loses only the pairing);
- the **measurement** of an arm: the UPR harness's own metrics (the decision
  view and the ranking view), the gold side and the background side, the
  **calibration** and the **probability compression**.

The call plan, the ``DecisionCell`` and the synthetic competence control come
from the OMCT probe (:mod:`cluster_rag.text_processors.decision.probe`) — they
are shape-agnostic: a UPR cell is a (text, question) pair like any other.

Pure core: numpy + the standard library only (the track's decision-side rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.eval.multilabel import (
    apply_decision_rule,
    decision_f1_metrics,
    label_coverage,
    ranking_metrics,
    shortlist_recall,
)
from cluster_rag.text_processors.decision.probe import DecisionCell
from cluster_rag.text_processors.upr_diagnostics import (
    cell_rank_percentiles,
    gold_hit_at_k,
    top_fifth_share,
)

# The cell groups (the UPR side's own names).
GROUP_GOLD = "gold"
GROUP_BACKGROUND = "background"


def upr_question_text(label: str) -> str:
    """The question text of a label: the label's own name (the topic).

    A UPR label is a taxonomy term ("Detention", "Enforced disappearances"),
    so the question is the topic itself. The label's *description* and *parent*
    would make a richer question, but the repo has no label catalog (the
    CS-8d gap), so the name is what the probe can ask about.
    """
    return label


def build_upr_cells(
    texts: Sequence[str],
    labels: Sequence[str],
    gold: np.ndarray,
    *,
    mismatched: bool = False,
) -> tuple[DecisionCell, ...]:
    """The probe's cells: every (text, label) cell of the sample.

    The **label-blind counterpart** (``mismatched=True``) replaces every
    text's label list by a **deterministic cyclic shift** of the label space:
    the model keeps its shape, its calibration and its cost, and loses only the
    pairing. It is the yardstick that turns the rule's cost into a quality
    claim (the CS-7b precedent).

    Args:
        texts: The sampled texts (in the sample's order).
        labels: The label space (in the harness's column order).
        gold: The bool gold matrix of the sample (``n_texts x n_labels``).
        mismatched: Whether the cells carry the label-blind counterpart.

    Returns:
        The cells, in the (text, label) order of the sample's grid.

    Raises:
        ValueError: If the gold matrix does not match the texts and labels.
    """
    truth = np.asarray(gold, dtype=bool)
    if truth.shape != (len(texts), len(labels)):
        raise ValueError(
            f"the gold matrix must have shape {(len(texts), len(labels))}, got {truth.shape}"
        )
    cells: list[DecisionCell] = []
    n_labels = len(labels)
    for row, text in enumerate(texts):
        for column in range(n_labels):
            asked = (column + 1) % n_labels if mismatched else column
            cells.append(
                DecisionCell(
                    cell_id=f"upr-{row}-{column}",
                    group=GROUP_GOLD if truth[row, column] else GROUP_BACKGROUND,
                    document="upr",
                    paragraph_row=row,
                    question_col=column,
                    paragraph_text=text,
                    question_text=upr_question_text(labels[asked]),
                    label=int(truth[row, column]),
                )
            )
    return tuple(cells)


def score_matrix(
    cells: Sequence[DecisionCell],
    probabilities: Sequence[float],
) -> np.ndarray:
    """The cells' scores as a ``(n_texts, n_labels)`` matrix.

    The cells are in the grid's own order (the texts' order, the labels'
    order), so the matrix lines up with the gold matrix cell by cell.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size != len(cells):
        raise ValueError(f"the probabilities ({values.size}) must match the cells ({len(cells)})")
    n_texts = max((cell.paragraph_row for cell in cells), default=-1) + 1
    n_labels = max((cell.question_col for cell in cells), default=-1) + 1
    matrix = np.zeros((n_texts, n_labels), dtype=np.float64)
    for cell, value in zip(cells, values, strict=True):
        matrix[cell.paragraph_row, cell.question_col] = value
    return matrix


def probability_compression(scores: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    """How compressed the arm's probability is (measured, not assumed).

    The compression is the gap between the gold cells' mean probability and
    the background's: a model whose gold cells sit at 0.4 answers "no" for most
    true pairs, so a **fixed** cut (0.5) is not the operating point the model
    was trained for (the CS-7b precedent on OMCT).
    """
    truth = np.asarray(gold, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    gold_values = values[truth]
    background_values = values[~truth]
    return {
        "gold_mean": float(gold_values.mean()) if gold_values.size else None,
        "background_mean": float(background_values.mean()) if background_values.size else None,
        "gold_p95": float(np.quantile(gold_values, 0.95)) if gold_values.size else None,
        "background_p95": (
            float(np.quantile(background_values, 0.95)) if background_values.size else None
        ),
        "share_gold_above_0.5": (float((gold_values > 0.5).mean()) if gold_values.size else None),
        "share_background_above_0.5": (
            float((background_values > 0.5).mean()) if background_values.size else None
        ),
        "note": (
            "the probability compression: the gold cells' mean probability "
            "against the background's, and the share of each side above the "
            "fixed 0.5 cut. A compressed arm answers 'no' for most true pairs, "
            "so the fixed cut is not the operating point."
        ),
    }


def measure_upr_arm(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int] = (1, 3, 5, 10),
    shortlist_ms: Sequence[int] = (20, 50, 100),
    bins: int = 10,
    bucket_masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """One arm's measurement on the UPR grid (the harness's own metrics).

    Args:
        scores: The arm's ``(n_texts, n_labels)`` score matrix.
        gold: The bool gold matrix of the same shape.
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        bins: The number of the reliability diagram's bins.
        bucket_masks: The label buckets' column masks (the per-bucket view:
            head / torso / tail / unseen).

    Returns:
        The measurement block: the decision view, the ranking view, the gold
        side, the background side, the calibration and the compression.
    """
    from cluster_rag.text_processors.calibration import reliability_diagram

    truth = np.asarray(gold, dtype=bool)
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.shape != truth.shape:
        raise ValueError(f"the scores {matrix.shape} must match the gold {truth.shape}")
    predicted = apply_decision_rule(
        matrix,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )
    f1 = decision_f1_metrics(predicted, truth)
    coverage = label_coverage(predicted, truth)
    ranking = ranking_metrics(matrix, truth, tuple(ks))
    shortlist = shortlist_recall(matrix, truth, tuple(shortlist_ms))
    percentiles = cell_rank_percentiles(matrix)
    gold_percentiles = percentiles[truth]
    background_percentiles = percentiles[~truth]
    hits = gold_hit_at_k(matrix, truth, (1, 10))
    calibration = reliability_diagram(
        matrix.reshape(-1), truth.reshape(-1).astype(int), n_bins=bins
    )
    block: dict[str, Any] = {
        "decision": {
            "rule": decision["rule"],
            "micro_f1": float(f1["micro_f1"]),
            "macro_f1": float(f1["macro_f1"]),
            "sample_f1": float(f1["sample_f1"]),
            "coverage": float(coverage["coverage"]),
            "recall": float(f1["micro_recall"]),
            "n_predicted_cells": int(f1["n_predicted_cells"]),
            "n_gold_cells": int(f1["n_gold_cells"]),
        },
        "ranking": {
            **{f"precision@{k}": float(ranking.get(f"precision@{k}", 0.0)) for k in ks},
            **{f"recall@{k}": float(ranking.get(f"recall@{k}", 0.0)) for k in ks},
            **{f"ndcg@{k}": float(ranking.get(f"ndcg@{k}", 0.0)) for k in ks},
            **{
                f"shortlist_recall@{m}": float(shortlist.get(f"recall@{m}", 0.0))
                for m in shortlist_ms
            },
        },
        "gold": {
            "n_cells": int(truth.sum()),
            "rank_p50": float(np.median(gold_percentiles)) if gold_percentiles.size else None,
            "top_fifth_share": top_fifth_share(gold_percentiles),
            "hit@1": hits["hit@1"],
            "hit@10": hits["hit@10"],
        },
        "background": {
            "n_cells": int((~truth).sum()),
            "mean": float(matrix[~truth].mean()) if (~truth).any() else None,
            "p50": float(np.median(matrix[~truth])) if (~truth).any() else None,
            "rank_p50": (
                float(np.median(background_percentiles)) if background_percentiles.size else None
            ),
            "top_fifth_share": top_fifth_share(background_percentiles),
        },
        "calibration": {
            "ece": calibration["ece"],
            "brier": calibration["brier"],
            "n_bins": calibration["n_bins"],
            "bins": calibration["bins"],
        },
        "compression": probability_compression(matrix, truth),
    }
    if bucket_masks:
        block["per_bucket"] = {
            name: _bucket_block(matrix, truth, mask, decision=decision, ks=ks)
            for name, mask in bucket_masks.items()
        }
    return block


def _bucket_block(
    matrix: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
) -> dict[str, Any]:
    """One label bucket's view (the columns restricted to the bucket).

    The decision matrix is computed on the **whole** grid (the decision rule
    ranks the labels against each other), and the bucket's view is the
    restriction of it — the same convention the harness's per-bucket report
    uses.
    """
    columns = np.asarray(mask, dtype=bool)
    predicted = apply_decision_rule(
        matrix,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )
    scores = matrix[:, columns]
    gold = truth[:, columns]
    f1 = decision_f1_metrics(predicted[:, columns], gold)
    coverage = label_coverage(predicted, truth, column_mask=columns)
    ranking = ranking_metrics(scores, gold, tuple(ks))
    percentiles = cell_rank_percentiles(scores)
    gold_percentiles = percentiles[gold]
    return {
        "n_labels": int(columns.sum()),
        "n_gold_cells": int(gold.sum()),
        "micro_f1": float(f1["micro_f1"]),
        "macro_f1": float(f1["macro_f1"]),
        "coverage": float(coverage["coverage"]),
        "recall": float(f1["micro_recall"]),
        **{f"ndcg@{k}": float(ranking.get(f"ndcg@{k}", 0.0)) for k in ks},
        "gold_rank_p50": float(np.median(gold_percentiles)) if gold_percentiles.size else None,
        "gold_top_fifth_share": top_fifth_share(gold_percentiles),
    }
