"""The UPR safe-negative rule and the shortlist rerank (the track's CS-9b).

The decision probe (CS-9) measures what a decision model's probability buys
on a **seeded sample** of the val split. Two things the OMCT stage already
has are missing on the UPR side, and this module is their port:

- the **safe-negative rule** in its three forms — the **fixed 0.5 cut**, the
  **fit-side threshold** and the **margin rule** — with the **label-blind
  counterpart** beside each form. The rule's fit side is the **train**
  split's grid, so no operating point is ever fitted on the evaluated split.
- the **shortlist rerank** (the Phase-6 shape): the champion's top-K labels
  per text, re-ordered by the decision model's own probability, with the
  shortlist's own recall as the ceiling.

**The counterpart gets the same operating point** (the CS-7b OMCT precedent):
the label-blind counterpart is the same model, the same prompts and the same
cost with **the wrong pairing**, and the rule's threshold (or margin) is the
one the fit side chose — so the two arms differ in the pairing and in
nothing else. That is what makes the difference between the two a *quality*
comparison rather than an operating-point comparison.

**The honesty rule carries over unchanged.** On UPR the non-gold cells are
the background (fully enumerable: 43,568 cells against 1,312 gold ones on
val), but their *true* positive rate is unmeasured — a cell that is not
recorded as gold may still be a label the dataset does not carry. So the
rule's numbers are a **reach** claim (the coverage over the background) and
a **cost** claim (the share of the gold cells the rule discards), never a
precision claim.

Pure core: numpy + the standard library only (the track's decision-side
rule — no model client, no I/O, no global state).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cluster_rag.text_processors.calibration import (
    choose_margin,
    choose_threshold,
    harm_at_coverage,
    margin_candidates,
    margin_curve,
    rule_counts,
    rule_curve,
    threshold_candidates,
)

# The rule's three forms (the names the artifact and the table carry).
RULE_FIXED_CUT = "fixed_cut"
RULE_FIT_THRESHOLD = "fit_threshold"
RULE_MARGIN = "margin"

# The fixed cut: the calibrated model's own operating point (the CS-7a
# precedent — tev1's ECE 0.089-0.137 on the reviewed set; on UPR the CS-9
# run measured tev1's gold cells' mean probability at 0.7578, i.e. the
# probabilities are *not* compressed, so a fixed 0.5 cut is meaningful).
DEFAULT_FIXED_CUT = 0.5


def upr_margins(scores: np.ndarray) -> np.ndarray:
    """Every cell's margin: how far below **its own text's** best label it sits.

    On UPR every label is a candidate of every text (the grid is the whole
    (text x label) space), so a row's best score is its maximum.

    Args:
        scores: The ``(n_texts, n_labels)`` score matrix.

    Returns:
        The margin matrix (the same shape); the best cell of a row has a
        margin of 0.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"the score matrix must be 2-dimensional, got {values.shape}")
    return values.max(axis=1, keepdims=True) - values


def _curve_for(
    values: np.ndarray,
    truth: np.ndarray,
    candidates: Sequence[float],
) -> list[dict[str, Any]]:
    """The rule's counts at every candidate threshold (the cells below it are cleared)."""
    return rule_curve(
        [values < float(candidate) for candidate in candidates],
        unjudged=~truth,
        positives=truth,
        candidates=candidates,
    )


def upr_rule_block(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    fit_scores: np.ndarray,
    fit_gold: np.ndarray,
    blind_scores: np.ndarray | None = None,
    fixed_cut: float = DEFAULT_FIXED_CUT,
    threshold_quantiles: int = 99,
    margin_quantiles: int = 49,
) -> dict[str, Any]:
    """The safe-negative rule on the UPR grid, in its three forms.

    The rule declares a (text, label) cell a **safe negative** when it
    clears it. Two numbers describe a form:

    - the **coverage** (the reach): the share of the **background** cells
      (the cells the dataset does not record as gold) the rule clears;
    - the **harm** (the cost): the share of the **gold** cells the rule
      would discard — the only measurable cost, since the background's true
      positive rate is unmeasured.

    The three forms:

    - the **fixed 0.5 cut**: the calibrated model's own operating point;
    - the **fit-side threshold**: the same cut, fitted on the **train** grid
      (the fit side) and applied to the evaluated split;
    - the **margin rule**: a cell at least X below **its own text's** best
      label, with X fitted on the fit side.

    Beside every form: the **label-blind counterpart** at the **same**
    operating point (the CS-7b OMCT precedent), and the counterpart's harm
    **at the matched coverage** — the comparison at the same reach.

    Args:
        scores: The evaluated split's ``(n_texts, n_labels)`` score matrix.
        gold: The evaluated split's bool gold matrix (the same shape).
        fit_scores: The **fit side's** (the train split's) score matrix.
        fit_gold: The fit side's bool gold matrix (the same shape).
        blind_scores: The label-blind counterpart's score matrix on the
            evaluated split (``None`` when it was not measured).
        fixed_cut: The fixed cut of the first form.
        threshold_quantiles: How many quantiles of the fit side's scores
            form the threshold's candidate grid.
        margin_quantiles: How many quantiles of the fit side's margins form
            the margin rule's candidate grid.

    Returns:
        The rule's block: the three forms, each with its fit side, its
        evaluated side, its curve and its label-blind counterpart.

    Raises:
        ValueError: If the matrices' shapes disagree.
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(gold, dtype=bool)
    if values.shape != truth.shape:
        raise ValueError(f"the scores {values.shape} must match the gold {truth.shape}")
    fit_values = np.asarray(fit_scores, dtype=np.float64)
    fit_truth = np.asarray(fit_gold, dtype=bool)
    if fit_values.shape != fit_truth.shape:
        raise ValueError(
            f"the fit side's scores {fit_values.shape} must match its gold {fit_truth.shape}"
        )
    blind = None if blind_scores is None else np.asarray(blind_scores, dtype=np.float64)
    if blind is not None and blind.shape != values.shape:
        raise ValueError(
            f"the counterpart's scores {blind.shape} must match the real ones {values.shape}"
        )

    background = ~truth
    fit_background = ~fit_truth

    # -- the fixed cut --------------------------------------------------
    fixed_counts = rule_counts(values < fixed_cut, unjudged=background, positives=truth)
    fixed_blind = (
        None
        if blind is None
        else rule_counts(blind < fixed_cut, unjudged=background, positives=truth)
    )

    # -- the fit-side threshold -----------------------------------------
    candidates = threshold_candidates(fit_values.reshape(-1), n_quantiles=threshold_quantiles)
    choice = choose_threshold(
        fit_values.reshape(-1),
        positives=fit_truth.reshape(-1),
        negatives=fit_background.reshape(-1),
        candidates=candidates,
    )
    threshold = float(choice["threshold"])
    threshold_counts = rule_counts(values < threshold, unjudged=background, positives=truth)
    threshold_curve = _curve_for(values, truth, candidates)
    blind_threshold_counts = (
        None
        if blind is None
        else rule_counts(blind < threshold, unjudged=background, positives=truth)
    )
    blind_threshold_curve = None if blind is None else _curve_for(blind, truth, candidates)

    # -- the margin rule -------------------------------------------------
    margins = upr_margins(values)
    fit_margins = upr_margins(fit_values)
    margin_grid = margin_candidates(fit_margins.reshape(-1), n_quantiles=margin_quantiles)
    margin_choice = choose_margin(
        fit_margins,
        unjudged=fit_background,
        positives=fit_truth,
        candidates=margin_grid,
    )
    margin_value = float(margin_choice["margin"])
    margin_counts = rule_counts(margins >= margin_value, unjudged=background, positives=truth)
    margin_eval_curve = margin_curve(
        margins, unjudged=background, positives=truth, candidates=margin_grid
    )
    blind_margins = None if blind is None else upr_margins(blind)
    blind_margin_counts = (
        None
        if blind_margins is None
        else rule_counts(blind_margins >= margin_value, unjudged=background, positives=truth)
    )
    blind_margin_curve = (
        None
        if blind_margins is None
        else margin_curve(
            blind_margins, unjudged=background, positives=truth, candidates=margin_grid
        )
    )

    def _matched(curve: list[dict[str, Any]] | None, coverage: float | None) -> float | None:
        """The counterpart's harm at the real form's coverage."""
        if curve is None or coverage is None:
            return None
        return harm_at_coverage(curve, float(coverage))

    counterpart_note = (
        "the label-blind counterpart at the SAME operating point: the same "
        "model, the same prompts, the wrong pairing (the CS-7b OMCT "
        "precedent). The difference between the two is a quality claim, not "
        "an operating-point one."
    )

    return {
        "space": {
            "n_background_cells": int(background.sum()),
            "n_gold_cells": int(truth.sum()),
            "n_texts": int(values.shape[0]),
            "n_labels": int(values.shape[1]),
            "note": (
                "the background is the cells the dataset does not record as "
                "gold (fully enumerable on UPR). Their TRUE positive rate is "
                "unmeasured, so the rule ships as a reach claim and a cost "
                "claim, never a precision claim."
            ),
        },
        "fit_side": {
            "n_texts": int(fit_values.shape[0]),
            "n_background_cells": int(fit_background.sum()),
            "n_gold_cells": int(fit_truth.sum()),
            "note": (
                "the FIT side is the train split: every operating point is "
                "chosen here, never on the evaluated split"
            ),
        },
        "forms": {
            RULE_FIXED_CUT: {
                "space": (
                    f"the decision model's probability, cut at a FIXED "
                    f"{fixed_cut:g} (the calibrated model's own operating point)"
                ),
                "fit_side": {
                    "cut": float(fixed_cut),
                    "note": "no fitting: the cut is the model's own probability scale",
                },
                "eval_side": fixed_counts,
                "label_blind_counterpart": {
                    "n_draws": 0 if fixed_blind is None else 1,
                    "eval_side": fixed_blind,
                    "note": counterpart_note,
                },
            },
            RULE_FIT_THRESHOLD: {
                "space": (
                    "the decision model's probability, cut at a threshold "
                    "fitted on the fit side (the train grid)"
                ),
                "threshold": threshold,
                "criterion": "youden",
                "fit_side": {
                    "cut": threshold,
                    "criterion": "youden",
                    "criterion_value": choice["value"],
                    "tpr": choice["tpr"],
                    "fpr": choice["fpr"],
                    "n_candidates": choice["n_candidates"],
                    "note": (
                        "the threshold is chosen on the FIT side (the train grid) "
                        "by the Youden separation of the train gold cells from the "
                        "train non-gold cells; the evaluated split never picks it"
                    ),
                },
                "eval_side": threshold_counts,
                "curve": threshold_curve,
                "label_blind_counterpart": {
                    "n_draws": 0 if blind is None else 1,
                    "eval_side": blind_threshold_counts,
                    "curve": blind_threshold_curve,
                    "harm_at_matched_coverage": {
                        "coverage": threshold_counts["coverage"],
                        "harm": _matched(blind_threshold_curve, threshold_counts["coverage"]),
                    },
                    "note": counterpart_note,
                },
            },
            RULE_MARGIN: {
                "space": (
                    "a cell at least X below its own text's best label, with X "
                    "fitted on the fit side (the train grid)"
                ),
                "fit_side": {
                    "margin": margin_value,
                    "criterion": "coverage - harm",
                    "criterion_value": margin_choice["value"],
                    "coverage": margin_choice["coverage"],
                    "harm": margin_choice["harm"],
                    "n_candidates": margin_choice["n_candidates"],
                    "note": (
                        "the margin is chosen on the FIT side by the rule's own "
                        "trade-off (the coverage minus the harm)"
                    ),
                },
                "eval_side": margin_counts,
                "curve": margin_eval_curve,
                "label_blind_counterpart": {
                    "n_draws": 0 if blind is None else 1,
                    "eval_side": blind_margin_counts,
                    "curve": blind_margin_curve,
                    "harm_at_matched_coverage": {
                        "coverage": margin_counts["coverage"],
                        "harm": _matched(blind_margin_curve, margin_counts["coverage"]),
                    },
                    "note": counterpart_note,
                },
            },
        },
        "note": (
            "the rule's three forms with the label-blind counterpart beside "
            "each. The counterpart's harm AT THE MATCHED COVERAGE is the "
            "comparison: at the same reach, the form that discards fewer gold "
            "cells is the better one."
        ),
    }


def upr_shortlist_rerank_block(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    reference_scores: np.ndarray,
    ks: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, Any]:
    """The shortlist rerank on UPR: retrieve with the champion, rank with the decision model.

    The **shortlist** is the champion's top-K labels per text (the geometry's
    own shortlist — the same one PLAN's Phase-6 reranker would consume). The
    **rerank** re-orders that shortlist by the decision model's probability,
    and the metric is the gold cells' **hit@1**: does the true label end up
    first among the text's candidates?

    Three numbers frame it (the CS-7c precedent on OMCT):

    - the **champion's own hit@1** (its first candidate over the whole label
      space): what the geometry does alone;
    - the **decision model's own hit@1** (its first candidate over the whole
      grid): what the decision model does alone, i.e. "rerank everything";
    - the **reranked hit@1**: what the hybrid does.

    The shortlist's own recall@K is the **ceiling** the rerank cannot beat (a
    rerank re-orders the shortlist; it does not enlarge it).

    Args:
        scores: The decision model's ``(n_texts, n_labels)`` score matrix.
        gold: The bool gold matrix (the same shape).
        reference_scores: The champion's score matrix (the same shape).
        ks: The shortlist's K grid.

    Returns:
        The rerank's block: the per-K numbers and the note.
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(gold, dtype=bool)
    reference = np.asarray(reference_scores, dtype=np.float64)
    if values.shape != truth.shape:
        raise ValueError(f"the scores {values.shape} must match the gold {truth.shape}")
    if reference.shape != values.shape:
        raise ValueError(
            f"the reference scores {reference.shape} must match the decision ones {values.shape}"
        )
    n_texts, n_labels = values.shape
    rows = np.arange(n_texts)
    n_gold = int(truth.sum())
    # the champion's own order per text and the decision model's order
    reference_order = np.argsort(-reference, axis=1, kind="stable")
    decision_order = np.argsort(-values, axis=1, kind="stable")
    decision_rank = np.empty((n_texts, n_labels), dtype=np.int64)
    decision_rank[rows[:, None], decision_order] = np.arange(n_labels)[None, :]
    champion_top1 = reference_order[:, 0]
    decision_top1 = decision_order[:, 0]

    per_k: dict[str, Any] = {}
    for k in ks:
        if k < 1 or k > n_labels:
            continue
        shortlist = reference_order[:, :k]
        # the shortlist member the decision model ranks first
        best = shortlist[rows, np.argmin(decision_rank[rows[:, None], shortlist], axis=1)]
        in_shortlist = (shortlist[:, :, None] == np.arange(n_labels)[None, None, :]).any(axis=1)
        n_in_shortlist = int((truth & in_shortlist).sum())
        per_k[str(k)] = {
            "n_gold_cells": n_gold,
            "shortlist_recall": (n_in_shortlist / n_gold) if n_gold else None,
            "champion_hit@1": float(truth[rows, champion_top1].sum() / n_gold) if n_gold else None,
            "decision_hit@1": float(truth[rows, decision_top1].sum() / n_gold) if n_gold else None,
            "rerank_hit@1": float(truth[rows, best].sum() / n_gold) if n_gold else None,
            "n_top1_gold": int(truth[rows, best].sum()),
            "n_top1_cells": int(n_texts),
            "note": (
                "the reranked top-1 is the champion's shortlist member the "
                "decision model ranks first; the shortlist's recall is the "
                "ceiling"
            ),
        }
    return {
        "per_k": per_k,
        "n_texts": int(n_texts),
        "n_labels": int(n_labels),
        "n_gold_cells": n_gold,
        "note": (
            "the shortlist rerank: the champion's top-K per text, re-ordered by "
            "the decision model's probability. The decision model's scores over "
            "the whole grid are already computed, so the rerank costs no "
            "additional calls. This is the Phase-6 shape; it is NOT a union of "
            "the two scorers (a hard shortlist against a soft combination)."
        ),
    }
