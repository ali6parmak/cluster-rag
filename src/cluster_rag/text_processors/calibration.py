"""The Stage-3 reporting's pure core: calibration, thresholds and margins.

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3, "Reporting") asks for
*"calibration curves + thresholds per document; pooled and per-document macro;
both negative subsets separately"*, and section 7's **CS-6e** adds the
**safe-negative margin** measured against the **control-role floor** — the
track's durable deliverable (section 8: *"a measured 'which unjudged pairs
are safe negatives' rule"*).

This module holds the pure pieces of that reporting:

- the **reliability diagram** (the predicted probability against the observed
  positive rate, with the bin counts), the **expected calibration error** and
  the **Brier score**;
- the **operating point** of a threshold (the counts and the rates), and the
  **fit-side choice** of a threshold (the candidate grid, the criterion's
  value per candidate and the argmax);
- the **margin** of a cell inside its own paragraph's candidate list — how far
  below the paragraph's best candidate it sits — and the **counts** of a
  margin rule (how many unjudged cells it clears, how many verified positives
  it would discard).

**Why a margin and not an absolute cut.** The pairwise file's hard negatives
sit *above* the verified positives' 95th percentile (the label anomaly of
`TEXT_PROCESSING_PLAN.md` section 0.1), so no absolute score cut separates
them; the track's rules therefore design around it with **rank and margin
rules, never an absolute cut**. A margin is also *relative*: it is measured
inside the paragraph's own candidate list, so a document's score level never
enters the rule.

**The two negative subsets are never pooled.** The functions here take the
positive and the negative side as **separate masks** (the verified positives
against the reviewer-verified hard negatives, or against the unjudged-proxy
control cells), so a caller cannot mix them by accident.

Pure core (AGENTS.md dependency direction): numpy and the standard library
only — no model client, no I/O, no global state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# The default number of the reliability diagram's bins (equal width over
# [0, 1]). Ten is the usual choice and the artifact records the number.
DEFAULT_RELIABILITY_BINS = 10

# The criteria a fit-side threshold can be chosen by (the config knob
# ``analysis.stage3.reporting.threshold_criterion``). Both maximize the
# *Youden-style* separation ``TPR - FPR`` on the fit side; they differ in the
# negative class the FPR is measured on.
THRESHOLD_CRITERION_FLOOR_GAP = "floor_gap"
THRESHOLD_CRITERION_YOUDEN = "youden"
THRESHOLD_CRITERIA: tuple[str, ...] = (
    THRESHOLD_CRITERION_FLOOR_GAP,
    THRESHOLD_CRITERION_YOUDEN,
)

# The description of each criterion (recorded in the artifact).
THRESHOLD_CRITERION_DESCRIPTIONS: dict[str, str] = {
    THRESHOLD_CRITERION_FLOOR_GAP: (
        "the fit side's Youden separation between the verified positives and "
        "the unjudged control cells: TPR(positives) - FPR(ordinary cells). The "
        "ordinary cells are the rule's target negative class (they are the "
        "unjudged space the safe-negative rule mines), and they are presumed "
        "negatives, never ground truth."
    ),
    THRESHOLD_CRITERION_YOUDEN: (
        "the fit side's Youden separation between the verified positives and "
        "the reviewer-verified hard negatives: TPR(positives) - FPR(verified "
        "negatives). This is the reviewed set's own discrimination, whose "
        "negative class is ground truth."
    ),
}


def reliability_diagram(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> dict[str, Any]:
    """The reliability diagram: the predicted probability against the observed rate.

    The bins are **equal width** over ``[0, 1]``; a probability of exactly 1.0
    falls into the last bin. Every bin reports its count, the mean predicted
    probability and the observed positive rate, so a reader can see both the
    curve and how many rows each point of the curve stands for.

    Args:
        probabilities: The predicted probabilities.
        labels: The 0/1 labels the probabilities are checked against.
        n_bins: The number of the bins.

    Returns:
        The diagram: the per-bin records, the expected calibration error, the
        Brier score and the number of rows.

    Raises:
        ValueError: If the arrays disagree in length, a label is not 0/1, or
            ``n_bins`` is smaller than two.
    """
    predicted = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if predicted.shape != targets.shape:
        raise ValueError(
            f"the probabilities have shape {predicted.shape} and the labels {targets.shape}"
        )
    if n_bins < 2:
        raise ValueError(f"the reliability diagram needs at least two bins, got {n_bins}")
    if targets.size and not set(np.unique(targets).tolist()) <= {0, 1}:
        raise ValueError("the labels must be 0/1")
    total = int(targets.size)
    # the bin index of every row (the last bin includes the probability 1.0)
    indexes = np.clip((predicted * n_bins).astype(np.int64), 0, n_bins - 1)
    bins: list[dict[str, Any]] = []
    error = 0.0
    for index in range(n_bins):
        mask = indexes == index
        count = int(mask.sum())
        mean_predicted = float(predicted[mask].mean()) if count else None
        observed = float(targets[mask].mean()) if count else None
        if count:
            error += (count / total) * abs(float(observed) - float(mean_predicted))
        bins.append(
            {
                "index": index,
                "lo": index / n_bins,
                "hi": (index + 1) / n_bins,
                "n": count,
                "mean_predicted": mean_predicted,
                "observed_rate": observed,
            }
        )
    return {
        "n": total,
        "n_bins": int(n_bins),
        "bins": bins,
        "ece": float(error) if total else None,
        "brier": brier_score(predicted, targets),
        "note": (
            "the reliability diagram: the predicted probability against the "
            "observed positive rate, per equal-width bin over [0, 1], with the "
            "bin's count. The expected calibration error is the count-weighted "
            "mean absolute gap between the two."
        ),
    }


def brier_score(
    probabilities: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
) -> float | None:
    """The Brier score: the mean squared error of the predicted probabilities.

    Args:
        probabilities: The predicted probabilities.
        labels: The 0/1 labels.

    Returns:
        The Brier score, or ``None`` when there is no row.
    """
    predicted = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if predicted.shape != targets.shape:
        raise ValueError(
            f"the probabilities have shape {predicted.shape} and the labels {targets.shape}"
        )
    if not predicted.size:
        return None
    return float(np.mean((predicted - targets) ** 2))


def operating_point(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """The operating point of a threshold: the counts and the rates.

    A row counts as **predicted positive** when its score is at least the
    threshold. The rates are ``None`` when their denominator is zero (a
    degenerate slice), never ``0.0`` — a missing number and a zero number are
    different claims.

    Args:
        scores: The scores of the rows.
        labels: The 0/1 labels of the rows.
        threshold: The cut.

    Returns:
        The counts (``tp``, ``fp``, ``fn``, ``tn``) and the rates
        (``precision``, ``recall``, ``specificity``, ``tpr``, ``fpr``).
    """
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.shape != targets.shape:
        raise ValueError(f"the scores have shape {values.shape} and the labels {targets.shape}")
    predicted = values >= threshold
    positive = targets == 1
    negative = ~positive
    tp = int((positive & predicted).sum())
    fp = int((negative & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((negative & ~predicted).sum())

    def _rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "threshold": float(threshold),
        "n": int(values.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "tpr": _rate(tp, tp + fn),
        "fpr": _rate(fp, fp + tn),
    }


def quantile_candidates(
    values: Sequence[float] | np.ndarray,
    *,
    n_quantiles: int = 99,
    low: float = 0.01,
    high: float = 0.99,
) -> np.ndarray:
    """A candidate grid: the values' own quantiles.

    The grid is built from the **fit side's** values, so a candidate never
    depends on the holdout. The quantiles run from ``low`` to ``high`` and the
    repeated values are dropped, so a degenerate distribution yields a short
    grid instead of a grid of identical candidates.

    Args:
        values: The fit side's values.
        n_quantiles: The number of the quantiles.
        low: The lowest quantile.
        high: The highest quantile.

    Returns:
        The sorted candidate values.

    Raises:
        ValueError: If there is no value, the quantile range is invalid, or
            ``n_quantiles`` is smaller than three.
    """
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        raise ValueError("the candidate grid needs at least one value")
    if n_quantiles < 3:
        raise ValueError(f"the candidate grid needs at least three quantiles, got {n_quantiles}")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"the quantile range must satisfy 0 <= low < high <= 1, got {low}, {high}")
    quantiles = np.linspace(low, high, int(n_quantiles))
    return np.unique(np.quantile(data, quantiles))


def threshold_candidates(
    scores: Sequence[float] | np.ndarray, *, n_quantiles: int = 99
) -> np.ndarray:
    """The candidate grid of a threshold: the scores' own quantiles.

    The grid is built from the **fit side's** scores, so a candidate never
    depends on the holdout. The quantiles run from 1% to 99% and the repeated
    values are dropped, so a degenerate score distribution yields a short
    grid instead of a grid of identical candidates.

    Args:
        scores: The fit side's scores (the positive and the negative side).
        n_quantiles: The number of the quantiles.

    Returns:
        The sorted candidate thresholds.

    Raises:
        ValueError: If there is no score or ``n_quantiles`` is smaller than 3.
    """
    return quantile_candidates(scores, n_quantiles=n_quantiles, low=0.01, high=0.99)


def margin_candidates(
    margins: Sequence[float] | np.ndarray, *, n_quantiles: int = 49
) -> np.ndarray:
    """The candidate grid of a margin rule: the margins' own quantiles.

    The grid runs from 2% to 98%: a margin at the very top of the
    distribution would clear almost nothing and one at the very bottom would
    clear everything, so the extremes carry no information about the rule.

    Args:
        margins: The fit side's margins.
        n_quantiles: The number of the quantiles.

    Returns:
        The sorted candidate margins.
    """
    return quantile_candidates(margins, n_quantiles=n_quantiles, low=0.02, high=0.98)


def youden_curve(
    scores: Sequence[float] | np.ndarray,
    *,
    positives: np.ndarray,
    negatives: np.ndarray,
    candidates: Sequence[float],
) -> list[dict[str, Any]]:
    """The criterion's value at every candidate threshold.

    The criterion is the **Youden separation** ``TPR - FPR``: the share of the
    positives the cut keeps minus the share of the negatives it lets through.
    The two sides are separate masks, so the caller decides which negative
    class it measures (the reviewer-verified hard negatives or the unjudged
    control cells) — the two are never mixed.

    Args:
        scores: The fit side's scores (all of them).
        positives: A boolean mask of the positive rows.
        negatives: A boolean mask of the negative rows.
        candidates: The candidate thresholds.

    Returns:
        One record per candidate: the threshold, the TPR, the FPR and the
        criterion's value.
    """
    values = np.asarray(scores, dtype=np.float64)
    positive_scores = values[positives]
    negative_scores = values[negatives]
    curve: list[dict[str, Any]] = []
    for candidate in candidates:
        threshold = float(candidate)
        tpr = float((positive_scores >= threshold).mean()) if positive_scores.size else None
        fpr = float((negative_scores >= threshold).mean()) if negative_scores.size else None
        value = None if tpr is None or fpr is None else tpr - fpr
        curve.append(
            {
                "threshold": threshold,
                "tpr": tpr,
                "fpr": fpr,
                "value": value,
            }
        )
    return curve


def choose_threshold(
    scores: Sequence[float] | np.ndarray,
    *,
    positives: np.ndarray,
    negatives: np.ndarray,
    candidates: Sequence[float],
) -> dict[str, Any]:
    """The threshold that maximizes the criterion on the fit side.

    The tie-break is deterministic: the **smallest** of the maximizing
    candidates wins (the candidates are sorted, and the first maximum is
    taken), so the same fit side always gives the same threshold.

    Args:
        scores: The fit side's scores (all of them).
        positives: A boolean mask of the positive rows.
        negatives: A boolean mask of the negative rows.
        candidates: The candidate thresholds.

    Returns:
        The chosen threshold, the criterion's value there, the curve and the
        number of candidates.

    Raises:
        ValueError: If no candidate has a usable criterion value.
    """
    curve = youden_curve(scores, positives=positives, negatives=negatives, candidates=candidates)
    usable = [entry for entry in curve if entry["value"] is not None]
    if not usable:
        raise ValueError(
            "the threshold could not be chosen: the fit side has no usable criterion value"
        )
    best = max(usable, key=lambda entry: float(entry["value"]))
    return {
        "threshold": float(best["threshold"]),
        "value": float(best["value"]),
        "tpr": best["tpr"],
        "fpr": best["fpr"],
        "n_candidates": len(curve),
        "curve": curve,
    }


def paragraph_best_scores(
    scores: np.ndarray, candidate_columns: Sequence[Sequence[int]]
) -> np.ndarray:
    """The best score of every paragraph row inside its own candidate list.

    The margin of a cell is measured against **its own paragraph's** best
    candidate, so the paragraph's best score is needed once per score matrix.
    A row with no candidate gets ``-inf`` (its cells have no margin).

    Args:
        scores: The (paragraphs x questions) score matrix.
        candidate_columns: The candidate questions of every paragraph row.

    Returns:
        The best score of every row.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    best = np.full(matrix.shape[0], -np.inf, dtype=np.float64)
    for row, candidates in enumerate(candidate_columns):
        if not len(candidates):
            continue
        best[row] = float(matrix[row, list(candidates)].max())
    return best


def cell_margins(
    scores: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    best_scores: np.ndarray,
) -> np.ndarray:
    """The margin of every cell: how far below its paragraph's best it sits.

    Args:
        scores: The (paragraphs x questions) score matrix.
        rows: The paragraph row of every cell.
        columns: The question column of every cell.
        best_scores: The best score of every paragraph row (see
            :func:`paragraph_best_scores`).

    Returns:
        The margin of every cell (0 for the paragraph's best cell).
    """
    matrix = np.asarray(scores, dtype=np.float64)
    return best_scores[rows] - matrix[rows, columns]


def rule_counts(
    cleared: np.ndarray,
    *,
    unjudged: np.ndarray,
    positives: np.ndarray,
) -> dict[str, Any]:
    """The counts of a rule that declares a set of cells safe.

    A rule declares a cell a **safe negative** when it clears it (the
    ``cleared`` mask). Two numbers describe it:

    - the **coverage**: the share of the **unjudged** cells the rule clears
      (the reach of the rule over the space it is meant to mine);
    - the **harm**: the share of the **verified positives** the rule would
      discard (the only measurable cost of the rule, since the unjudged
      cells' true positive rate is unmeasured).

    Args:
        cleared: A boolean mask of the cells the rule clears.
        unjudged: A boolean mask of the unjudged cells.
        positives: A boolean mask of the verified positive cells.

    Returns:
        The counts and the two rates.
    """
    mask = np.asarray(cleared, dtype=bool)
    n_unjudged = int(np.asarray(unjudged, dtype=bool).sum())
    n_cleared = int((mask & np.asarray(unjudged, dtype=bool)).sum())
    n_positives = int(np.asarray(positives, dtype=bool).sum())
    n_discarded = int((mask & np.asarray(positives, dtype=bool)).sum())
    return {
        "n_unjudged_cells": n_unjudged,
        "n_unjudged_cleared": n_cleared,
        "coverage": (n_cleared / n_unjudged) if n_unjudged else None,
        "n_positives": n_positives,
        "n_positives_discarded": n_discarded,
        "harm": (n_discarded / n_positives) if n_positives else None,
    }


def margin_counts(
    margins: np.ndarray,
    *,
    unjudged: np.ndarray,
    positives: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """The counts of a margin rule at one margin threshold.

    The rule declares a cell a **safe negative** when its margin is at least
    the threshold (the cell sits that far below its own paragraph's best
    candidate).

    Args:
        margins: The margin of every cell.
        unjudged: A boolean mask of the unjudged cells.
        positives: A boolean mask of the verified positive cells.
        threshold: The margin threshold.

    Returns:
        The counts and the two rates.
    """
    counts = rule_counts(
        np.asarray(margins, dtype=np.float64) >= threshold,
        unjudged=unjudged,
        positives=positives,
    )
    return {"margin": float(threshold), **counts}


def margin_curve(
    margins: np.ndarray,
    *,
    unjudged: np.ndarray,
    positives: np.ndarray,
    candidates: Sequence[float],
) -> list[dict[str, Any]]:
    """The rule's counts at every candidate margin.

    Args:
        margins: The margin of every cell.
        unjudged: A boolean mask of the unjudged cells.
        positives: A boolean mask of the verified positive cells.
        candidates: The candidate margins.

    Returns:
        One record per candidate margin (see :func:`margin_counts`).
    """
    return [
        margin_counts(margins, unjudged=unjudged, positives=positives, threshold=float(candidate))
        for candidate in candidates
    ]


def rule_curve(
    cleared: Sequence[np.ndarray] | np.ndarray,
    *,
    unjudged: np.ndarray,
    positives: np.ndarray,
    candidates: Sequence[float],
) -> list[dict[str, Any]]:
    """The rule's counts at every candidate margin (a mask per candidate).

    The two-part rule (the cosine's margin **and** the arm's margin) has no
    single margin, so its curve is given as one cleared mask per candidate.

    Args:
        cleared: One boolean mask per candidate margin.
        unjudged: A boolean mask of the unjudged cells.
        positives: A boolean mask of the verified positive cells.
        candidates: The candidate margins.

    Returns:
        One record per candidate margin (the counts and the two rates).
    """
    return [
        {
            "margin": float(candidate),
            **rule_counts(mask, unjudged=unjudged, positives=positives),
        }
        for candidate, mask in zip(candidates, cleared, strict=True)
    ]


def choose_margin(
    margins: np.ndarray,
    *,
    unjudged: np.ndarray,
    positives: np.ndarray,
    candidates: Sequence[float],
) -> dict[str, Any]:
    """The margin that maximizes the fit side's ``coverage - harm``.

    The criterion is the rule's own trade-off: the reach over the unjudged
    space minus the share of the verified positives the rule would throw
    away. The tie-break is deterministic (the smallest maximizing margin).

    Args:
        margins: The margin of every cell.
        unjudged: A boolean mask of the unjudged cells.
        positives: A boolean mask of the verified positive cells.
        candidates: The candidate margins.

    Returns:
        The chosen margin, the criterion's value there and the curve.

    Raises:
        ValueError: If no candidate has a usable criterion value.
    """
    curve = margin_curve(margins, unjudged=unjudged, positives=positives, candidates=candidates)
    usable = [
        entry for entry in curve if entry["coverage"] is not None and entry["harm"] is not None
    ]
    if not usable:
        raise ValueError(
            "the margin could not be chosen: the fit side has no usable criterion value"
        )
    best = max(usable, key=lambda entry: float(entry["coverage"]) - float(entry["harm"]))
    return {
        "margin": float(best["margin"]),
        "value": float(best["coverage"]) - float(best["harm"]),
        "coverage": best["coverage"],
        "harm": best["harm"],
        "n_candidates": len(curve),
        "curve": curve,
    }


def harm_at_coverage(curve: Sequence[dict[str, Any]], coverage: float) -> float | None:
    """The harm of a rule's curve at a given coverage (linear interpolation).

    The curve is sorted by its coverage and the harm is read at the wanted
    coverage: the two neighbours of the point are interpolated. It is how the
    real arm and the **control-role floor** are compared — at the *same*
    reach, the rule that discards fewer verified positives is the better one.

    Args:
        curve: The curve (the records of :func:`margin_curve`).
        coverage: The wanted coverage.

    Returns:
        The harm at that coverage, or ``None`` when the curve cannot answer
        it (an empty curve or a coverage outside its range).
    """
    points = sorted(
        (
            (float(entry["coverage"]), float(entry["harm"]))
            for entry in curve
            if entry.get("coverage") is not None and entry.get("harm") is not None
        ),
        key=lambda pair: pair[0],
    )
    if not points:
        return None
    if coverage < points[0][0] or coverage > points[-1][0]:
        return None
    for (left_coverage, left_harm), (right_coverage, right_harm) in zip(
        points, points[1:], strict=False
    ):
        if left_coverage <= coverage <= right_coverage:
            if right_coverage == left_coverage:
                return left_harm
            weight = (coverage - left_coverage) / (right_coverage - left_coverage)
            return left_harm + weight * (right_harm - left_harm)
    return points[-1][1]
