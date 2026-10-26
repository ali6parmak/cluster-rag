"""Tests for the Stage-3 reporting's pure core (CS-6e).

Contracts under test (TEXT_PROCESSING_PLAN.md section 3, Stage 3, "Reporting"):

- the **reliability diagram**: the predicted probability against the observed
  positive rate, with the bin counts, the expected calibration error and the
  Brier score;
- the **operating point** of a threshold: the counts and the rates, with a
  missing rate reported as ``None`` (never ``0.0``);
- the **fit-side choice** of a threshold and of a margin: the criterion's
  value per candidate and the deterministic argmax;
- the **margin** of a cell inside its own paragraph's candidate list, and the
  rule's counts (the coverage over the unjudged cells, the harm on the
  verified positives);
- the two negative subsets are separate masks everywhere: a function never
  pools them on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.text_processors.calibration import (
    brier_score,
    cell_margins,
    choose_margin,
    choose_threshold,
    harm_at_coverage,
    margin_candidates,
    margin_counts,
    operating_point,
    paragraph_best_scores,
    quantile_candidates,
    reliability_diagram,
    rule_counts,
    threshold_candidates,
)


def test_reliability_diagram_reports_the_bins_and_their_counts() -> None:
    """A hand-computed diagram: two bins, one perfect and one perfect."""
    probabilities = [0.05, 0.95, 0.96, 0.04]
    labels = [0, 1, 1, 0]
    diagram = reliability_diagram(probabilities, labels, n_bins=2)
    assert diagram["n"] == 4
    assert diagram["n_bins"] == 2
    low, high = diagram["bins"]
    assert (low["lo"], low["hi"]) == (0.0, 0.5)
    assert (high["lo"], high["hi"]) == (0.5, 1.0)
    assert low["n"] == 2
    assert high["n"] == 2
    # the low bin holds the two negatives, the high bin the two positives
    assert low["observed_rate"] == 0.0
    assert high["observed_rate"] == 1.0
    assert low["mean_predicted"] == pytest.approx(0.045)
    assert high["mean_predicted"] == pytest.approx(0.955)
    # the bins are well separated but not perfectly calibrated: the low bin
    # predicts 0.045 where the observed rate is 0, and the high bin predicts
    # 0.955 where it is 1 -> the error is 0.045 in both bins
    assert diagram["ece"] == pytest.approx(0.045)
    # the Brier score: (0.05^2 + 0.95^2 + 0.96^2 + 0.04^2) / 4
    assert diagram["brier"] == pytest.approx((0.0025 + 0.0025 + 0.0016 + 0.0016) / 4)


def test_reliability_diagram_measures_a_miscalibration() -> None:
    """A predicted 0.5 on a positive row is off by 0.5."""
    diagram = reliability_diagram([0.5], [1], n_bins=2)
    assert diagram["ece"] == pytest.approx(0.5)
    assert diagram["brier"] == pytest.approx(0.25)


def test_reliability_diagram_rejects_bad_input() -> None:
    """The arrays must agree, the labels must be 0/1 and the bins at least two."""
    with pytest.raises(ValueError, match="shape"):
        reliability_diagram([0.1, 0.2], [1])
    with pytest.raises(ValueError, match="0/1"):
        reliability_diagram([0.1], [2])
    with pytest.raises(ValueError, match="at least two bins"):
        reliability_diagram([0.1], [1], n_bins=1)


def test_brier_score_of_an_empty_row_is_none() -> None:
    """A missing number is ``None``, never a zero."""
    assert brier_score([], []) is None


def test_operating_point_counts_and_rates() -> None:
    """A hand-computed operating point."""
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    point = operating_point(scores, labels, 0.5)
    assert (point["tp"], point["fp"], point["fn"], point["tn"]) == (2, 0, 0, 2)
    assert point["precision"] == 1.0
    assert point["recall"] == 1.0
    assert point["specificity"] == 1.0
    assert point["fpr"] == 0.0


def test_operating_point_reports_a_missing_rate_as_none() -> None:
    """A slice with no positive row has no recall: ``None``, not ``0.0``."""
    point = operating_point([0.9, 0.1], [0, 0], 0.5)
    assert point["recall"] is None
    # the precision is defined (0 of the 1 predicted positives is right)
    assert point["precision"] == 0.0
    assert point["tn"] == 1


def test_threshold_candidates_are_the_scores_own_quantiles() -> None:
    """The grid comes from the values themselves, so it is fit-side only."""
    values = np.arange(100, dtype=np.float64)
    grid = threshold_candidates(values, n_quantiles=9)
    assert grid.min() == pytest.approx(np.quantile(values, 0.01))
    assert grid.max() == pytest.approx(np.quantile(values, 0.99))
    # the repeated values are dropped, so the grid is strictly increasing
    assert np.all(np.diff(grid) > 0)
    with pytest.raises(ValueError, match="at least one value"):
        threshold_candidates([])
    with pytest.raises(ValueError, match="at least three quantiles"):
        threshold_candidates(values, n_quantiles=2)


def test_quantile_candidates_rejects_a_bad_range() -> None:
    """The quantile range must be a proper interval of [0, 1]."""
    with pytest.raises(ValueError, match="quantile range"):
        quantile_candidates([1.0, 2.0], low=0.9, high=0.1)


def test_choose_threshold_maximizes_the_youden_separation() -> None:
    """The threshold that keeps the positives and drops the negatives."""
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)
    positives = np.array([True, True, True, False, False])
    negatives = ~positives
    choice = choose_threshold(
        scores,
        positives=positives,
        negatives=negatives,
        candidates=np.array([0.15, 0.5, 0.75]),
    )
    # at 0.5: TPR 3/3, FPR 0 -> the best separation
    assert choice["threshold"] == 0.5
    assert choice["value"] == pytest.approx(1.0)
    assert choice["n_candidates"] == 3
    # the curve is recorded per candidate
    assert [entry["threshold"] for entry in choice["curve"]] == [0.15, 0.5, 0.75]


def test_choose_threshold_breaks_a_tie_on_the_smallest_candidate() -> None:
    """The choice is deterministic: the smallest maximizing candidate wins."""
    scores = np.array([1.0, 0.0], dtype=np.float64)
    positives = np.array([True, False])
    choice = choose_threshold(
        scores,
        positives=positives,
        negatives=~positives,
        candidates=np.array([0.0, 0.5, 1.0]),
    )
    # 0.5 and 1.0 separate the two rows perfectly; 0.0 does not (the
    # negative is above it), so the tie is broken on 0.5
    assert choice["threshold"] == 0.5
    assert choice["value"] == pytest.approx(1.0)


def test_choose_threshold_needs_a_usable_criterion() -> None:
    """Without a positive or a negative row there is no criterion value."""
    scores = np.array([1.0, 2.0], dtype=np.float64)
    positives = np.array([False, False])
    with pytest.raises(ValueError, match="no usable criterion value"):
        choose_threshold(
            scores,
            positives=positives,
            negatives=~positives,
            candidates=np.array([0.5]),
        )


def test_paragraph_best_scores_and_cell_margins() -> None:
    """A cell's margin is its distance below its own paragraph's best cell."""
    scores = np.array(
        [
            [0.9, 0.5, 0.1],
            [0.2, 0.8, 0.7],
        ],
        dtype=np.float64,
    )
    candidate_columns = [(0, 1), (1, 2)]
    best = paragraph_best_scores(scores, candidate_columns)
    assert best.tolist() == [0.9, 0.8]
    margins = cell_margins(scores, np.array([0, 0, 1]), np.array([0, 1, 2]), best)
    # the best cell of each row has a margin of zero
    assert margins.tolist() == pytest.approx([0.0, 0.4, 0.1])


def test_rule_counts_separates_the_coverage_from_the_harm() -> None:
    """The coverage is over the unjudged cells, the harm over the positives."""
    cleared = np.array([True, True, True, False])
    unjudged = np.array([True, True, False, True])
    positives = np.array([False, True, True, False])
    counts = rule_counts(cleared, unjudged=unjudged, positives=positives)
    assert counts["n_unjudged_cells"] == 3
    assert counts["n_unjudged_cleared"] == 2
    assert counts["coverage"] == pytest.approx(2 / 3)
    assert counts["n_positives"] == 2
    # two of the positives are cleared (the second and the third row)
    assert counts["n_positives_discarded"] == 2
    assert counts["harm"] == pytest.approx(1.0)


def test_margin_counts_and_candidates() -> None:
    """The margin rule's counts and its fit-side candidate grid."""
    margins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    unjudged = np.array([True, True, True, True, True, False])
    positives = np.array([False, False, True, True, False, False])
    counts = margin_counts(margins, unjudged=unjudged, positives=positives, threshold=0.5)
    assert counts["margin"] == 0.5
    assert counts["n_unjudged_cleared"] == 2
    assert counts["harm"] == pytest.approx(0.5)
    grid = margin_candidates(margins[unjudged], n_quantiles=5)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0


def test_choose_margin_maximizes_the_coverage_minus_the_harm() -> None:
    """The rule's own trade-off: the reach minus the harm."""
    margins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    unjudged = np.array([True] * 9 + [False])
    positives = np.array([False] * 2 + [True] * 3 + [False] * 5)
    choice = choose_margin(
        margins,
        unjudged=unjudged,
        positives=positives,
        candidates=np.array([0.25, 0.45, 0.85]),
    )
    # at 0.45: 4 of the 9 unjudged cells are cleared and none of the
    # positives is -> the criterion is 4/9
    assert choice["margin"] == 0.45
    assert choice["coverage"] == pytest.approx(4 / 9)
    assert choice["harm"] == 0.0


def test_harm_at_coverage_interpolates_the_curve() -> None:
    """The floor's harm at the real arm's reach is read off the curve."""
    curve = [
        {"coverage": 0.2, "harm": 0.4},
        {"coverage": 0.6, "harm": 0.2},
        {"coverage": 0.9, "harm": 0.1},
    ]
    assert harm_at_coverage(curve, 0.6) == pytest.approx(0.2)
    assert harm_at_coverage(curve, 0.4) == pytest.approx(0.3)
    # outside the curve's range there is no answer
    assert harm_at_coverage(curve, 0.95) is None
    assert harm_at_coverage([], 0.5) is None
