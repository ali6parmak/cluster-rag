"""Hand-computed tests for the cardinality-targeted threshold (TC-12)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.thresholding import target_cardinality_threshold


def test_target_one_label_per_document() -> None:
    scores = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.2]], dtype=np.float32)
    threshold = target_cardinality_threshold(scores, 1.0)
    assert threshold == pytest.approx(0.8)
    assert (scores >= threshold).sum(axis=1).mean() == pytest.approx(1.0)


def test_target_half_label_per_document() -> None:
    scores = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.2]], dtype=np.float32)
    threshold = target_cardinality_threshold(scores, 0.5)
    assert threshold == pytest.approx(0.9)
    assert (scores >= threshold).sum(axis=1).mean() == pytest.approx(0.5)


def test_ties_at_the_threshold_are_all_predicted() -> None:
    scores = np.array([[0.5, 0.5, 0.1]], dtype=np.float32)
    threshold = target_cardinality_threshold(scores, 0.6)
    assert threshold == pytest.approx(0.5)
    # the budget was one cell, but the tie predicts both 0.5 entries
    assert int((scores >= threshold).sum()) == 2


def test_zero_budget_predicts_nothing() -> None:
    scores = np.array([[0.9, 0.5]], dtype=np.float32)
    threshold = target_cardinality_threshold(scores, 0.1)  # round(0.2) == 0
    assert threshold == float("inf")
    assert not (scores >= threshold).any()


def test_full_budget_predicts_everything() -> None:
    scores = np.array([[0.9, 0.5]], dtype=np.float32)
    threshold = target_cardinality_threshold(scores, 2.0)
    assert threshold == float("-inf")
    assert (scores >= threshold).all()


def test_validation() -> None:
    with pytest.raises(ValueError):
        target_cardinality_threshold(np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        target_cardinality_threshold(np.array([[np.nan, 0.0]]), 1.0)
    with pytest.raises(ValueError):
        target_cardinality_threshold(np.zeros((1, 2)), 0.0)
