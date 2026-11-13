"""Hand-computed tests for the label co-occurrence helpers (TC-10)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.cooccurrence import (
    cooccurrence_boost,
    cooccurrence_matrix,
)


def test_cooccurrence_matrix_hand_computed() -> None:
    labels = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    matrix = cooccurrence_matrix(labels)
    expected = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-6)


def test_cooccurrence_matrix_ignores_singletons_and_diagonal() -> None:
    labels = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
    matrix = cooccurrence_matrix(labels)
    assert matrix[0].tolist() == [0.0, 0.0, 0.0]
    assert matrix[1, 1] == 0.0
    assert matrix[1, 2] == pytest.approx(1.0)
    assert matrix[2, 1] == pytest.approx(1.0)


def test_boost_hand_computed() -> None:
    scores = np.array([[0.9, 0.8, 0.1]], dtype=np.float32)
    matrix = np.array(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    boost = cooccurrence_boost(scores, matrix, top_m=2)
    weights = np.array([0.9 / 1.7, 0.8 / 1.7])
    expected = weights[0] * matrix[0] + weights[1] * matrix[1]
    np.testing.assert_allclose(boost[0], expected, atol=1e-6)


def test_boost_is_zero_when_the_top_scores_are_zero() -> None:
    scores = np.zeros((2, 3), dtype=np.float32)
    matrix = np.full((3, 3), 0.5, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    boost = cooccurrence_boost(scores, matrix, top_m=2)
    assert not boost.any()


def test_top_m_is_clamped() -> None:
    scores = np.array([[0.9, 0.8, 0.1]], dtype=np.float32)
    matrix = np.array(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    boost = cooccurrence_boost(scores, matrix, top_m=100)  # clamped to 3 labels
    weights = np.array([0.9, 0.8, 0.1]) / 1.8
    expected = weights[0] * matrix[0] + weights[1] * matrix[1] + weights[2] * matrix[2]
    np.testing.assert_allclose(boost[0], expected, atol=1e-6)


def test_validation() -> None:
    with pytest.raises(ValueError):
        cooccurrence_matrix(np.zeros(3))
    scores = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        cooccurrence_boost(scores, np.zeros((2, 3)), top_m=1)
    with pytest.raises(ValueError):
        cooccurrence_boost(scores, np.zeros((3, 3)), top_m=0)
