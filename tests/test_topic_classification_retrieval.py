"""Hand-computed tests for the similarity-block k-NN vote scoring (TC-9)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.retrieval import vote_scores_from_similarity


def test_single_query_hand_computed() -> None:
    # train rows: 0 and 1 carry label 0, row 2 carries label 1
    labels = np.array([[1, 0], [1, 0], [0, 1]], dtype=bool)
    similarity = np.array([[0.9, 0.5, 0.2]], dtype=np.float32)
    scores = vote_scores_from_similarity(similarity, labels, k=2)
    assert scores.shape == (1, 2)
    assert scores[0, 0] == pytest.approx(0.9 + 0.5)
    assert scores[0, 1] == pytest.approx(0.0)


def test_negative_similarities_contribute_nothing() -> None:
    labels = np.array([[1], [1]], dtype=bool)
    similarity = np.array([[0.5, -0.9]], dtype=np.float32)
    scores = vote_scores_from_similarity(similarity, labels, k=2)
    assert scores[0, 0] == pytest.approx(0.5)


def test_power_sharpens_the_votes() -> None:
    labels = np.array([[1], [1]], dtype=bool)
    similarity = np.array([[0.9, 0.5]], dtype=np.float32)
    scores = vote_scores_from_similarity(similarity, labels, k=2, power=2.0)
    assert scores[0, 0] == pytest.approx(0.9**2 + 0.5**2)


def test_k_is_clamped_to_the_number_of_train_rows() -> None:
    labels = np.array([[1], [0]], dtype=bool)
    similarity = np.array([[0.9, 0.5]], dtype=np.float32)
    scores = vote_scores_from_similarity(similarity, labels, k=100)
    assert scores[0, 0] == pytest.approx(0.9)


def test_ties_break_to_the_earlier_train_row() -> None:
    labels = np.array([[1], [0]], dtype=bool)
    similarity = np.array([[0.9, 0.9]], dtype=np.float32)
    scores = vote_scores_from_similarity(similarity, labels, k=1)
    assert scores[0, 0] == pytest.approx(0.9)  # row 0 wins the tie, not row 1


def test_validation() -> None:
    labels = np.zeros((2, 1), dtype=bool)
    with pytest.raises(ValueError):
        vote_scores_from_similarity(np.zeros((2, 3), dtype=np.float32), labels, k=1)
    with pytest.raises(ValueError):
        vote_scores_from_similarity(np.zeros((1, 2), dtype=np.float32), labels, k=0)
    with pytest.raises(ValueError):
        vote_scores_from_similarity(np.zeros((1, 2), dtype=np.float32), labels, k=1, power=0)
    with pytest.raises(ValueError):
        vote_scores_from_similarity(np.array([[0.5, np.nan]], dtype=np.float32), labels, k=1)
