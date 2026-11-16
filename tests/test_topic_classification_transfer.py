"""Hand-computed tests for the EURLEX-4K label-text transfer helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.transfer import (
    disjoint_hybrid,
    neighbour_transfer,
    row_zscore_columns,
    top_k_by_similarity,
    transfer_reading,
)


def test_top_k_by_similarity_sorts_and_stable_breaks_ties() -> None:
    similarity = np.array(
        [
            [1.0, 0.1, 0.2, 0.9, 0.3],
            [0.1, 1.0, 0.4, 0.2, 0.8],
            [0.2, 0.4, 1.0, 0.7, 0.6],
            [0.9, 0.2, 0.7, 1.0, 0.5],
            [0.3, 0.8, 0.6, 0.5, 1.0],
        ]
    )
    neighbours = top_k_by_similarity(similarity, np.array([0, 1, 2]), np.array([3, 4]), 2)
    # row 3: 0 (0.9) then 2 (0.7); row 4: 1 (0.8) then 2 (0.6)
    assert neighbours.tolist() == [[0, 2], [1, 2]]


def test_top_k_by_similarity_clamps_k_and_rejects_empty_candidates() -> None:
    similarity = np.eye(3)
    neighbours = top_k_by_similarity(similarity, np.array([0, 1]), np.array([2]), 10)
    assert neighbours.shape == (1, 2)
    with pytest.raises(ValueError):
        top_k_by_similarity(similarity, np.array([], dtype=np.int64), np.array([2]), 1)
    with pytest.raises(ValueError):
        top_k_by_similarity(similarity, np.array([0, 1]), np.array([2]), 0)


def test_neighbour_transfer_averages_the_retrieval_scores() -> None:
    retrieval = np.array([[10.0, 20.0, 30.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0, 0.0]])
    neighbours = np.array([[0, 1], [1, 2]])
    transfer = neighbour_transfer(retrieval, neighbours)
    assert transfer.tolist() == [[15.0, 25.0], [1.5, 2.5]]


def test_neighbour_transfer_keeps_the_unseen_axis_with_other_shapes() -> None:
    # Three unseen labels sharing two neighbours each: the mean is over the
    # neighbour axis (2), not over the unseen axis (3).
    retrieval = np.array([[1.0, 3.0, 5.0, 0.0]])
    neighbours = np.array([[0, 1], [1, 2], [0, 2]])
    transfer = neighbour_transfer(retrieval, neighbours)
    assert transfer.shape == (1, 3)
    assert transfer.tolist() == [[2.0, 4.0, 3.0]]


def test_row_zscore_columns_is_standard_normal_or_zero() -> None:
    scores = np.array([[1.0, 2.0, 3.0, 100.0], [5.0, 5.0, 5.0, 5.0]])
    z = row_zscore_columns(scores, np.array([0, 1, 2]))
    assert z[0].mean() == pytest.approx(0.0)
    assert z[0].std() == pytest.approx(1.0)
    assert z[0, 1] == pytest.approx(0.0)
    assert np.allclose(z[1], 0.0)


def test_disjoint_hybrid_keeps_each_label_set_on_its_own_scale() -> None:
    retrieval = np.array([[1.0, 2.0, 3.0, 0.0, 0.0]])
    transfer = np.array([[10.0, 20.0]])
    hybrid = disjoint_hybrid(retrieval, transfer, np.array([3, 4]))
    # seen z of [1, 2, 3]; unseen z of [10, 20]
    assert hybrid[0, 0] == pytest.approx(-3.0 / np.sqrt(6.0))
    assert hybrid[0, 1] == pytest.approx(0.0)
    assert hybrid[0, 2] == pytest.approx(3.0 / np.sqrt(6.0))
    assert hybrid[0, 3] == pytest.approx(-1.0)
    assert hybrid[0, 4] == pytest.approx(1.0)


def test_disjoint_hybrid_validates_shapes() -> None:
    retrieval = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        disjoint_hybrid(retrieval, np.zeros((2, 2)), np.array([1, 1]))
    with pytest.raises(ValueError):
        disjoint_hybrid(retrieval, np.zeros((2, 1)), np.array([1, 2]))
    with pytest.raises(ValueError):
        disjoint_hybrid(retrieval, np.zeros((2, 1)), np.array([4]))


def test_transfer_reading_applies_the_pre_registered_bar() -> None:
    hybrid_pooled = {
        "family_new/label_recall@50/unseen": {"mean": 0.60},
        "family_new/recall@50": {"mean": 0.90},
    }
    title_pooled = {"family_new/label_recall@50/unseen": {"mean": 0.50}}
    lexical_pooled = {"family_new/recall@50": {"mean": 0.905}}
    comparison = {"family_new": {"unseen_r50": {"seeds_ci_positive": 4}}}
    reading = transfer_reading(
        hybrid_pooled=hybrid_pooled,
        title_pooled=title_pooled,
        lexical_pooled=lexical_pooled,
        comparison=comparison,
    )
    assert reading["unseen_gain"] == 0.1
    assert reading["family_new_r50_loss_vs_lexical"] == 0.005
    assert reading["licensed"] is True

    weak = transfer_reading(
        hybrid_pooled={
            "family_new/label_recall@50/unseen": {"mean": 0.51},
            "family_new/recall@50": {"mean": 0.90},
        },
        title_pooled=title_pooled,
        lexical_pooled=lexical_pooled,
        comparison=comparison,
    )
    assert weak["licensed"] is False
