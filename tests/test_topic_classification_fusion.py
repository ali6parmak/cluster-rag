"""Hand-computed tests for reciprocal-rank fusion (EURLEX-4K menu item 2)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.fusion import reciprocal_rank_fusion


def test_two_views_hand_computed() -> None:
    # rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 1/3 (offset 0, top_k 3)
    view_a = np.array([[0.9, 0.8, 0.1]])  # ranks: 0, 1, 2
    view_b = np.array([[0.1, 0.9, 0.8]])  # ranks: 1, 2, 0
    fused = reciprocal_rank_fusion([view_a, view_b], top_k=3, offset=0)
    # label 0: 1.0 + 1/3 ; label 1: 0.5 + 1.0 ; label 2: 1/3 + 0.5
    assert fused[0].tolist() == pytest.approx([1.0 + 1.0 / 3.0, 1.5, 1.0 / 3.0 + 0.5])
    assert fused.dtype == np.float32
    assert int(np.argmax(fused[0])) == 1


def test_labels_below_top_k_contribute_nothing() -> None:
    view = np.array([[0.9, 0.8, 0.7]])
    fused = reciprocal_rank_fusion([view], top_k=1, offset=0)
    assert fused[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_offset_changes_the_weights() -> None:
    view = np.array([[0.9, 0.8]])
    fused = reciprocal_rank_fusion([view], top_k=2, offset=60)
    assert fused[0].tolist() == pytest.approx([1.0 / 61.0, 1.0 / 62.0])


def test_blank_view_row_contributes_nothing() -> None:
    view_a = np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
    view_b = np.array([[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]])
    valid = [np.array([True, True]), np.array([True, False])]
    fused = reciprocal_rank_fusion([view_a, view_b], top_k=3, offset=0, valid=valid)
    # row 1 gets nothing from view_b
    assert fused[1].tolist() == pytest.approx([1.0, 0.5, 1.0 / 3.0])


def test_shape_and_parameter_validation() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=1, offset=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([np.zeros((2, 3)), np.zeros((2, 4))], top_k=1, offset=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([np.zeros((2, 3))], top_k=0, offset=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([np.zeros((2, 3))], top_k=1, offset=-1)
