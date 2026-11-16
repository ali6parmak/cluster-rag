"""Hand-computed tests for the EURLEX-4K soft word-cluster bridge helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.clusters import (
    cluster_affinity_scores,
    cluster_hybrid,
    home_clusters,
    positive_row_normalize,
    top_cluster_hits,
    top_clusters,
)


def test_positive_row_normalize_is_a_soft_assignment() -> None:
    values = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    normalized = positive_row_normalize(values)
    assert normalized[0].tolist() == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert normalized[1].tolist() == [0.0, 0.0, 0.0]
    assert normalized[2].tolist() == [0.0, 0.0, 0.0]


def test_positive_row_normalize_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        positive_row_normalize(np.array([[np.nan, 1.0]]))


def test_cluster_affinity_scores_is_a_dot_product() -> None:
    membership = np.array([[1.0, 0.0], [0.5, 0.5]])
    affinity = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    scores = cluster_affinity_scores(membership, affinity)
    assert scores.tolist() == [[1.0, 0.0, 0.5], [0.5, 0.5, 0.5]]


def test_cluster_affinity_scores_validates_cluster_count() -> None:
    with pytest.raises(ValueError):
        cluster_affinity_scores(np.zeros((2, 3)), np.zeros((4, 2)))


def test_home_and_top_clusters() -> None:
    affinity = np.array([[0.1, 0.9, 0.0], [0.4, 0.4, 0.2]])
    assert home_clusters(affinity).tolist() == [1, 0]
    membership = np.array([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]])
    assert top_clusters(membership, 2).tolist() == [[1, 2], [0, 1]]


def test_top_cluster_hits_counts_assignments() -> None:
    gold = [{0, 2}, {1}]
    home = np.array([0, 2, 1])
    top = np.array([[1, 2], [0, 2]])
    hits, total = top_cluster_hits(gold, home, top)
    # doc 0: label 0's home (0) misses, label 2's home (1) hits.
    # doc 1: label 1's home (2) hits.
    assert (hits, total) == (2, 3)


def test_cluster_hybrid_keeps_seen_z_and_zscores_unseen() -> None:
    seen_z = np.array([[1.0, -1.0, 0.0]])
    cluster_scores = np.array([[0.0, 0.0, 0.0, 10.0, 20.0]])
    hybrid = cluster_hybrid(
        seen_z,
        cluster_scores,
        np.array([0, 1, 2]),
        np.array([3, 4]),
        5,
    )
    assert hybrid[0].tolist() == pytest.approx([1.0, -1.0, 0.0, -1.0, 1.0])


def test_cluster_hybrid_validates_shapes() -> None:
    with pytest.raises(ValueError):
        cluster_hybrid(np.zeros((1, 2)), np.zeros((1, 4)), np.array([0, 1]), np.array([2, 3]), 5)
    with pytest.raises(ValueError):
        cluster_hybrid(np.zeros((1, 2)), np.zeros((1, 4)), np.array([0, 1, 2]), np.array([3]), 4)
