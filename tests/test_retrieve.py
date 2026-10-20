"""Unit tests for the retrieval scoring cores (Phase 4): hand-computed cases.

Fixture geometry: train examples sit on orthonormal 2-D axes,
``e1 = (1, 0)`` / ``e2 = (0, 1)`` / ``e3 = (0, -1)``, so cosine similarities are
exactly {0, ±1} or ±1/sqrt(2) — every number below is hand-checkable. Label
matrix: r1=[A, B], r2=[A], r3=[C]; Z never appears (the unseen-label stand-in).
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.retrieve.knn import knn_vote_scores
from cluster_rag.retrieve.prototypes import label_prototypes, prototype_scores

TRAIN = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
# rows: r1 [A, B], r2 [A], r3 [C] — columns ordered (A, B, C)
LABELS = np.asarray([[True, True, False], [True, False, False], [False, False, True]])

Q_UP = np.asarray([[0.0, 1.0]], dtype=np.float32)  # sims: r1 0, r2 1, r3 -1
Q_DIAG = np.asarray([[1.0, 1.0]]) / np.sqrt(2.0)  # sims: r1 = r2 = 1/sqrt2, r3 -1/sqrt2
Q_DIAG = Q_DIAG.astype(np.float32)
Q_RIGHT = np.asarray([[1.0, 0.0]], dtype=np.float32)  # sims: r1 1, r2 0, r3 0


def test_knn_votes_topk_weighted() -> None:
    """k=2 votes from the two nearest examples; unseen-by-neighborhood stays 0."""
    scores = knn_vote_scores(TRAIN, LABELS, Q_UP, k=2)
    assert scores.shape == (1, 3) and scores.dtype == np.float32
    # neighbors r2 (sim 1) then r1 (sim 0): A = 1 + 0, B = 0, C = 0
    np.testing.assert_allclose(scores[0], [1.0, 0.0, 0.0], atol=1e-6)


def test_knn_ties_break_to_the_earlier_train_row() -> None:
    """Equal similarities pick r1 before r2 (stable descending sort)."""
    scores = knn_vote_scores(TRAIN, LABELS, Q_DIAG, k=2)
    # tie -> r1 [A, B] first, then r2 [A]: A = s + s, B = s, C = 0
    s = 1.0 / np.sqrt(2.0)
    assert np.allclose(scores[0], [2 * s, s, 0.0])


def test_knn_power_sharpens_neighbor_weights() -> None:
    """sim**2 on the tied diagonal: weights halve, rankings unchanged."""
    scores = knn_vote_scores(TRAIN, LABELS, Q_DIAG, k=2, power=2.0)
    s2 = 0.5
    assert np.allclose(scores[0], [2 * s2, s2, 0.0])


def test_knn_clamps_neighbor_k_to_n_train() -> None:
    """k larger than the index selects every example (never an error)."""
    scores = knn_vote_scores(TRAIN, LABELS, Q_DIAG, k=10)
    s = 1.0 / np.sqrt(2.0)
    # r3 (sim -s) clipped to 0: votes = [r1, r2, r3(0)]
    assert np.allclose(scores[0], [2 * s, s, 0.0])


def test_knn_negative_similarities_clip_to_zero_even_fractional_power() -> None:
    """A selected negative-similarity neighbor votes 0; fractional power is NaN-safe."""
    scores = knn_vote_scores(TRAIN, LABELS, Q_DIAG, k=3, power=0.5)
    s = (1.0 / np.sqrt(2.0)) ** 0.5
    assert np.isfinite(scores).all()
    assert np.allclose(scores[0], [2 * s, s, 0.0])


def test_knn_single_neighbor_and_label_matrix_drives_votes() -> None:
    """k=1 votes are the nearest neighbor's labels, unweighted by others."""
    scores = knn_vote_scores(TRAIN, LABELS, Q_RIGHT, k=1)
    assert np.allclose(scores[0], [1.0, 1.0, 0.0])  # r1 [A, B] is the nearest


def test_knn_rejects_bad_params_and_shapes() -> None:
    """Contract violations fail loudly, before any scoring."""
    with pytest.raises(ValueError, match="positive integer"):
        knn_vote_scores(TRAIN, LABELS, Q_UP, k=0)
    with pytest.raises(ValueError, match="positive integer"):
        knn_vote_scores(TRAIN, LABELS, Q_UP, k=True)  # bool is not an int here
    with pytest.raises(ValueError, match="positive finite number"):
        knn_vote_scores(TRAIN, LABELS, Q_UP, k=1, power=0.0)
    with pytest.raises(ValueError, match="positive finite number"):
        knn_vote_scores(TRAIN, LABELS, Q_UP, k=1, power=float("nan"))
    with pytest.raises(ValueError, match="2-D"):
        knn_vote_scores(TRAIN[0], LABELS, Q_UP, k=1)  # 1-D train matrix
    with pytest.raises(ValueError, match="n_train matching"):
        knn_vote_scores(TRAIN, LABELS[:2], Q_UP, k=1)
    with pytest.raises(ValueError, match="float32"):
        knn_vote_scores(TRAIN.astype(np.float64), LABELS, Q_UP, k=1)
    with pytest.raises(ValueError, match="float32"):
        knn_vote_scores(TRAIN, LABELS, Q_UP.astype(np.float64), k=1)


PROTO_TRAIN = np.asarray(
    [[0.6, 0.8], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32
)  # r1 [A], r2 [A], r3 [B]
PROTO_LABELS = np.asarray([[True, False], [True, False], [False, True]])


def test_label_prototypes_are_renormalized_means() -> None:
    """Prototype = normalized mean of the label's examples; unit rows except empty."""
    prototypes = label_prototypes(PROTO_TRAIN, PROTO_LABELS)
    assert prototypes.shape == (2, 2) and prototypes.dtype == np.float32
    mean_a = np.asarray([0.3, 0.9]) / np.sqrt(0.9)  # renormalized mean of rows 0+1
    assert np.allclose(prototypes[0], mean_a, atol=1e-6)
    assert np.allclose(prototypes[1], [0.0, -1.0], atol=1e-6)  # single example: itself
    assert np.allclose(np.linalg.norm(prototypes, axis=1), 1.0)


def test_label_prototypes_zero_example_label_is_a_zero_vector() -> None:
    """A label with no train examples gets an all-zero prototype (cosine 0)."""
    matrix = np.zeros((3, 3), dtype=bool)
    matrix[0, 0] = matrix[1, 0] = matrix[2, 1] = True
    prototypes = label_prototypes(PROTO_TRAIN, matrix)
    assert prototypes.shape == (3, 2) and np.allclose(prototypes[2], 0.0)  # Z row is zero
    assert np.allclose(np.linalg.norm(prototypes[:2], axis=1), 1.0)  # A/B rows unit


def test_prototype_scores_are_cosines() -> None:
    """Dot products against unit prototypes are the cosine similarities."""
    prototypes = label_prototypes(PROTO_TRAIN, PROTO_LABELS)
    q = np.asarray([[0.6, 0.8], [0.0, -1.0]], dtype=np.float32)
    scores = prototype_scores(prototypes, q)
    assert scores.shape == (2, 2) and scores.dtype == np.float32
    assert np.allclose(scores[0], [np.sqrt(0.9), -0.8], atol=1e-6)  # cos to A, B
    assert np.allclose(scores[1], [-np.sqrt(0.9), 1.0], atol=1e-6)


def test_prototype_scores_reject_dimension_mismatch() -> None:
    """A (d) mismatch between prototypes and queries fails loudly."""
    prototypes = label_prototypes(PROTO_TRAIN, PROTO_LABELS)
    with pytest.raises(ValueError, match="dimension mismatch"):
        prototype_scores(
            prototypes,
            np.zeros((1, 5), dtype=np.float32),
        )
