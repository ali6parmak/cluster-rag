"""Unit tests for the fusion predictor (Phase 4): a·knn + b·proto mixing.

The load-bearing property is the corner equivalence — (a=1, b=0) must
reproduce the kNN predictor exactly and (a=0, b=1) the prototype predictor,
because the weight sweep anchors its ratio grid on them. Fixture geometry as
in tests/test_retrieve.py / tests/test_predictors_retrieval.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor, predictor_embedding_requirements
from cluster_rag.predictors.base import PredictorContext

LABELS = ("A", "B", "C", "Z")
TRAIN_RECORDS = (
    Record(id="r1", text="text r1", language="en", labels=["A", "B"]),
    Record(id="r2", text="text r2", language="en", labels=["A"]),
    Record(id="r3", text="text r3", language="en", labels=["C"]),
)
TRAIN_VECTORS = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
QUERIES = np.asarray([[0.0, 1.0], [0.6, 0.8], [-0.28, -0.96]], dtype=np.float32)


def _context() -> PredictorContext:
    return PredictorContext(
        labels=LABELS, train_records=TRAIN_RECORDS, train_document_vectors=TRAIN_VECTORS
    )


def _scores(name: str, params: dict, queries: np.ndarray) -> np.ndarray:
    predictor = build_predictor(name, params, _context(), seed=0)
    return predictor.score(["q1", "q2", "q3"], query_vectors=queries)


def test_knn_corner_reproduces_the_knn_predictor() -> None:
    """(a=1, b=0) is exactly the kNN voting predictor at the same (k, power)."""
    fused = _scores("fusion", {"a": 1.0, "b": 0.0, "k": 2}, QUERIES)
    knn = _scores("knn", {"k": 2}, QUERIES)
    assert np.array_equal(fused, knn)
    fused_p2 = _scores("fusion", {"a": 1.0, "b": 0.0, "k": 2, "power": 2.0}, QUERIES)
    knn_p2 = _scores("knn", {"k": 2, "power": 2.0}, QUERIES)
    assert np.array_equal(fused_p2, knn_p2)


def test_prototype_corner_reproduces_the_prototype_predictor() -> None:
    """(a=0, b=1) is exactly the prototype (mean) predictor."""
    fused = _scores("fusion", {"a": 0.0, "b": 1.0, "k": 5}, QUERIES)
    proto = _scores("prototype", {}, QUERIES)
    assert np.array_equal(fused, proto)


def test_mixed_scores_are_the_weighted_sum_of_components() -> None:
    """The mix is linear in the components for interior weights."""
    params = {"a": 0.5, "b": 1.5, "k": 2}
    fused = _scores("fusion", params, QUERIES)
    knn = _scores("knn", {"k": 2}, QUERIES)
    proto = _scores("prototype", {}, QUERIES)
    np.testing.assert_allclose(fused, 0.5 * knn + 1.5 * proto, rtol=0, atol=1e-6)


def test_rank_ratio_invariance() -> None:
    """Scaling both weights by a constant cannot change any ranking (top_k)."""
    small = _scores("fusion", {"a": 0.5, "b": 1.0, "k": 10}, QUERIES)
    big = _scores("fusion", {"a": 2.0, "b": 4.0, "k": 10}, QUERIES)
    for row_small, row_big in zip(small, big, strict=False):
        order_small = np.argsort(-row_small, kind="stable")
        order_big = np.argsort(-row_big, kind="stable")
        assert np.array_equal(order_small, order_big)


def test_fusion_requires_positive_total_weight_and_valid_params() -> None:
    """Missing keys, negative/non-numeric weights and a+b=0 all fail early."""
    with pytest.raises(ValueError, match="requires params\\['a'\\]"):
        build_predictor("fusion", {"b": 1.0, "k": 5}, _context(), seed=0)
    with pytest.raises(ValueError, match="requires params\\['k'\\]"):
        build_predictor("fusion", {"a": 1.0, "b": 1.0}, _context(), seed=0)
    with pytest.raises(ValueError, match="non-negative finite number"):
        build_predictor("fusion", {"a": -1.0, "b": 1.0, "k": 5}, _context(), seed=0)
    with pytest.raises(ValueError, match="must be a number"):
        build_predictor("fusion", {"a": "half", "b": 1.0, "k": 5}, _context(), seed=0)
    with pytest.raises(ValueError, match="a \\+ b > 0"):
        build_predictor("fusion", {"a": 0.0, "b": 0.0, "k": 5}, _context(), seed=0)
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("fusion", {"a": 1.0, "b": 1.0, "k": 5, "c": 1.0}, _context(), seed=0)


def test_fusion_unseen_labels_stay_at_zero() -> None:
    """Both components give unseen labels nothing, so the mix does too."""
    scores = _scores("fusion", {"a": 2.0, "b": 1.0, "k": 3}, QUERIES)
    assert np.all(scores[:, LABELS.index("Z")] == 0.0)


def test_fusion_registry_wiring_and_describe() -> None:
    """Registry declares the same embedding needs as its components."""
    assert predictor_embedding_requirements("fusion") == (True, ("document", "query"))
    predictor = build_predictor("fusion", {"a": 0.5, "b": 1.0, "k": 2}, _context(), seed=0)
    assert predictor.describe() == {
        "name": "fusion",
        "a": 0.5,
        "b": 1.0,
        "k": 2,
        "power": 1.0,
        "n_train_examples": 3,
        "dimension": 2,
        "n_labels": 4,
    }
