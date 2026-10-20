"""Unit tests for the Phase 4 embedding-dependent predictors (knn, prototype).

Same fixture geometry as tests/test_retrieve.py (orthonormal 2-D train
vectors), wired through the :class:`PredictorContext` train-only boundary. The
registry-level behavior (unknown names, unknown params) lives in
tests/test_predictors.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor, predictor_embedding_requirements
from cluster_rag.predictors.base import PredictorContext

LABELS = ("A", "B", "C", "Z")  # Z is the unseen holdout-label stand-in

TRAIN_RECORDS = (
    Record(id="r1", text="text r1", language="en", labels=["A", "B"]),
    Record(id="r2", text="text r2", language="en", labels=["A"]),
    Record(id="r3", text="text r3", language="en", labels=["C"]),
)
# rows align with TRAIN_RECORDS: e1 = (1, 0), e2 = (0, 1), e3 = (0, -1)
TRAIN_VECTORS = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)


def _context(vectors: np.ndarray | None = TRAIN_VECTORS) -> PredictorContext:
    return PredictorContext(
        labels=LABELS,
        train_records=TRAIN_RECORDS,
        train_document_vectors=None if vectors is None else vectors,
    )


def test_embedding_requirements_are_declared_per_predictor() -> None:
    """The registry exposes needs (evaluate-side wiring reads exactly this)."""
    assert predictor_embedding_requirements("knn") == (True, ("document", "query"))
    assert predictor_embedding_requirements("prototype") == (True, ("document", "query"))
    assert predictor_embedding_requirements("most_frequent") == (False, ())
    with pytest.raises(ValueError, match="Unknown predictor"):
        predictor_embedding_requirements("no_such_predictor")


def test_knn_predictor_votes_match_the_pure_core() -> None:
    """Registry-built knn reproduces the hand-computed weighted votes."""
    predictor = build_predictor("knn", {"k": 2}, _context(), seed=0)
    queries = np.asarray([[0.0, 1.0]], dtype=np.float32)
    scores = predictor.score(["q1"], query_vectors=queries)
    assert scores.shape == (1, 4) and scores.dtype == np.float32
    np.testing.assert_allclose(scores[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert predictor.describe() == {
        "name": "knn",
        "k": 2,
        "power": 1.0,
        "n_train_examples": 3,
        "dimension": 2,
        "n_labels": 4,
    }


def test_knn_scores_depend_on_the_query_embedding() -> None:
    """Different query embeddings vote for different labels (retrieval works)."""
    predictor = build_predictor("knn", {"k": 1}, _context(), seed=0)
    up = np.asarray([[0.0, 1.0]], dtype=np.float32)  # nearest: r2 [A]
    right = np.asarray([[1.0, 0.0]], dtype=np.float32)  # nearest: r1 [A, B]
    assert np.allclose(predictor.score(["q"], query_vectors=up)[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(predictor.score(["q"], query_vectors=right)[0], [1.0, 1.0, 0.0, 0.0])


def test_retrieval_predictors_never_score_unseen_labels() -> None:
    """Z has no train examples: 0 votes and a zero prototype, exactly."""
    knn = build_predictor("knn", {"k": 3}, _context(), seed=0)
    proto = build_predictor("prototype", {}, _context(), seed=0)
    queries = np.asarray([[0.3, 0.95], [-0.7, -0.7]], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    for scores in (
        knn.score(["q1", "q2"], query_vectors=queries.astype(np.float32)),
        proto.score(["q1", "q2"], query_vectors=queries.astype(np.float32)),
    ):
        assert np.all(scores[:, LABELS.index("Z")] == 0.0)


def test_prototype_predictor_matches_hand_cosines() -> None:
    """Registry-built prototype reproduces the renormalized-mean cosines."""
    proto = build_predictor("prototype", {}, _context(), seed=0)
    queries = np.asarray([[0.6, 0.8]], dtype=np.float32)
    scores = proto.score(["q"], query_vectors=queries)
    # prot A = normalize((0.5, 0.5)) -> (0.70711, 0.70711); prot B = (1, 0) (r1 only);
    # prot C = (0, -1) (r3 only); prot Z = 0. cosines: 0.98995, 0.6, -0.8, 0.
    np.testing.assert_allclose(scores[0], [0.70710677 * 1.4, 0.6, -0.8, 0.0], atol=1e-6)
    assert proto.describe() == {
        "name": "prototype",
        "n_train_examples": 3,
        "n_labels": 4,
        "n_labels_with_prototype": 3,
    }


def test_build_knn_without_context_vectors_fails_with_wiring_message() -> None:
    """The failure names the config key that fixes it (eval.embedding_run_id)."""
    with pytest.raises(ValueError, match="embedding_run_id"):
        build_predictor("knn", {"k": 2}, _context(vectors=None), seed=0)
    with pytest.raises(ValueError, match="embedding_run_id"):
        build_predictor("prototype", {}, _context(vectors=None), seed=0)


def test_param_validation() -> None:
    """k is required and validated; unknown params and bad types fail early."""
    with pytest.raises(ValueError, match="requires params\\['k'\\]"):
        build_predictor("knn", {}, _context(), seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        build_predictor("knn", {"k": 0}, _context(), seed=0)
    with pytest.raises(ValueError, match="must be a number"):
        build_predictor("knn", {"k": 2, "power": "sharp"}, _context(), seed=0)
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("knn", {"k": 2, "temperature": 1.0}, _context(), seed=0)
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("prototype", {"k": 2}, _context(), seed=0)


def test_score_time_contract_checks() -> None:
    """Missing query vectors, wrong counts/geometry all fail loudly."""
    knn = build_predictor("knn", {"k": 2}, _context(), seed=0)
    with pytest.raises(ValueError, match="requires query_vectors"):
        knn.score(["q"])
    with pytest.raises(ValueError, match="align row-by-row"):  # 2 texts, 1 vector
        knn.score(["a", "b"], query_vectors=np.asarray([[0.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        knn.score(["a"], query_vectors=np.asarray([[0.0, 1.0]]))
    bad_norm = np.asarray([[0.0, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="unit-norm|L2-normalized"):
        knn.score(["a"], query_vectors=bad_norm)


def test_context_row_alignment_is_enforced_at_construction() -> None:
    """A mismatched matrix cannot sneak past the predictor boundary."""
    short = TRAIN_VECTORS[:2]  # 2 rows for 3 records: misaligned by construction
    with pytest.raises(ValueError, match="align row-by-row"):
        build_predictor("knn", {"k": 2}, _context(vectors=short), seed=0)


def test_determinism_and_seed_irrelevance() -> None:
    """Identical inputs -> identical scores; the seed plays no role here."""
    a = build_predictor("knn", {"k": 2}, _context(), seed=0)
    b = build_predictor("knn", {"k": 2}, _context(), seed=99)
    queries = np.asarray([[0.0, 1.0]], dtype=np.float32)
    first = a.score(["q"], query_vectors=queries)
    assert np.array_equal(first, b.score(["q"], query_vectors=queries))
    assert a.describe() == b.describe()
