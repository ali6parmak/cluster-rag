"""Unit tests for the **kNN union** predictor (the track's CS-46).

Two retrievers and their fusion: the union's score matrix must be exactly the
track's own per-row z-max of the **existing** lexical kNN's and the **existing**
embedding kNN's matrices (nothing re-implemented), the embedding half's query
vectors must be required, and the label-blind counterpart must hand **one**
permutation to both halves.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor, predictor_requirements
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.predictors.knn import KNNVotingPredictor
from cluster_rag.predictors.lexical_knn import LexicalKNNPredictor
from cluster_rag.text_processors.decision.upr_union import z_max_union
from cluster_rag.text_processors.lexical.label_blind import permuted_label_sets

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in


def _record(rid: str, text: str, labels: list[str]) -> Record:
    return Record(id=rid, text=text, language="en", labels=labels)


def _context() -> PredictorContext:
    """Train-only context with document vectors aligned to the records."""
    records = (
        _record("r1", "alpha alpha alpha one", ["A"]),
        _record("r2", "alpha alpha two two", ["A"]),
        _record("r3", "alpha three three three", ["A"]),
        _record("r4", "beta beta beta one", ["B"]),
        _record("r5", "beta beta two two", ["B"]),
        _record("r6", "beta three three three", ["B"]),
    )
    # A texts point along the first axis, B texts along the second (unit rows)
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.98, 0.20, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.20, 0.98, 0.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return PredictorContext(labels=LABELS, train_records=records, train_document_vectors=vectors)


def _query() -> np.ndarray:
    """The query vectors of the two test texts (aligned to their rows)."""
    query = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return query / np.linalg.norm(query, axis=1, keepdims=True)


def test_the_union_is_the_z_max_of_the_two_existing_predictors() -> None:
    """CS-46: nothing is re-implemented -- the halves are the existing classes."""
    context = _context()
    union = build_predictor("knn_union", {"k": 3}, context, seed=0)
    lexical = LexicalKNNPredictor(context, 0, {"k": 3})
    embedding = KNNVotingPredictor(context, 0, {"k": 3})
    texts = ["alpha alpha alpha", "beta beta beta"]
    expected = z_max_union(lexical.score(texts), embedding.score(texts, query_vectors=_query()))
    scores = union.score(texts, query_vectors=_query())
    assert scores.shape == (2, 3)
    assert np.allclose(scores, expected.astype(np.float32))
    # the fusion points the right way: each text's own label is on top
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_the_embedding_half_needs_the_query_vectors() -> None:
    """The union is embedding-dependent: without the query vectors it fails loudly."""
    union = build_predictor("knn_union", {"k": 3}, _context(), seed=0)
    with pytest.raises(ValueError, match="query"):
        union.score(["alpha alpha alpha"])


def test_the_label_blind_union_hands_one_permutation_to_both_halves() -> None:
    """The blind union is the two halves on the SAME permuted train labels."""
    context = _context()
    seed = 7
    blind = build_predictor("knn_union", {"k": 3, "label_blind": True}, context, seed=seed)
    permuted = permuted_label_sets([list(record.labels) for record in context.train_records], seed)
    permuted_context = PredictorContext(
        labels=context.labels,
        train_records=tuple(
            record.model_copy(update={"labels": list(label_set)})
            for record, label_set in zip(context.train_records, permuted, strict=True)
        ),
        train_document_vectors=context.train_document_vectors,
    )
    lexical = LexicalKNNPredictor(permuted_context, seed, {"k": 3})
    embedding = KNNVotingPredictor(permuted_context, seed, {"k": 3})
    texts = ["alpha alpha alpha", "beta beta beta"]
    expected = z_max_union(lexical.score(texts), embedding.score(texts, query_vectors=_query()))
    assert np.allclose(blind.score(texts, query_vectors=_query()), expected.astype(np.float32))
    # the control is not a no-op: the blind scores differ from the real ones
    real = build_predictor("knn_union", {"k": 3}, context, seed=seed)
    assert not np.allclose(
        blind.score(texts, query_vectors=_query()), real.score(texts, query_vectors=_query())
    )


def test_the_union_is_deterministic_and_needs_the_embedding_artifact() -> None:
    """The same inputs give the same scores; the registry declares the need."""
    requires, kinds, needs_labels = predictor_requirements("knn_union", {"k": 10})
    assert requires is True and kinds == ("document", "query") and needs_labels is False
    first = build_predictor("knn_union", {"k": 3}, _context(), seed=0)
    second = build_predictor("knn_union", {"k": 3}, _context(), seed=0)
    texts = ["alpha alpha beta"]
    assert np.array_equal(
        first.score(texts, query_vectors=_query()[:1]),
        second.score(texts, query_vectors=_query()[:1]),
    )


def test_the_union_rejects_bad_params() -> None:
    """A missing k, a bad power and an unknown key fail before the fit."""
    with pytest.raises(ValueError, match="requires params\\['k'\\]"):
        build_predictor("knn_union", {}, _context(), seed=0)
    with pytest.raises(ValueError, match="power"):
        build_predictor("knn_union", {"k": 3, "power": "fast"}, _context(), seed=0)
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("knn_union", {"k": 3, "temperature": 1.0}, _context(), seed=0)
